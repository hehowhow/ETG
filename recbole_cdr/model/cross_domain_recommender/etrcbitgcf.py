import torch
import torch.nn.functional as F

from recbole.utils import InputType

from .bitgcftgf import BiTGCFTGF
from .etrc_modules import EvidentialTransferHead, evidential_classification_loss


class ETRCBiTGCF(BiTGCFTGF):
    input_type = InputType.POINTWISE

    HELPFUL = 0
    NEUTRAL = 1
    HARMFUL = 2

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.etrc_num_classes = config["etrc_num_classes"]
        self.etrc_edl_weight = config["etrc_edl_weight"]
        self.etrc_noharm_weight = config["etrc_noharm_weight"]
        self.etrc_tgf_weight = config["etrc_tgf_weight"]
        self.etrc_delta_reg_weight = config["etrc_delta_reg_weight"]
        self.etrc_delta_tau = config["etrc_delta_tau"]
        self.etrc_noharm_margin = config["etrc_noharm_margin"]
        self.etrc_kl_anneal_epochs = config["etrc_kl_anneal_epochs"]
        self.etrc_lambda_min = config["etrc_lambda_min"]
        self.etrc_lambda_max = config["etrc_lambda_max"]
        self.etrc_lambda_base = config["etrc_lambda_base"]
        self.etrc_gate_temperature = config["etrc_gate_temperature"]
        self.etrc_harmful_penalty = config["etrc_harmful_penalty"]
        self.etrc_tgf_margin = config["etrc_tgf_margin"]

        self.transfer_head = EvidentialTransferHead(
            input_size=3,
            hidden_size=config["etrc_hidden_size"],
            dropout=config["etrc_dropout"],
            num_classes=self.etrc_num_classes,
        )
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def reliability_features(self, transfer_score, target_only_score):
        transfer_score = transfer_score.reshape(-1)
        target_only_score = target_only_score.reshape(-1)
        return torch.stack(
            [
                target_only_score,
                transfer_score,
                transfer_score - target_only_score,
            ],
            dim=-1,
        ).detach()

    def transfer_evidence(self, transfer_score, target_only_score):
        alpha, probs, uncertainty = self.transfer_head(self.reliability_features(transfer_score, target_only_score))
        reliability = (probs[:, self.HELPFUL] - probs[:, self.HARMFUL]) * (1.0 - uncertainty)
        return alpha, probs, uncertainty, reliability

    def _gate_from_probs(self, probs, reliability):
        gate = (
            self.etrc_lambda_base
            + torch.sigmoid(reliability / max(self.etrc_gate_temperature, 1e-6))
            - self.etrc_harmful_penalty * probs[:, self.HARMFUL]
        )
        return gate.clamp(self.etrc_lambda_min, self.etrc_lambda_max)

    def transfer_gate(self, transfer_score, target_only_score):
        alpha, probs, uncertainty, reliability = self.transfer_evidence(transfer_score, target_only_score)
        return self._gate_from_probs(probs, reliability), probs, alpha

    def _target_logits(self, user_embeddings, item_embeddings, user, item):
        u_embeddings = user_embeddings[user]
        i_embeddings = item_embeddings[item]
        return torch.mul(u_embeddings, i_embeddings).sum(dim=1)

    def _make_transfer_labels(self, delta):
        labels = torch.full_like(delta, self.NEUTRAL, dtype=torch.long)
        labels[delta > self.etrc_delta_tau] = self.HELPFUL
        labels[delta < -self.etrc_delta_tau] = self.HARMFUL
        return labels

    def calculate_loss(self, interaction):
        self.init_restore_e()
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings = self.forward()
        target_only_user_embeddings, target_only_item_embeddings = self.target_only_embeddings()

        losses = []

        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        source_logits = self._target_logits(source_user_all_embeddings, source_item_all_embeddings, source_user, source_item)
        source_output = self.sigmoid(source_logits)
        source_bce_loss = self.loss(source_output, source_label)
        source_reg_loss = self.reg_loss(self.source_user_embedding(source_user), self.source_item_embedding(source_item))
        source_loss = source_bce_loss + self.reg_weight * source_reg_loss
        losses.append(source_loss)

        transfer_logits = self._target_logits(target_user_all_embeddings, target_item_all_embeddings, target_user, target_item)
        target_only_logits = self._target_logits(target_only_user_embeddings, target_only_item_embeddings, target_user, target_item)
        gate, probs, alpha = self.transfer_gate(transfer_logits, target_only_logits)
        final_logits = (1.0 - gate) * target_only_logits + gate * transfer_logits

        final_output = self.sigmoid(final_logits)
        target_bce_loss = self.loss(final_output, target_label)
        target_reg_loss = self.reg_loss(self.target_user_embedding(target_user), self.target_item_embedding(target_item))
        target_loss = target_bce_loss + self.reg_weight * target_reg_loss
        losses.append(target_loss)

        with torch.no_grad():
            target_only_loss_each = F.binary_cross_entropy_with_logits(target_only_logits, target_label, reduction="none")
            transfer_loss_each = F.binary_cross_entropy_with_logits(transfer_logits, target_label, reduction="none")
            delta = target_only_loss_each - transfer_loss_each
            transfer_labels = self._make_transfer_labels(delta)

        edl_loss = evidential_classification_loss(
            alpha,
            transfer_labels,
            epoch=self.current_epoch,
            anneal_epochs=self.etrc_kl_anneal_epochs,
        )
        expected_delta = probs[:, self.HELPFUL] - probs[:, self.HARMFUL]
        delta_reg_loss = F.smooth_l1_loss(expected_delta, delta.detach())
        if self.etrc_edl_weight > 0:
            losses.append(self.etrc_edl_weight * (edl_loss + self.etrc_delta_reg_weight * delta_reg_loss))

        if self.etrc_noharm_weight > 0 or self.etrc_tgf_weight > 0:
            final_loss_each = F.binary_cross_entropy_with_logits(final_logits, target_label, reduction="none")

        if self.etrc_noharm_weight > 0:
            risk_weight = (probs[:, self.HARMFUL] + (1.0 - probs[:, self.HELPFUL])).detach()
            noharm_loss = (risk_weight * F.relu(final_loss_each - target_only_loss_each.detach() + self.etrc_noharm_margin)).mean()
            losses.append(self.etrc_noharm_weight * noharm_loss)

        if self.etrc_tgf_weight > 0:
            overlap_mask = target_user < self.overlapped_num_users
            non_overlap_mask = ~overlap_mask
            if overlap_mask.any() and non_overlap_mask.any():
                gain = target_only_loss_each.detach() - final_loss_each
                overlap_gain = gain[overlap_mask].mean()
                non_overlap_gain = gain[non_overlap_mask].mean()
                tgf_loss = F.relu(overlap_gain - non_overlap_gain - self.etrc_tgf_margin).pow(2)
            else:
                tgf_loss = final_logits.new_tensor(0.0)
            losses.append(self.etrc_tgf_weight * tgf_loss)

        return tuple(losses)

    def predict(self, interaction):
        transfer_score = super().predict(interaction)
        target_only_score = self.predict_target_only(interaction)
        gate, _, _ = self.transfer_gate(transfer_score, target_only_score)
        return (1.0 - gate) * target_only_score + gate * transfer_score

    def _score_level_gate(self, transfer_scores, target_only_scores, chunk_size=65536):
        shape = transfer_scores.shape
        transfer_flat = transfer_scores.reshape(-1)
        target_only_flat = target_only_scores.reshape(-1)
        gate_chunks = []
        for start in range(0, transfer_flat.numel(), chunk_size):
            end = start + chunk_size
            gate, _, _ = self.transfer_gate(transfer_flat[start:end], target_only_flat[start:end])
            gate_chunks.append(gate)
        return torch.cat(gate_chunks, dim=0).view(shape)

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        target_user_all_embeddings, target_item_all_embeddings = self.target_only_embeddings()
        _, _, transfer_user_all_embeddings, transfer_item_all_embeddings = self.forward()

        target_u_embeddings = target_user_all_embeddings[user]
        transfer_u_embeddings = transfer_user_all_embeddings[user]
        target_i_embeddings = target_item_all_embeddings[:self.target_num_items]
        transfer_i_embeddings = transfer_item_all_embeddings[:self.target_num_items]

        target_scores = torch.matmul(target_u_embeddings, target_i_embeddings.transpose(0, 1))
        transfer_scores = torch.matmul(transfer_u_embeddings, transfer_i_embeddings.transpose(0, 1))
        gate = self._score_level_gate(transfer_scores, target_scores)
        scores = (1.0 - gate) * target_scores + gate * transfer_scores
        return scores.view(-1)
