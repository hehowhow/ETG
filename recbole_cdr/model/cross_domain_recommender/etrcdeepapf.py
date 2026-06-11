import torch
import torch.nn.functional as F

from recbole.utils import InputType

from .deepapf import DeepAPF
from .etrc_modules import EvidentialTransferHead, evidential_classification_loss


class ETRCDeepAPF(DeepAPF):
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
        alpha, probs, uncertainty = self.transfer_head(
            self.reliability_features(transfer_score, target_only_score)
        )
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

    def _bce_each(self, score, label):
        return F.binary_cross_entropy(score.clamp(1e-7, 1.0 - 1e-7), label, reduction="none")

    def _make_transfer_labels(self, delta):
        labels = torch.full_like(delta, self.NEUTRAL, dtype=torch.long)
        labels[delta > self.etrc_delta_tau] = self.HELPFUL
        labels[delta < -self.etrc_delta_tau] = self.HARMFUL
        return labels

    def _predict_from_embeddings(self, user_embedding, item_embedding):
        return self.sigmoid(self.predict_layer(torch.mul(user_embedding, item_embedding))).squeeze(-1)

    def target_transfer_score(self, user, item):
        return self.target_forward(user, item)

    def target_only_score(self, user, item):
        user_embedding = self.target_user_embedding(user)
        item_embedding = self.target_item_embedding(item)
        return self._predict_from_embeddings(user_embedding, item_embedding)

    def _gated_target_score(self, user, item):
        transfer_score = self.target_transfer_score(user, item)
        target_only_score = self.target_only_score(user, item)
        gate, _, _ = self.transfer_gate(transfer_score, target_only_score)
        return (1.0 - gate) * target_only_score + gate * transfer_score

    def _target_overlap_mask(self, user, item):
        if self.mode == "overlap_users":
            return user < self.overlapped_num_users
        return item < self.overlapped_num_items

    def calculate_loss(self, interaction):
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        source_score = self.source_forward(source_user, source_item)
        transfer_score = self.target_transfer_score(target_user, target_item)
        target_only_score = self.target_only_score(target_user, target_item)
        gate, probs, alpha = self.transfer_gate(transfer_score, target_only_score)
        final_score = (1.0 - gate) * target_only_score + gate * transfer_score

        losses = [
            self.loss(source_score, source_label) + self.loss(final_score, target_label)
        ]

        with torch.no_grad():
            target_only_loss_each = self._bce_each(target_only_score, target_label)
            transfer_loss_each = self._bce_each(transfer_score, target_label)
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
            final_loss_each = self._bce_each(final_score, target_label)

        if self.etrc_noharm_weight > 0:
            risk_weight = (probs[:, self.HARMFUL] + (1.0 - probs[:, self.HELPFUL])).detach()
            noharm_loss = (
                risk_weight * F.relu(final_loss_each - target_only_loss_each.detach() + self.etrc_noharm_margin)
            ).mean()
            losses.append(self.etrc_noharm_weight * noharm_loss)

        if self.etrc_tgf_weight > 0:
            overlap_mask = self._target_overlap_mask(target_user, target_item)
            non_overlap_mask = ~overlap_mask
            if overlap_mask.any() and non_overlap_mask.any():
                gain = target_only_loss_each.detach() - final_loss_each
                overlap_gain = gain[overlap_mask].mean()
                non_overlap_gain = gain[non_overlap_mask].mean()
                tgf_loss = F.relu(overlap_gain - non_overlap_gain - self.etrc_tgf_margin).pow(2)
            else:
                tgf_loss = final_score.new_tensor(0.0)
            losses.append(self.etrc_tgf_weight * tgf_loss)

        return tuple(losses)

    def predict_target_only(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        return self.target_only_score(user, item)

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        return self._gated_target_score(user, item)

    def _full_sort_with_score_fn(self, user, score_fn):
        all_item = torch.arange(self.target_num_items, device=user.device)
        score_list = []
        for uid in user:
            batch_user = uid.repeat(self.target_num_items)
            score_list.append(score_fn(batch_user, all_item).view(1, -1))
        return torch.cat(score_list, dim=0)

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        return self._full_sort_with_score_fn(user, self._gated_target_score).view(-1)

    def full_sort_predict_target_only(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        return self._full_sort_with_score_fn(user, self.target_only_score).view(-1)
