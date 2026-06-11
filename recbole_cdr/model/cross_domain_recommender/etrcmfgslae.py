import torch
import torch.nn.functional as F

from recbole.utils import InputType

from .etrc_modules import EvidentialTransferHead, evidential_classification_loss
from .mfgslaetgf import MFGSLAETGF


class ETRCMFGSLAE(MFGSLAETGF):
    input_type = InputType.PAIRWISE

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
        self.etrc_user_topk = config["etrc_user_topk"]

        self.transfer_head = EvidentialTransferHead(
            input_size=3,
            hidden_size=config["etrc_hidden_size"],
            dropout=config["etrc_dropout"],
            num_classes=self.etrc_num_classes,
        )
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _user_summary_score(self, logits):
        topk = min(int(self.etrc_user_topk), logits.size(1))
        return logits.topk(topk, dim=1).values.mean(dim=1)

    def reliability_features(self, transfer_logits, target_only_logits):
        transfer_score = self._user_summary_score(transfer_logits)
        target_only_score = self._user_summary_score(target_only_logits)
        return torch.stack(
            [
                target_only_score,
                transfer_score,
                transfer_score - target_only_score,
            ],
            dim=-1,
        ).detach()

    def transfer_evidence(self, transfer_logits, target_only_logits):
        alpha, probs, uncertainty = self.transfer_head(self.reliability_features(transfer_logits, target_only_logits))
        reliability = (probs[:, self.HELPFUL] - probs[:, self.HARMFUL]) * (1.0 - uncertainty)
        return alpha, probs, uncertainty, reliability

    def _gate_from_probs(self, probs, reliability):
        gate = (
            self.etrc_lambda_base
            + torch.sigmoid(reliability / max(self.etrc_gate_temperature, 1e-6))
            - self.etrc_harmful_penalty * probs[:, self.HARMFUL]
        )
        return gate.clamp(self.etrc_lambda_min, self.etrc_lambda_max)

    def transfer_gate(self, transfer_logits, target_only_logits):
        alpha, probs, uncertainty, reliability = self.transfer_evidence(transfer_logits, target_only_logits)
        return self._gate_from_probs(probs, reliability), probs, alpha

    def _make_transfer_labels(self, delta):
        labels = torch.full_like(delta, self.NEUTRAL, dtype=torch.long)
        labels[delta > self.etrc_delta_tau] = self.HELPFUL
        labels[delta < -self.etrc_delta_tau] = self.HARMFUL
        return labels

    def _reconstruction_loss_each(self, logits, rating_matrix):
        return -(F.log_softmax(logits, dim=1) * rating_matrix).sum(dim=1)

    def _compactness_and_l1_losses(self):
        if self.factor > 1:
            compactness_loss = self.alpha * (self.compactness(self.h_factor) + self.compactness(self.h_gsl))
        else:
            compactness_loss = torch.tensor(0.0, device=self.device)

        l1_loss = self.l1_rate * (
            F.l1_loss(self.mask_source, torch.zeros_like(self.mask_source, device=self.device))
            + F.l1_loss(self.mask_target, torch.zeros_like(self.mask_target, device=self.device))
        )
        return compactness_loss, l1_loss

    def _gated_target_logits(self, target_user):
        _, transfer_logits = self.forward(target_user=target_user)
        target_only_logits = self.target_only_forward(target_user=target_user)
        gate, probs, alpha = self.transfer_gate(transfer_logits, target_only_logits)
        final_logits = (1.0 - gate.unsqueeze(-1)) * target_only_logits + gate.unsqueeze(-1) * transfer_logits
        return final_logits, transfer_logits, target_only_logits, gate, probs, alpha

    def calculate_loss(self, interaction):
        source_user_id = interaction[self.SOURCE_USER_ID]
        target_user_id = interaction[self.TARGET_USER_ID]

        rating_matrix_source = self.rating_matrix_source[source_user_id.cpu()].to(self.device)
        rating_matrix_target = self.rating_matrix_target[target_user_id.cpu()].to(self.device)

        reconstructed_source, transfer_target = self.forward(source_user_id, target_user_id)
        compactness_loss, l1_loss = self._compactness_and_l1_losses()
        target_only_target = self.target_only_forward(target_user_id)
        gate, probs, alpha = self.transfer_gate(transfer_target, target_only_target)
        final_target = (1.0 - gate.unsqueeze(-1)) * target_only_target + gate.unsqueeze(-1) * transfer_target

        ce_loss_source = self._reconstruction_loss_each(reconstructed_source, rating_matrix_source).mean()
        ce_loss_target = self._reconstruction_loss_each(final_target, rating_matrix_target).mean()

        losses = [
            0.1 * ce_loss_source,
            0.9 * ce_loss_target,
            0.5 * compactness_loss,
            0.2 * l1_loss,
        ]

        with torch.no_grad():
            target_only_loss_each = self._reconstruction_loss_each(target_only_target, rating_matrix_target)
            transfer_loss_each = self._reconstruction_loss_each(transfer_target, rating_matrix_target)
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
            final_loss_each = self._reconstruction_loss_each(final_target, rating_matrix_target)

        if self.etrc_noharm_weight > 0:
            risk_weight = (probs[:, self.HARMFUL] + (1.0 - probs[:, self.HELPFUL])).detach()
            noharm_loss = (risk_weight * F.relu(final_loss_each - target_only_loss_each.detach() + self.etrc_noharm_margin)).mean()
            losses.append(self.etrc_noharm_weight * noharm_loss)

        if self.etrc_tgf_weight > 0:
            overlap_mask = target_user_id < self.overlapped_num_users
            non_overlap_mask = ~overlap_mask
            if overlap_mask.any() and non_overlap_mask.any():
                gain = target_only_loss_each.detach() - final_loss_each
                overlap_gain = gain[overlap_mask].mean()
                non_overlap_gain = gain[non_overlap_mask].mean()
                tgf_loss = F.relu(overlap_gain - non_overlap_gain - self.etrc_tgf_margin).pow(2)
            else:
                tgf_loss = final_target.new_tensor(0.0)
            losses.append(self.etrc_tgf_weight * tgf_loss)

        return tuple(losses)

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        final_logits, _, _, _, _, _ = self._gated_target_logits(user)
        row_index = torch.arange(user.shape[0], device=user.device)
        return final_logits[row_index, item]

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        final_logits, _, _, _, _, _ = self._gated_target_logits(user)
        return final_logits.reshape(-1)
