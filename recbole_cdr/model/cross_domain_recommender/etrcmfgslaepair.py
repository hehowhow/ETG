import torch
import torch.nn.functional as F

from recbole.utils import InputType

from .etrc_modules import EvidentialTransferHead, evidential_classification_loss
from .mfgslaetgf import MFGSLAETGF


class ETRCMFGSLAEPair(MFGSLAETGF):
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
        self.etrc_score_chunk_size = config["etrc_score_chunk_size"]

        self.transfer_head = EvidentialTransferHead(
            input_size=3,
            hidden_size=config["etrc_hidden_size"],
            dropout=config["etrc_dropout"],
            num_classes=self.etrc_num_classes,
        )
        self.transfer_head_device = torch.device("cpu")
        self.current_epoch = 0

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.transfer_head.to(self.transfer_head_device)
        return self

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

    def _gate_from_probs(self, probs, reliability):
        gate = (
            self.etrc_lambda_base
            + torch.sigmoid(reliability / max(self.etrc_gate_temperature, 1e-6))
            - self.etrc_harmful_penalty * probs[:, self.HARMFUL]
        )
        return gate.clamp(self.etrc_lambda_min, self.etrc_lambda_max)

    def transfer_gate(self, transfer_scores, target_only_scores):
        shape = transfer_scores.shape
        transfer_flat = transfer_scores.reshape(-1)
        target_only_flat = target_only_scores.reshape(-1)
        head_device = next(self.transfer_head.parameters()).device

        gate_chunks = []
        probs_chunks = []
        alpha_chunks = []

        for start in range(0, transfer_flat.numel(), self.etrc_score_chunk_size):
            end = start + self.etrc_score_chunk_size
            features = self.reliability_features(
                transfer_flat[start:end], target_only_flat[start:end]
            ).to(head_device)
            alpha, probs, uncertainty = self.transfer_head(features)
            reliability = (probs[:, self.HELPFUL] - probs[:, self.HARMFUL]) * (
                1.0 - uncertainty
            )
            gate_chunks.append(self._gate_from_probs(probs, reliability).to(transfer_scores.device))
            probs_chunks.append(probs.to(transfer_scores.device))
            alpha_chunks.append(alpha.to(transfer_scores.device))

        gate = torch.cat(gate_chunks, dim=0).view(shape)
        probs = torch.cat(probs_chunks, dim=0)
        alpha = torch.cat(alpha_chunks, dim=0)
        return gate, probs, alpha

    def _score_level_gate_only(self, transfer_scores, target_only_scores):
        shape = transfer_scores.shape
        transfer_flat = transfer_scores.reshape(-1)
        target_only_flat = target_only_scores.reshape(-1)
        head_device = next(self.transfer_head.parameters()).device

        gate_chunks = []
        for start in range(0, transfer_flat.numel(), self.etrc_score_chunk_size):
            end = start + self.etrc_score_chunk_size
            features = self.reliability_features(
                transfer_flat[start:end], target_only_flat[start:end]
            ).to(head_device)
            alpha, probs, uncertainty = self.transfer_head(features)
            reliability = (probs[:, self.HELPFUL] - probs[:, self.HARMFUL]) * (
                1.0 - uncertainty
            )
            gate_chunks.append(self._gate_from_probs(probs, reliability).to(transfer_scores.device))
        return torch.cat(gate_chunks, dim=0).view(shape)

    def _make_transfer_labels(self, delta):
        labels = torch.full_like(delta, self.NEUTRAL, dtype=torch.long)
        labels[delta > self.etrc_delta_tau] = self.HELPFUL
        labels[delta < -self.etrc_delta_tau] = self.HARMFUL
        return labels

    def _reconstruction_loss_each(self, logits, rating_matrix):
        return -(F.log_softmax(logits, dim=1) * rating_matrix).sum(dim=1)

    def _pairwise_loss_each(self, logits, rating_matrix):
        return F.binary_cross_entropy_with_logits(logits, rating_matrix, reduction="none")

    def _compactness_and_l1_losses(self):
        if self.factor > 1:
            compactness_loss = self.alpha * (
                self.compactness(self.h_factor) + self.compactness(self.h_gsl)
            )
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
        final_logits = (1.0 - gate) * target_only_logits + gate * transfer_logits
        return final_logits, transfer_logits, target_only_logits, gate, probs, alpha

    def _chunkwise_aux_losses(
        self,
        transfer_target,
        target_only_target,
        delta,
        final_loss_each=None,
        target_only_loss_each=None,
    ):
        total = delta.numel()
        delta_flat = delta.reshape(-1)
        transfer_flat = transfer_target.reshape(-1)
        target_only_flat = target_only_target.reshape(-1)
        final_loss_flat = None if final_loss_each is None else final_loss_each.reshape(-1)
        target_only_loss_flat = (
            None if target_only_loss_each is None else target_only_loss_each.reshape(-1)
        )
        head_device = next(self.transfer_head.parameters()).device

        edl_acc = torch.tensor(0.0, device=head_device)
        delta_reg_acc = torch.tensor(0.0, device=head_device)
        noharm_acc = torch.tensor(0.0, device=head_device)

        for start in range(0, total, self.etrc_score_chunk_size):
            end = start + self.etrc_score_chunk_size
            features = self.reliability_features(
                transfer_flat[start:end], target_only_flat[start:end]
            ).to(head_device)
            alpha, probs, uncertainty = self.transfer_head(features)
            alpha = alpha.float()
            probs = probs.float()
            uncertainty = uncertainty.float()
            chunk_size = end - start

            if self.etrc_edl_weight > 0:
                labels = self._make_transfer_labels(delta_flat[start:end].to(head_device))
                chunk_edl = evidential_classification_loss(
                    alpha,
                    labels,
                    epoch=self.current_epoch,
                    anneal_epochs=self.etrc_kl_anneal_epochs,
                )
                expected_delta = probs[:, self.HELPFUL] - probs[:, self.HARMFUL]
                chunk_delta_reg = F.smooth_l1_loss(
                    expected_delta,
                    delta_flat[start:end].detach().to(head_device),
                    reduction="mean",
                )
                edl_acc = edl_acc + chunk_edl * chunk_size
                delta_reg_acc = delta_reg_acc + chunk_delta_reg * chunk_size

            if self.etrc_noharm_weight > 0:
                risk_weight = (
                    probs[:, self.HARMFUL] + (1.0 - probs[:, self.HELPFUL])
                ).detach()
                chunk_noharm = (
                    risk_weight
                    * F.relu(
                        final_loss_flat[start:end].to(head_device)
                        - target_only_loss_flat[start:end].detach().to(head_device)
                        + self.etrc_noharm_margin
                    )
                ).mean()
                noharm_acc = noharm_acc + chunk_noharm * chunk_size

        outputs = {}
        if self.etrc_edl_weight > 0:
            outputs["edl_loss"] = (edl_acc / total).to(self.device)
            outputs["delta_reg_loss"] = (delta_reg_acc / total).to(self.device)
        if self.etrc_noharm_weight > 0:
            outputs["noharm_loss"] = (noharm_acc / total).to(self.device)
        return outputs

    def calculate_loss(self, interaction):
        source_user_id = interaction[self.SOURCE_USER_ID]
        target_user_id = interaction[self.TARGET_USER_ID]

        rating_matrix_source = self.rating_matrix_source[source_user_id.cpu()].to(self.device)
        rating_matrix_target = self.rating_matrix_target[target_user_id.cpu()].to(self.device)

        reconstructed_source, transfer_target = self.forward(source_user_id, target_user_id)
        compactness_loss, l1_loss = self._compactness_and_l1_losses()
        target_only_target = self.target_only_forward(target_user_id)
        gate = self._score_level_gate_only(transfer_target, target_only_target)
        final_target = (1.0 - gate) * target_only_target + gate * transfer_target

        ce_loss_source = self._reconstruction_loss_each(
            reconstructed_source, rating_matrix_source
        ).mean()
        ce_loss_target = self._reconstruction_loss_each(
            final_target, rating_matrix_target
        ).mean()

        losses = [
            0.1 * ce_loss_source,
            0.9 * ce_loss_target,
            0.5 * compactness_loss,
            0.2 * l1_loss,
        ]

        with torch.no_grad():
            target_only_loss_each = self._pairwise_loss_each(
                target_only_target, rating_matrix_target
            )
            transfer_loss_each = self._pairwise_loss_each(
                transfer_target, rating_matrix_target
            )
            delta = target_only_loss_each - transfer_loss_each

        if self.etrc_noharm_weight > 0 or self.etrc_tgf_weight > 0:
            final_loss_each = self._pairwise_loss_each(final_target, rating_matrix_target)

        del reconstructed_source
        del rating_matrix_source
        del transfer_loss_each
        del gate
        del final_target
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

        if self.etrc_edl_weight > 0 or self.etrc_noharm_weight > 0:
            aux_losses = self._chunkwise_aux_losses(
                transfer_target,
                target_only_target,
                delta,
                final_loss_each=final_loss_each if self.etrc_noharm_weight > 0 else None,
                target_only_loss_each=target_only_loss_each if self.etrc_noharm_weight > 0 else None,
            )
            if self.etrc_edl_weight > 0:
                losses.append(
                    self.etrc_edl_weight
                    * (
                        aux_losses["edl_loss"]
                        + self.etrc_delta_reg_weight * aux_losses["delta_reg_loss"]
                    )
                )
            if self.etrc_noharm_weight > 0:
                losses.append(self.etrc_noharm_weight * aux_losses["noharm_loss"])

        if self.etrc_tgf_weight > 0:
            overlap_mask = target_user_id < self.overlapped_num_users
            non_overlap_mask = ~overlap_mask
            if overlap_mask.any() and non_overlap_mask.any():
                gain = target_only_loss_each.detach() - final_loss_each
                overlap_gain = gain[overlap_mask].mean()
                non_overlap_gain = gain[non_overlap_mask].mean()
                tgf_loss = F.relu(
                    overlap_gain - non_overlap_gain - self.etrc_tgf_margin
                ).pow(2)
            else:
                tgf_loss = final_target.new_tensor(0.0)
            losses.append(self.etrc_tgf_weight * tgf_loss)

        return tuple(losses)

    def predict(self, interaction):
        transfer_score = super().predict(interaction)
        target_only_score = self.predict_target_only(interaction)
        gate, _, _ = self.transfer_gate(transfer_score, target_only_score)
        return (1.0 - gate) * target_only_score + gate * transfer_score

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        _, transfer_logits = self.forward(target_user=user)
        target_only_logits = self.target_only_forward(target_user=user)
        gate = self._score_level_gate_only(transfer_logits, target_only_logits)
        final_logits = (1.0 - gate) * target_only_logits + gate * transfer_logits
        return final_logits.reshape(-1)
