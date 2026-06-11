import torch
import torch.nn.functional as F

from recbole.utils import InputType

from .etrc_delta_label import ETRCDeltaLabelMixin
from .etrc_modules import evidential_classification_loss
from .etrcconet import ETRCCoNet


class ETRCCoNetDelta(ETRCDeltaLabelMixin, ETRCCoNet):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.init_delta_labeler(config, dataset)

    def calculate_loss(self, interaction):
        if not self.is_transfer:
            return super().calculate_loss(interaction)

        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        source_score = self.source_forward(source_user, source_item)
        transfer_score = self.target_transfer_score(target_user, target_item)
        target_only_score = self.target_only_score(target_user, target_item)
        alpha, probs, uncertainty, reliability = self.transfer_evidence(transfer_score, target_only_score)
        gate = self._gate_from_probs(probs, reliability)
        final_score = (1.0 - gate) * target_only_score + gate * transfer_score

        source_loss = self.loss(source_score, source_label)
        target_loss = self.loss(final_score, target_label)
        losses = [source_loss + target_loss + self._base_reg_loss()]

        with torch.no_grad():
            target_only_loss_each = self._bce_each(target_only_score, target_label)
            delta = self.compute_delta_signal(
                transfer_score,
                target_only_score,
                target_label,
                score_type='prob',
            )
            label_delta, transfer_labels = self.make_delta_labels(delta, target_user)

        edl_loss = evidential_classification_loss(
            alpha,
            transfer_labels,
            epoch=self.current_epoch,
            anneal_epochs=self.etrc_kl_anneal_epochs,
        )
        expected_delta = probs[:, self.HELPFUL] - probs[:, self.HARMFUL]
        delta_reg_loss = F.smooth_l1_loss(expected_delta, self.delta_regression_target(label_delta))
        if self.etrc_edl_weight > 0:
            losses.append(self.etrc_edl_weight * (edl_loss + self.etrc_delta_reg_weight * delta_reg_loss))

        if self.etrc_noharm_weight > 0 or self.etrc_tgf_weight > 0:
            final_loss_each = self._bce_each(final_score, target_label)

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
                tgf_loss = final_score.new_tensor(0.0)
            losses.append(self.etrc_tgf_weight * tgf_loss)

        return tuple(losses)
