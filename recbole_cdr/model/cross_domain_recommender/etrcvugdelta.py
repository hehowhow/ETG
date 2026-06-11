import torch
import torch.nn.functional as F

from recbole.utils import InputType

from .etrc_delta_label import ETRCDeltaLabelMixin
from .etrc_modules import evidential_classification_loss
from .etrcvug import ETRCVUG


class ETRCVUGDelta(ETRCDeltaLabelMixin, ETRCVUG):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.init_delta_labeler(config, dataset)

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
            target_only_loss_each = F.binary_cross_entropy_with_logits(target_only_logits, target_label, reduction='none')
            delta = self.compute_delta_signal(
                transfer_logits,
                target_only_logits,
                target_label,
                score_type='logit',
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
            final_loss_each = F.binary_cross_entropy_with_logits(final_logits, target_label, reduction='none')

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
