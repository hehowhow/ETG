import torch
import torch.nn.functional as F

from recbole.utils import InputType

from .dtcdrtgf import DTCDRTGF
from .etrc_modules import (
    EvidentialTransferHead,
    PlainTransferHead,
    evidential_classification_loss,
)


class ETRCDTCDRLogit(DTCDRTGF):
    input_type = InputType.POINTWISE

    HELPFUL = 0
    NEUTRAL = 1
    HARMFUL = 2

    @staticmethod
    def _cfg(config, key, default):
        if hasattr(config, "final_config_dict"):
            return config.final_config_dict.get(key, default)
        return config.get(key, default)

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if self.base_model != "NeuMF":
            raise NotImplementedError("ETRCDTCDRLogit currently supports the default NeuMF backbone.")

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
        self.etrc_plain_softmax_head = self._cfg(config, "etrc_plain_softmax_head", False)
        self.etrc_disable_uncertainty = self._cfg(config, "etrc_disable_uncertainty", False)
        self.etrc_use_target_only_feature = self._cfg(config, "etrc_use_target_only_feature", True)
        self.etrc_use_delta_feature = self._cfg(config, "etrc_use_delta_feature", True)

        head_cls = PlainTransferHead if self.etrc_plain_softmax_head else EvidentialTransferHead
        self.transfer_head = head_cls(
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
        target_feature = target_only_score if self.etrc_use_target_only_feature else torch.zeros_like(target_only_score)
        delta_feature = (
            transfer_score - target_only_score
            if self.etrc_use_delta_feature
            else torch.zeros_like(transfer_score)
        )
        return torch.stack(
            [
                target_feature,
                transfer_score,
                delta_feature,
            ],
            dim=-1,
        ).detach()

    def transfer_evidence(self, transfer_score, target_only_score):
        features = self.reliability_features(transfer_score, target_only_score)
        state_output, probs, uncertainty = self.transfer_head(features)
        reliability = probs[:, self.HELPFUL] - probs[:, self.HARMFUL]
        if not self.etrc_disable_uncertainty and not self.etrc_plain_softmax_head:
            reliability = reliability * (1.0 - uncertainty)
        return state_output, probs, uncertainty, reliability

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

    def _make_transfer_labels(self, delta):
        labels = torch.full_like(delta, self.NEUTRAL, dtype=torch.long)
        labels[delta > self.etrc_delta_tau] = self.HELPFUL
        labels[delta < -self.etrc_delta_tau] = self.HARMFUL
        return labels

    def _overlap_mask(self, user, item):
        if self.overlapped_num_users > 1:
            return user < self.overlapped_num_users
        if self.overlapped_num_items > 1:
            return item < self.overlapped_num_items
        return torch.zeros_like(user, dtype=torch.bool)

    def _shared_transfer_embeddings(self, user, item):
        user_source_e = self.source_user_embedding(user)
        user_target_e = self.target_user_embedding(user)
        user_e = torch.maximum(user_source_e, user_target_e)

        item_source_e = self.source_item_embedding(item)
        item_target_e = self.target_item_embedding(item)
        item_e = torch.maximum(item_source_e, item_target_e)
        return user_e, item_e

    def source_logits(self, user, item):
        user_e, item_e = self._shared_transfer_embeddings(user, item)
        hidden = self.source_mlp_layers(torch.cat((user_e, item_e), -1))
        return self.source_predict_layer(hidden).squeeze(-1)

    def target_transfer_logits(self, user, item):
        user_e, item_e = self._shared_transfer_embeddings(user, item)
        hidden = self.target_mlp_layers(torch.cat((user_e, item_e), -1))
        return self.target_predict_layer(hidden).squeeze(-1)

    def target_only_logits(self, user, item):
        user_e = self.target_user_embedding(user)
        item_e = self.target_item_embedding(item)
        hidden = self.target_mlp_layers(torch.cat((user_e, item_e), -1))
        return self.target_predict_layer(hidden).squeeze(-1)

    def target_transfer_score(self, user, item):
        return self.target_transfer_logits(user, item)

    def target_only_score(self, user, item):
        return self.target_only_logits(user, item)

    def calculate_loss(self, interaction):
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        source_logits = self.source_logits(source_user, source_item)
        transfer_logits = self.target_transfer_score(target_user, target_item)
        target_only_logits = self.target_only_score(target_user, target_item)
        gate, probs, state_output = self.transfer_gate(transfer_logits, target_only_logits)
        final_logits = (1.0 - gate) * target_only_logits + gate * transfer_logits

        loss_s = F.binary_cross_entropy_with_logits(source_logits, source_label)
        loss_t = F.binary_cross_entropy_with_logits(final_logits, target_label)
        losses = [loss_s * self.alpha + loss_t * (1 - self.alpha)]

        with torch.no_grad():
            target_only_loss_each = F.binary_cross_entropy_with_logits(
                target_only_logits, target_label, reduction="none"
            )
            transfer_loss_each = F.binary_cross_entropy_with_logits(
                transfer_logits, target_label, reduction="none"
            )
            delta = target_only_loss_each - transfer_loss_each
            transfer_labels = self._make_transfer_labels(delta)

        if self.etrc_plain_softmax_head:
            state_loss = F.nll_loss(torch.log(probs.clamp_min(1e-8)), transfer_labels, reduction="mean")
        else:
            state_loss = evidential_classification_loss(
                state_output,
                transfer_labels,
                epoch=self.current_epoch,
                anneal_epochs=self.etrc_kl_anneal_epochs,
            )
        expected_delta = probs[:, self.HELPFUL] - probs[:, self.HARMFUL]
        delta_reg_loss = F.smooth_l1_loss(expected_delta, delta.detach())
        if self.etrc_edl_weight > 0:
            losses.append(self.etrc_edl_weight * (state_loss + self.etrc_delta_reg_weight * delta_reg_loss))

        if self.etrc_noharm_weight > 0 or self.etrc_tgf_weight > 0:
            final_loss_each = F.binary_cross_entropy_with_logits(final_logits, target_label, reduction="none")

        if self.etrc_noharm_weight > 0:
            risk_weight = (probs[:, self.HARMFUL] + (1.0 - probs[:, self.HELPFUL])).detach()
            noharm_loss = (
                risk_weight
                * F.relu(final_loss_each - target_only_loss_each.detach() + self.etrc_noharm_margin)
            ).mean()
            losses.append(self.etrc_noharm_weight * noharm_loss)

        if self.etrc_tgf_weight > 0:
            overlap_mask = self._overlap_mask(target_user, target_item)
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

    def predict_target_only(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        return self.target_only_score(user, item)

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        return self._gated_pair_score(user, item)

    def _full_sort_with_score_fn(self, user, score_fn):
        all_item = torch.arange(self.target_num_items, device=user.device)
        score_list = []
        for uid in user:
            batch_user = uid.repeat(self.target_num_items)
            score_list.append(score_fn(batch_user, all_item).view(1, -1))
        return torch.cat(score_list, dim=0)

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        return self._full_sort_with_score_fn(user, self._gated_pair_score).view(-1)

    def full_sort_predict_target_only(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        return self._full_sort_with_score_fn(user, self.target_only_score).view(-1)

    def _gated_pair_score(self, user, item):
        transfer_score = self.target_transfer_score(user, item)
        target_only_score = self.target_only_score(user, item)
        gate, _, _ = self.transfer_gate(transfer_score, target_only_score)
        return (1.0 - gate) * target_only_score + gate * transfer_score
