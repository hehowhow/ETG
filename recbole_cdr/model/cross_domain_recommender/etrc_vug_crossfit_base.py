import csv
import os

import torch
import torch.nn.functional as F

from .etrc_modules import evidential_classification_loss
from .etrcvug import ETRCVUG


def build_pair_key(user_id, item_id, item_num):
    return int(user_id) * int(item_num) + int(item_id)


class ETRCVUGCrossFitBase(ETRCVUG):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.etrc_crossfit_delta_path = config['etrc_crossfit_delta_path']
        self.etrc_crossfit_missing_strategy = str(config['etrc_crossfit_missing_strategy']).lower()
        self.etrc_crossfit_supervise_positive_only = bool(config['etrc_crossfit_supervise_positive_only'])

        if not self.etrc_crossfit_delta_path:
            raise ValueError('`etrc_crossfit_delta_path` must be set for cross-fit ETRCVUG variants.')
        if not os.path.exists(self.etrc_crossfit_delta_path):
            raise FileNotFoundError(f'Cross-fit delta table not found: {self.etrc_crossfit_delta_path}')
        if self.etrc_crossfit_missing_strategy not in {'skip', 'neutral', 'online'}:
            raise ValueError(
                f'Unsupported etrc_crossfit_missing_strategy: {self.etrc_crossfit_missing_strategy}'
            )

        self._load_crossfit_table(self.etrc_crossfit_delta_path)

    def _load_crossfit_table(self, path):
        raise NotImplementedError

    def _lookup_crossfit_delta(self, target_user, target_item):
        raise NotImplementedError

    def _supervision_base_mask(self, target_label):
        if self.etrc_crossfit_supervise_positive_only:
            return target_label > 0.5
        return torch.ones_like(target_label, dtype=torch.bool)

    def _resolve_crossfit_targets(self, target_user, target_item, target_label, default_delta):
        base_mask = self._supervision_base_mask(target_label)
        label_delta, found_mask = self._lookup_crossfit_delta(target_user, target_item)

        if self.etrc_crossfit_missing_strategy == 'skip':
            supervision_mask = base_mask & found_mask
            resolved_delta = label_delta
        elif self.etrc_crossfit_missing_strategy == 'neutral':
            supervision_mask = base_mask
            resolved_delta = torch.where(found_mask, label_delta, torch.zeros_like(default_delta))
        else:
            supervision_mask = base_mask
            resolved_delta = torch.where(found_mask, label_delta, default_delta.detach())

        transfer_labels = self._make_transfer_labels(resolved_delta)
        return resolved_delta, transfer_labels, supervision_mask

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
            transfer_loss_each = F.binary_cross_entropy_with_logits(transfer_logits, target_label, reduction='none')
            default_delta = target_only_loss_each - transfer_loss_each
            label_delta, transfer_labels, supervision_mask = self._resolve_crossfit_targets(
                target_user,
                target_item,
                target_label,
                default_delta,
            )

        if self.etrc_edl_weight > 0 and supervision_mask.any():
            masked_alpha = alpha[supervision_mask]
            masked_labels = transfer_labels[supervision_mask]
            masked_probs = probs[supervision_mask]
            masked_delta = label_delta[supervision_mask]

            edl_loss = evidential_classification_loss(
                masked_alpha,
                masked_labels,
                epoch=self.current_epoch,
                anneal_epochs=self.etrc_kl_anneal_epochs,
            )
            expected_delta = masked_probs[:, self.HELPFUL] - masked_probs[:, self.HARMFUL]
            delta_reg_loss = F.smooth_l1_loss(expected_delta, masked_delta.detach())
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


class _PairCrossFitTableMixin:
    def _load_crossfit_table(self, path):
        self.crossfit_pair_delta_map = {}
        with open(path, 'r', encoding='utf-8') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                user_id = int(row['target_user_id'])
                item_id = int(row['target_item_id'])
                delta = float(row['pair_delta'])
                self.crossfit_pair_delta_map[build_pair_key(user_id, item_id, self.total_num_items)] = delta

    def _lookup_crossfit_delta(self, target_user, target_item):
        users = target_user.detach().cpu().tolist()
        items = target_item.detach().cpu().tolist()
        deltas = []
        found = []
        for user_id, item_id in zip(users, items):
            key = build_pair_key(user_id, item_id, self.total_num_items)
            if key in self.crossfit_pair_delta_map:
                deltas.append(self.crossfit_pair_delta_map[key])
                found.append(True)
            else:
                deltas.append(0.0)
                found.append(False)
        return (
            torch.tensor(deltas, dtype=torch.float32, device=target_user.device),
            torch.tensor(found, dtype=torch.bool, device=target_user.device),
        )


class _UserCrossFitTableMixin:
    def _load_crossfit_table(self, path):
        self.crossfit_user_delta_map = {}
        with open(path, 'r', encoding='utf-8') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                user_id = int(row['target_user_id'])
                delta = float(row['user_delta'])
                self.crossfit_user_delta_map[user_id] = delta

    def _lookup_crossfit_delta(self, target_user, target_item):
        del target_item
        users = target_user.detach().cpu().tolist()
        deltas = []
        found = []
        for user_id in users:
            if user_id in self.crossfit_user_delta_map:
                deltas.append(self.crossfit_user_delta_map[user_id])
                found.append(True)
            else:
                deltas.append(0.0)
                found.append(False)
        return (
            torch.tensor(deltas, dtype=torch.float32, device=target_user.device),
            torch.tensor(found, dtype=torch.bool, device=target_user.device),
        )
