import torch

from recbole.utils import InputType

from .natr import NATR


class NATRTGF(NATR):
    input_type = InputType.POINTWISE

    def target_transfer_score(self, user, item):
        return self.phase2_forward(user, item)

    def target_only_score(self, user, item):
        user_e = self.target_user_embedding(user)
        item_e = self.target_item_embedding(item)
        return self.sigmoid(torch.mul(user_e, item_e).sum(dim=1))

    def predict(self, interaction):
        if self.phase == 'SOURCE':
            return super().predict(interaction)
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        return self.target_transfer_score(user, item)

    def predict_target_only(self, interaction):
        if self.phase == 'SOURCE':
            return super().predict(interaction)
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        return self.target_only_score(user, item)

    def _full_sort_with_score_fn(self, user, score_fn):
        all_item = torch.arange(self.target_num_items, device=user.device)
        score_list = []
        for uid in user:
            batch_user = uid.repeat(self.target_num_items)
            score_list.append(score_fn(batch_user, all_item).view(1, -1))
        return torch.cat(score_list, dim=0)

    def _source_full_sort_predict(self, interaction):
        user = interaction[self.SOURCE_USER_ID]
        user_e = self.source_user_embedding(user)
        overlap_item_e = self.source_item_embedding.weight[:self.overlapped_num_items]
        source_item_e = self.source_item_embedding.weight[self.target_num_items:]
        all_item_e = torch.cat([overlap_item_e, source_item_e], dim=0)
        return torch.matmul(user_e, all_item_e.transpose(0, 1)).view(-1)

    def full_sort_predict(self, interaction):
        if self.phase == 'SOURCE':
            return self._source_full_sort_predict(interaction)
        user = interaction[self.TARGET_USER_ID]
        return self._full_sort_with_score_fn(user, self.target_transfer_score).view(-1)

    def full_sort_predict_target_only(self, interaction):
        if self.phase == 'SOURCE':
            return self._source_full_sort_predict(interaction)
        user = interaction[self.TARGET_USER_ID]
        return self._full_sort_with_score_fn(user, self.target_only_score).view(-1)
