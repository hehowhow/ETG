import torch

from recbole.utils import InputType

from .conet import CoNet


class CoNetTGF(CoNet):
    input_type = InputType.POINTWISE

    def target_transfer_score(self, user, item):
        if self.is_transfer:
            return self.target_forward(user, item)
        return self.target_forward_without_cross(user, item)

    def target_only_score(self, user, item):
        return self.target_forward_without_cross(user, item)

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        return self.target_transfer_score(user, item)

    def predict_target_only(self, interaction):
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

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        return self._full_sort_with_score_fn(user, self.target_transfer_score).view(-1)

    def full_sort_predict_target_only(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        return self._full_sort_with_score_fn(user, self.target_only_score).view(-1)
