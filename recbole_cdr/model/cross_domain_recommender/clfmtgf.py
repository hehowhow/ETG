import torch

from recbole.utils import InputType

from .clfm import CLFM


class CLFMTGF(CLFM):
    input_type = InputType.POINTWISE

    def target_transfer_score(self, user, item):
        return self.target_forward(user, item)

    def target_only_forward(self, user, item):
        user_embedding = self.target_user_embedding(user)
        item_embedding = self.target_item_embedding(item)
        factors = []
        if self.target_item_embedding_size - self.share_embedding_size > 0:
            factors.append(self.target_only_linear(user_embedding))
        if not factors:
            raise NotImplementedError('CLFMTGF requires target-only dimensions beyond the shared subspace.')
        factors = torch.cat(factors, dim=1)
        item_embedding = item_embedding[:, self.share_embedding_size:]
        return self.sigmoid(torch.mul(factors, item_embedding).sum(dim=1))

    def target_only_score(self, user, item):
        return self.target_only_forward(user, item)

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
