import torch

from .dtcdr import DTCDR


class DTCDRTGF(DTCDR):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if self.base_model != 'NeuMF':
            raise NotImplementedError('DTCDRTGF currently supports the default NeuMF backbone.')

    def target_only_neumf_forward(self, user, item):
        user_e = self.target_user_embedding(user)
        item_e = self.target_item_embedding(item)
        output = self.target_sigmoid(
            self.target_predict_layer(self.target_mlp_layers(torch.cat((user_e, item_e), -1)))
        )
        return output.squeeze(-1)

    def target_only_score(self, user, item):
        return self.target_only_neumf_forward(user, item)

    def predict_target_only(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        return self.target_only_score(user, item)

    def full_sort_predict_target_only(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        return self._full_sort_with_score_fn(user, self.target_only_score).view(-1)
