import numpy as np
import scipy.sparse as sp

import torch

from recbole.utils import InputType

from .bitgcf import BiTGCF


class BiTGCFTGF(BiTGCF):
    input_type = InputType.POINTWISE

    def get_norm_adj_mat(self, interaction_matrix, n_users=None, n_items=None):
        if n_users is None or n_items is None:
            n_users, n_items = interaction_matrix.shape
        inter_m = interaction_matrix
        inter_m_t = interaction_matrix.transpose()
        rows = np.concatenate([inter_m.row, inter_m_t.row + n_users])
        cols = np.concatenate([inter_m.col + n_users, inter_m_t.col])
        data = np.ones(rows.shape[0], dtype=np.float32)
        adjacency = sp.coo_matrix((data, (rows, cols)), shape=(n_users + n_items, n_users + n_items), dtype=np.float32)
        degree = np.array((adjacency > 0).sum(axis=1)).flatten() + 1e-7
        degree = np.power(degree, -0.5)
        laplace = sp.diags(degree) * adjacency * sp.diags(degree)
        laplace = sp.coo_matrix(laplace)
        indices = torch.LongTensor([laplace.row, laplace.col])
        values = torch.FloatTensor(laplace.data)
        return torch.sparse.FloatTensor(indices, values, torch.Size(laplace.shape))

    def forward_with_transfer_settings(self, is_transfer=None):
        old_is_transfer = self.is_transfer
        old_restore_user_e = self.target_restore_user_e
        old_restore_item_e = self.target_restore_item_e

        try:
            if is_transfer is not None:
                self.is_transfer = is_transfer
            self.target_restore_user_e = None
            self.target_restore_item_e = None
            return self.forward()
        finally:
            self.is_transfer = old_is_transfer
            self.target_restore_user_e = old_restore_user_e
            self.target_restore_item_e = old_restore_item_e

    def target_embeddings_with_transfer_settings(self, is_transfer=None):
        _, _, target_user_all_embeddings, target_item_all_embeddings = self.forward_with_transfer_settings(
            is_transfer=is_transfer
        )
        return target_user_all_embeddings, target_item_all_embeddings

    def target_only_embeddings(self):
        return self.target_embeddings_with_transfer_settings(is_transfer=False)

    def predict_target_only(self, interaction):
        target_user_all_embeddings, target_item_all_embeddings = self.target_only_embeddings()
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]

        u_embeddings = target_user_all_embeddings[user]
        i_embeddings = target_item_all_embeddings[item]
        return torch.mul(u_embeddings, i_embeddings).sum(dim=1)

    def full_sort_predict_target_only(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        target_user_all_embeddings, target_item_all_embeddings = self.target_only_embeddings()
        u_embeddings = target_user_all_embeddings[user]
        i_embeddings = target_item_all_embeddings[:self.target_num_items]
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)
