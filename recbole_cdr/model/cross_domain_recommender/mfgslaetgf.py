import torch

from recbole.utils import InputType

from .mfgslae import MFGSLAE


class MFGSLAETGF(MFGSLAE):
    input_type = InputType.PAIRWISE

    def _target_only_hidden(self):
        input_matrix = self.sprase_drop(self.sparse_norm_rating_target)
        hidden = torch.sparse.mm(input_matrix, self.target_encoder.weight.T) + self.target_encoder.bias
        h_target = self.act(hidden)

        had_h_factor = hasattr(self, "h_factor")
        had_h_gsl = hasattr(self, "h_gsl")
        old_h_factor = getattr(self, "h_factor", None)
        old_h_gsl = getattr(self, "h_gsl", None)

        try:
            # Reuse the original GSL augmentation stack, but feed it only target-domain interactions.
            gsl_input = self.sprase_drop(self.sparse_norm_rating_target)
            hidden = torch.sparse.mm(gsl_input, self.gsl_encoder.weight.T) + self.gsl_encoder.bias
            res = hidden[0:1]
            h_gsl = self.act(hidden[1:])
            h_gsl = self.factor_gsl(h_gsl)
            h_gsl = torch.cat([res, h_gsl])
            self.h_gsl = h_gsl.reshape(h_gsl.shape[0], self.factor, -1)
            if self.factor > 1:
                _, h_target_aug = self.factor_selection(h_gsl)
            else:
                h_target_aug = self.proj_target(h_gsl)

            h_target = self.ratio * h_target + (1 - self.ratio) * h_target_aug
        finally:
            if had_h_factor:
                self.h_factor = old_h_factor
            elif hasattr(self, "h_factor"):
                del self.h_factor
            if had_h_gsl:
                self.h_gsl = old_h_gsl
            elif hasattr(self, "h_gsl"):
                del self.h_gsl

        return h_target

    def target_only_forward(self, target_user=[]):
        hidden = self._target_only_hidden()
        reconstructed_target = hidden[target_user] @ self.decoder_target.weight.T + self.decoder_target.bias
        return reconstructed_target

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        _, reconstructed_target = self.forward(target_user=user)
        row_index = torch.arange(user.shape[0], device=user.device)
        return reconstructed_target[row_index, item]

    def predict_target_only(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        reconstructed_target = self.target_only_forward(target_user=user)
        row_index = torch.arange(user.shape[0], device=user.device)
        return reconstructed_target[row_index, item]

    def full_sort_predict_target_only(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        reconstructed_target = self.target_only_forward(target_user=user)
        return reconstructed_target.reshape(-1)
