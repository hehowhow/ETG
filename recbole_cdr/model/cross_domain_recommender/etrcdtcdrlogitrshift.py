import torch

from .etrcdtcdrlogit import ETRCDTCDRLogit


class ETRCDTCDRLogitRShift(ETRCDTCDRLogit):
    """Variant that folds the harmful penalty into reliability before sigmoid."""

    def transfer_evidence(self, transfer_score, target_only_score):
        features = self.reliability_features(transfer_score, target_only_score)
        state_output, probs, uncertainty = self.transfer_head(features)
        reliability = probs[:, self.HELPFUL] - (1.0 + self.etrc_harmful_penalty) * probs[:, self.HARMFUL]
        if not self.etrc_disable_uncertainty and not self.etrc_plain_softmax_head:
            reliability = reliability * (1.0 - uncertainty)
        return state_output, probs, uncertainty, reliability

    def _gate_from_probs(self, probs, reliability):
        return torch.sigmoid(reliability / max(self.etrc_gate_temperature, 1e-6))
