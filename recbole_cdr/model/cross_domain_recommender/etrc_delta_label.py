import torch
import torch.nn.functional as F


class ETRCDeltaLabelMixin:
    HELPFUL = 0
    NEUTRAL = 1
    HARMFUL = 2

    _SIGNALS = {'loss', 'prob', 'margin', 'hybrid'}
    _STRATEGIES = {
        'instant',
        'confidence',
        'ema',
        'ema_confidence',
        'shrink',
        'shrink_confidence',
        'shrink_ema',
        'shrink_ema_confidence',
    }

    def init_delta_labeler(self, config, dataset):
        self.etrc_delta_mode = str(self._cfg(config, 'etrc_delta_mode', 'loss')).lower()
        if self.etrc_delta_mode in {'custom', 'separate'}:
            self.etrc_delta_signal = self._cfg(config, 'etrc_delta_signal', 'loss')
            self.etrc_delta_label_strategy = self._cfg(config, 'etrc_delta_label_strategy', 'instant')
        else:
            self.etrc_delta_signal, self.etrc_delta_label_strategy = self._parse_delta_mode(self.etrc_delta_mode)

        if self.etrc_delta_signal not in self._SIGNALS:
            raise ValueError(f'Unsupported etrc_delta_signal: {self.etrc_delta_signal}')
        if self.etrc_delta_label_strategy not in self._STRATEGIES:
            raise ValueError(f'Unsupported etrc_delta_label_strategy: {self.etrc_delta_label_strategy}')

        self.etrc_delta_conf_c = float(self._cfg(config, 'etrc_delta_conf_c', 0.0))
        self.etrc_delta_shrinkage_kappa = float(self._cfg(config, 'etrc_delta_shrinkage_kappa', 5.0))
        self.etrc_delta_ema_momentum = float(self._cfg(config, 'etrc_delta_ema_momentum', 0.9))
        self.etrc_delta_prior_momentum = float(self._cfg(config, 'etrc_delta_prior_momentum', 0.9))
        self.etrc_delta_reg_clip = float(self._cfg(config, 'etrc_delta_reg_clip', 1.0))

        self.etrc_delta_hybrid_loss_weight = float(self._cfg(config, 'etrc_delta_hybrid_loss_weight', 1.0))
        self.etrc_delta_hybrid_prob_weight = float(self._cfg(config, 'etrc_delta_hybrid_prob_weight', 1.0))
        self.etrc_delta_hybrid_margin_weight = float(self._cfg(config, 'etrc_delta_hybrid_margin_weight', 0.0))
        self.etrc_delta_loss_scale = float(self._cfg(config, 'etrc_delta_loss_scale', 1.0))
        self.etrc_delta_margin_scale = float(self._cfg(config, 'etrc_delta_margin_scale', 1.0))

        target_counts = self._build_target_user_counts(dataset)
        self.register_buffer('etrc_target_user_count', target_counts)
        self.register_buffer('etrc_user_delta_ema', torch.zeros(self.total_num_users))
        self.register_buffer('etrc_user_delta_seen', torch.zeros(self.total_num_users))
        self.register_buffer('etrc_delta_prior', torch.zeros(3))
        self.register_buffer('etrc_delta_prior_seen', torch.zeros(3))

    @staticmethod
    def _cfg(config, key, default):
        if hasattr(config, 'final_config_dict'):
            return config.final_config_dict.get(key, default)
        return config.get(key, default)

    def _parse_delta_mode(self, mode):
        mode = str(mode).lower()
        if mode in self._SIGNALS:
            return mode, 'instant'

        parts = mode.split('_')
        if not parts or parts[-1] not in self._SIGNALS:
            return mode, 'instant'

        signal = parts[-1]
        strategy = '_'.join(parts[:-1]) or 'instant'
        if strategy == 'conf':
            strategy = 'confidence'
        return signal, strategy

    def _build_target_user_counts(self, dataset):
        counts = torch.zeros(self.total_num_users, dtype=torch.float)
        target_dataset = getattr(dataset, 'target_domain_dataset', None)
        if target_dataset is None or target_dataset.inter_feat is None:
            return counts

        user_field = target_dataset.uid_field
        if user_field not in target_dataset.inter_feat:
            return counts

        user_ids = target_dataset.inter_feat[user_field]
        if not torch.is_tensor(user_ids):
            user_ids = torch.as_tensor(user_ids)
        user_ids = user_ids.long()
        valid_mask = (user_ids >= 0) & (user_ids < self.total_num_users)
        user_ids = user_ids[valid_mask]
        if user_ids.numel() > 0:
            counts.scatter_add_(0, user_ids, torch.ones_like(user_ids, dtype=torch.float))
        return counts

    def _prob_from_score(self, score, score_type):
        if score_type == 'logit':
            return torch.sigmoid(score)
        if score_type == 'prob':
            return score.clamp(1e-7, 1.0 - 1e-7)
        raise ValueError(f'Unsupported score_type: {score_type}')

    def _bce_each_for_delta(self, score, label, score_type):
        if score_type == 'logit':
            return F.binary_cross_entropy_with_logits(score, label, reduction='none')
        if score_type == 'prob':
            return F.binary_cross_entropy(score.clamp(1e-7, 1.0 - 1e-7), label, reduction='none')
        raise ValueError(f'Unsupported score_type: {score_type}')

    def compute_delta_signal(self, transfer_score, target_only_score, label, score_type):
        transfer_score = transfer_score.reshape(-1)
        target_only_score = target_only_score.reshape(-1)
        label = label.reshape(-1)

        loss_delta = (
            self._bce_each_for_delta(target_only_score, label, score_type)
            - self._bce_each_for_delta(transfer_score, label, score_type)
        )
        transfer_prob = self._prob_from_score(transfer_score, score_type)
        target_only_prob = self._prob_from_score(target_only_score, score_type)
        signed_label = label.mul(2.0).sub(1.0)
        prob_delta = signed_label * (transfer_prob - target_only_prob)

        if score_type == 'logit':
            margin_delta = signed_label * (transfer_score - target_only_score)
        else:
            margin_delta = prob_delta

        if self.etrc_delta_signal == 'loss':
            return loss_delta
        if self.etrc_delta_signal == 'prob':
            return prob_delta
        if self.etrc_delta_signal == 'margin':
            return margin_delta

        loss_component = torch.tanh(loss_delta / max(self.etrc_delta_loss_scale, 1e-6))
        margin_component = torch.tanh(margin_delta / max(self.etrc_delta_margin_scale, 1e-6))
        return (
            self.etrc_delta_hybrid_loss_weight * loss_component
            + self.etrc_delta_hybrid_prob_weight * prob_delta
            + self.etrc_delta_hybrid_margin_weight * margin_component
        )

    def make_delta_labels(self, delta, user):
        delta = delta.detach().reshape(-1)
        user = user.reshape(-1).long()

        strategy = self.etrc_delta_label_strategy
        use_ema = strategy in {'ema', 'ema_confidence', 'shrink_ema', 'shrink_ema_confidence'}
        use_shrink = strategy in {'shrink', 'shrink_confidence', 'shrink_ema', 'shrink_ema_confidence'}
        use_confidence = strategy in {'confidence', 'ema_confidence', 'shrink_confidence', 'shrink_ema_confidence'}

        self._update_delta_priors(delta, user)

        if use_ema:
            self._update_user_delta_ema(delta, user)
            label_delta = self.etrc_user_delta_ema[user]
        else:
            label_delta = delta

        if use_shrink:
            label_delta = self._shrink_delta_to_group_prior(label_delta, user)

        threshold = self._delta_threshold(user, use_confidence)
        labels = torch.full_like(user, self.NEUTRAL, dtype=torch.long)
        labels[label_delta > threshold] = self.HELPFUL
        labels[label_delta < -threshold] = self.HARMFUL
        return label_delta, labels

    def delta_regression_target(self, delta):
        if self.etrc_delta_reg_clip <= 0:
            return delta.detach()
        return delta.detach().clamp(-self.etrc_delta_reg_clip, self.etrc_delta_reg_clip)

    def _delta_threshold(self, user, use_confidence):
        threshold = torch.full_like(user, float(self.etrc_delta_tau), dtype=torch.float)
        if use_confidence and self.etrc_delta_conf_c > 0:
            counts = self.etrc_target_user_count[user].to(threshold.dtype)
            threshold = threshold + self.etrc_delta_conf_c / torch.sqrt(counts + 1.0)
        return threshold

    def _update_user_delta_ema(self, delta, user):
        unique_user, inverse = torch.unique(user, return_inverse=True)
        sums = torch.zeros(unique_user.size(0), device=delta.device, dtype=delta.dtype)
        nums = torch.zeros(unique_user.size(0), device=delta.device, dtype=delta.dtype)
        sums.scatter_add_(0, inverse, delta)
        nums.scatter_add_(0, inverse, torch.ones_like(delta))
        means = sums / nums.clamp_min(1.0)

        old = self.etrc_user_delta_ema[unique_user]
        seen = self.etrc_user_delta_seen[unique_user] > 0
        momentum = self.etrc_delta_ema_momentum
        updated = torch.where(seen, momentum * old + (1.0 - momentum) * means, means)
        self.etrc_user_delta_ema[unique_user] = updated
        self.etrc_user_delta_seen[unique_user] += nums

    def _update_delta_priors(self, delta, user):
        self._update_prior_slot(0, delta.mean())

        overlap_mask = user < self.overlapped_num_users
        if overlap_mask.any():
            self._update_prior_slot(1, delta[overlap_mask].mean())
        if (~overlap_mask).any():
            self._update_prior_slot(2, delta[~overlap_mask].mean())

    def _update_prior_slot(self, index, value):
        if self.etrc_delta_prior_seen[index] <= 0:
            self.etrc_delta_prior[index] = value
        else:
            momentum = self.etrc_delta_prior_momentum
            self.etrc_delta_prior[index] = momentum * self.etrc_delta_prior[index] + (1.0 - momentum) * value
        self.etrc_delta_prior_seen[index] += 1.0

    def _shrink_delta_to_group_prior(self, delta, user):
        counts = self.etrc_target_user_count[user].to(delta.dtype)
        kappa = max(self.etrc_delta_shrinkage_kappa, 0.0)
        if kappa <= 0:
            return delta

        global_prior = self.etrc_delta_prior[0]
        overlap_prior = torch.where(self.etrc_delta_prior_seen[1] > 0, self.etrc_delta_prior[1], global_prior)
        nonoverlap_prior = torch.where(self.etrc_delta_prior_seen[2] > 0, self.etrc_delta_prior[2], global_prior)
        prior = torch.where(user < self.overlapped_num_users, overlap_prior, nonoverlap_prior)
        weight = counts / (counts + kappa)
        return weight * delta + (1.0 - weight) * prior
