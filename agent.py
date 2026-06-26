"""
Networks for PPO + Random Network Distillation (RND).

Why RND and not a forward-model curiosity (ICM)?  ---------------------------
A forward-prediction curiosity reward is `||f(s,a) - s'||`. If part of the
observation is unpredictable *no matter how much you look at it* (a TV showing
static, a random animation, a stochastic transition), that error never goes to
zero, so the agent gets permanently "hooked" on the noise. That is the
noisy-TV problem.

RND sidesteps it. We keep a fixed, randomly initialized `target` network and
train a `predictor` to copy its output on observations we actually visit. The
target is *deterministic in the observation*, so there is no irreducible
stochastic component to chase: prediction error simply measures how *novel /
rarely-seen* an observation is, and it shrinks for anything visited enough.
Novelty, not predictability, drives exploration -> robust to noisy TVs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class RunningMeanStd:
    """Tracks running mean/variance (Welford / parallel update)."""
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var = M2 / tot
        self.count = tot


def _conv_stack(in_ch):
    # input 64x64 -> 7x7x64 (computed dynamically below, this is just the stack)
    return nn.Sequential(
        layer_init(nn.Conv2d(in_ch, 32, 8, stride=4)), nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
        nn.Flatten(),
    )


def _conv_out_dim(stack, in_ch, hw):
    with torch.no_grad():
        return stack(torch.zeros(1, in_ch, hw, hw)).shape[1]


class ActorCritic(nn.Module):
    """Shared CNN trunk -> policy logits + SEPARATE extrinsic & intrinsic values."""
    def __init__(self, in_ch=3, n_actions=5, hw=64):
        super().__init__()
        self.conv = _conv_stack(in_ch)
        flat = _conv_out_dim(self.conv, in_ch, hw)
        self.fc = nn.Sequential(layer_init(nn.Linear(flat, 256)), nn.ReLU())
        self.actor = layer_init(nn.Linear(256, n_actions), std=0.01)
        # extra hidden for the value heads tends to help (cleanrl-style)
        self.critic_ext = nn.Sequential(layer_init(nn.Linear(256, 256)), nn.ReLU(),
                                        layer_init(nn.Linear(256, 1), std=0.01))
        self.critic_int = nn.Sequential(layer_init(nn.Linear(256, 256)), nn.ReLU(),
                                        layer_init(nn.Linear(256, 1), std=0.01))

    def _features(self, x):
        # x: (B, C, H, W) float in [0,1]
        return self.fc(self.conv(x))

    def get_value(self, x):
        h = self._features(x)
        return self.critic_ext(h).squeeze(-1), self.critic_int(h).squeeze(-1)

    def get_action_and_value(self, x, action=None, mask=None):
        h = self._features(x)
        logits = self.actor(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e8)  # forbid invalid actions
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return (action,
                dist.log_prob(action),
                dist.entropy(),
                self.critic_ext(h).squeeze(-1),
                self.critic_int(h).squeeze(-1))


class RNDModel(nn.Module):
    """Fixed random `target` + trainable `predictor`. Intrinsic reward = MSE.

    With lpm=True this also builds the Learning Progress Monitoring error
    predictor (Hou et al. 2025): an MLP that predicts RND's own error from the
    (frozen target features + action). The LPM intrinsic reward subtracts the
    *predictable* part of the error:
        curiosity = rnd_error - alpha * predicted_error
    A noisy / already-mastered region has high-but-predictable error -> the
    predictor learns it -> it is subtracted away. Only error that is *higher
    than expected* (genuine learning progress) is rewarded. Unlike plain RND,
    this signal does not saturate to a flat constant on small state spaces.
    """
    def __init__(self, in_ch=3, out_dim=512, hw=64, lpm=False, n_actions=5):
        super().__init__()
        self.target = _conv_stack(in_ch)
        flat = _conv_out_dim(self.target, in_ch, hw)
        self.target = nn.Sequential(self.target, layer_init(nn.Linear(flat, out_dim)))

        pred_conv = _conv_stack(in_ch)
        self.predictor = nn.Sequential(
            pred_conv,
            layer_init(nn.Linear(flat, out_dim)), nn.ReLU(),
            layer_init(nn.Linear(out_dim, out_dim)), nn.ReLU(),
            layer_init(nn.Linear(out_dim, out_dim)),
        )
        for p in self.target.parameters():   # target is frozen forever
            p.requires_grad_(False)

        self.lpm = lpm
        self.n_actions = n_actions
        if lpm:
            self.error_predictor = nn.Sequential(
                layer_init(nn.Linear(out_dim + n_actions, 256)), nn.ReLU(),
                layer_init(nn.Linear(256, 256)), nn.ReLU(),
                layer_init(nn.Linear(256, 1)),
            )

    def forward(self, x):
        """x: normalized observation. Returns (predict_feat, target_feat)."""
        return self.predictor(x), self.target(x)

    def intrinsic_reward(self, x):
        with torch.no_grad():
            pred, tgt = self.forward(x)
            return ((pred - tgt) ** 2).mean(dim=1)  # per-sample novelty (plain RND)

    # ----- LPM additions -----
    def _err_and_target(self, x):
        pred, tgt = self.forward(x)
        err = ((pred - tgt) ** 2).mean(dim=1)
        return err, tgt

    def _predict_error(self, target_feat, act_onehot):
        inp = torch.cat([target_feat, act_onehot], dim=1)
        return self.error_predictor(inp).squeeze(-1)

    def lpm_curiosity(self, x, act_onehot, alpha):
        """Intrinsic reward = rnd_error - alpha * clamp(predicted_error, 0)."""
        with torch.no_grad():
            err, tgt = self._err_and_target(x)
            pe = torch.clamp(self._predict_error(tgt, act_onehot), min=0.0)
            return err - alpha * pe

    def error_predictor_loss(self, x, act_onehot):
        """Train the error predictor to regress RND's (detached) per-sample error."""
        with torch.no_grad():
            err, tgt = self._err_and_target(x)
        pe = self._predict_error(tgt, act_onehot)
        return F.mse_loss(pe, err)

