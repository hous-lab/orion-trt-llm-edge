"""轻量 ego 轨迹预测头：从 Orion.pth 只提取 ego 相关权重（~1.6GB），
避免 test_trt.py 因加载完整 38GB Orion.pth 而 OOM。

与 orion.py 的 VAE 分支完全等价的子模块与 forward：
  - distribution_forward   (present_distribution + future_distribution)
  - future_states_predict  (predict_model, GRU + MLP)
  - ego_fut_decoder        (Linear-8192 -> ReLU -> Linear-8192 -> ReLU -> Linear-12)
"""
import os
import torch
import torch.nn as nn

from mmcv.models.utils.distributions import DistributionModule, PredictModel
from mmcv.models.utils import DistributionModule as _DU  # noqa
from mmcv.models.utils.transformer import inverse_sigmoid  # noqa


class LightweightEgoModule(nn.Module):
    """只含 ego 预测头的最小模型，key 前缀与 orion.py 子模块完全一致。"""

    def __init__(self):
        super().__init__()
        self.fut_ts = 6
        self.ego_fut_mode = 6
        self.layer_dim = 4
        self.latent_dim = 32
        self.with_cur = True
        self.MIN_LOG_SIGMA = -5.0
        self.MAX_LOG_SIGMA = 5.0
        self.present_distribution_in_channels = 4096
        self.future_distribution_in_channels = 4096 + 12

        self.present_distribution = DistributionModule(
            self.present_distribution_in_channels,
            self.latent_dim,
            min_log_sigma=self.MIN_LOG_SIGMA,
            max_log_sigma=self.MAX_LOG_SIGMA,
        )
        self.future_distribution = DistributionModule(
            self.future_distribution_in_channels,
            self.latent_dim,
            min_log_sigma=self.MIN_LOG_SIGMA,
            max_log_sigma=self.MAX_LOG_SIGMA,
        )
        self.predict_model = PredictModel(
            in_channels=self.latent_dim,
            out_channels=self.present_distribution_in_channels,
            hidden_channels=int(self.present_distribution_in_channels / self.layer_dim),
            num_layers=self.layer_dim,
        )
        decoder = []
        for _ in range(2):
            decoder.append(nn.Linear(8192, 8192))
            decoder.append(nn.ReLU())
        decoder.append(nn.Linear(8192, self.ego_fut_mode * 2))
        self.ego_fut_decoder = nn.Sequential(*decoder)

    def distribution_forward(self, present_features, future_distribution_inputs=None, noise=None):
        b = present_features.shape[0]
        c = present_features.shape[1]
        present_mu, present_log_sigma = self.present_distribution(present_features)

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            future_features = torch.cat([present_features, future_distribution_inputs], dim=2)
            future_mu, future_log_sigma = self.future_distribution(future_features)

        if noise is None:
            noise = torch.randn_like(present_mu)
        mu = present_mu
        sigma = torch.exp(present_log_sigma)
        sample = mu + sigma * noise
        sample = sample.permute(0, 2, 1).expand(b, self.latent_dim, c)

        output_distribution = {
            'present_mu': present_mu,
            'present_log_sigma': present_log_sigma,
            'future_mu': future_mu,
            'future_log_sigma': future_log_sigma,
        }
        return sample, output_distribution

    def future_states_predict(self, batch_size, sample, hidden_states, current_states):
        future_prediction_input = sample.unsqueeze(0).expand(self.fut_ts, -1, -1, -1)
        future_prediction_input = future_prediction_input.reshape(self.fut_ts, -1, self.latent_dim)

        hidden_states = hidden_states.permute(1, 0, 2)  # (4, 1, 4096) -> (1, 4, 4096)
        hidden_state = hidden_states.reshape(self.layer_dim, -1, int(4096 / 4))
        future_states = self.predict_model(future_prediction_input, hidden_state)

        current_states_hs = current_states.unsqueeze(0).repeat(6, 1, 1, 1)
        future_states_hs = future_states.reshape(self.fut_ts, batch_size, -1, future_states.shape[2])

        if self.with_cur:
            states_hs = torch.cat((current_states_hs, future_states_hs), dim=-1)
        else:
            states_hs = future_states_hs
        return states_hs, future_states_hs

    def predict_trajectory(self, ego_feature):
        """输入 LLM waypoint hidden_state [1, 4096]，输出 ego_fut_pred [1, 6, 6, 2]."""
        current_states = ego_feature.unsqueeze(1)  # [1, 1, 4096]
        sample, _ = self.distribution_forward(current_states)
        hidden_states = ego_feature.unsqueeze(1)
        states_hs, _ = self.future_states_predict(1, sample, hidden_states, current_states)
        ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
        trajs = []
        for i in range(self.fut_ts):
            out = self.ego_fut_decoder(ego_query_hs[i]).reshape(1, self.ego_fut_mode, 2)
            trajs.append(out)
        return torch.stack(trajs, dim=2)  # [1, 6, 6, 2]


def load_lightweight_ego(orion_pth, device='cpu'):
    """从 Orion.pth 提取 ego 权重，加载到轻量模型."""
    if isinstance(orion_pth, str):
        sd = torch.load(orion_pth, map_location='cpu', weights_only=False)
        if 'state_dict' in sd:
            sd = sd['state_dict']
    else:
        sd = orion_pth

    ego_prefix = ('present_distribution.', 'future_distribution.',
                  'predict_model.', 'ego_fut_decoder.')
    ego_sd = {k: v for k, v in sd.items() if k.startswith(ego_prefix)}

    model = LightweightEgoModule()
    model.eval()
    missing, unexpected = model.load_state_dict(ego_sd, strict=False)
    if missing:
        print(f"[WARN] missing ego keys: {missing}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected[:5]} ...")
    model.to(device)
    nbytes = sum(v.numel() * v.element_size() for v in model.state_dict().values())
    print(f"[ego] loaded {len(ego_sd)} keys, {nbytes / 2**20:.1f} MB -> {device}")
    return model


if __name__ == '__main__':
    import sys
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_ROOT)
    m = load_lightweight_ego(os.path.join(PROJECT_ROOT, 'ckpts/Orion.pth'), 'cpu')
    feat = torch.randn(1, 4096)
    with torch.no_grad():
        out = m.predict_trajectory(feat)
    print("traj:", tuple(out.shape), out.dtype)
    assert out.shape == (1, 6, 6, 2)
    print("OK: LightweightEgoModule works")
