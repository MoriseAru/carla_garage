"""BridgeDrive planning head for TransFuser++ (carla_garage): anchor-guided diffusion-bridge (DDBM) trajectory prediction.

Port of the released LEAD adaptation (github.com/shuliu-ethz/BridgeDrive, BridgeDrive_adaptation_LEAD/lead/tfv6/
diffusion_modules/model_diffusion_head_ddbm.py + blocks.py + multimodal_loss_ddbm.py) onto this fork, 2026-10-15.
What the head replaces: the GRU checkpoint decoder of TF++ (`checkpoint_decoder`). The 10 checkpoint queries of the
TF++ transformer decoder (`join`) become the ego queries of the head; the raw fused BEV map of the backbone
(bs, 1512, 8, 8) is grid-sampled at the trajectory points; the target-speed head of TF++ is untouched
(`diffusion_speed=false` in the released recipe: speed comes from the base head).
Kept verbatim from the reference: the VP bridge schedule (beta_d 2, beta_min 0.1, T 1, t in 1..1000), the two
2-layer decoders (regression + classification), 60 anchors, 20 ODE steps, the focal anchor-classification loss and the
L1 on the best mode's regression, the dropout 0.1, zero-initialised BEV attention weights.
Differences, all data-driven: the per-point mean / std of the trajectory and the k-means anchors are computed from OUR
expert labels (tools/bridgedrive_anchors.py) instead of the LEAD-data constants hard-coded in the reference.
Reference quirk reproduced on purpose (flag `zero_last_y`): at inference the reference sets the y of the LAST pose to 0
at every ODE step even when no speed holder is appended (the line is unconditional in forward_test); the released
checkpoint drove with it. Default keeps it; set config.bridgedrive_zero_last_y = 0 to drop it.
"""
import copy
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_BETA_D = 2.0
DEFAULT_BETA_MIN = 0.1
DEFAULT_T = 1.0
T_NORMALIZE = 1000.0
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_TIME_EMBED_DIMS = 256


# ----------------------------------------------------------------------------------------------------- building blocks
def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
  if input_dims is None:
    input_dims = embed_dims
  layers = []
  for _ in range(out_loops):
    for _ in range(in_loops):
      layers.append(nn.Linear(input_dims, embed_dims))
      layers.append(nn.ReLU(inplace=True))
      input_dims = embed_dims
    layers.append(nn.LayerNorm(embed_dims))
  return layers


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
  """DAB-DETR sine embedding of (..., 2) positions -> (..., hidden_dim); y half first, then x half."""
  half_hidden_dim = hidden_dim // 2
  scale = 2 * math.pi
  dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
  dim_t = 10000**(2 * (dim_t // 2) / half_hidden_dim)
  x_embed = pos_tensor[..., 0] * scale
  y_embed = pos_tensor[..., 1] * scale
  pos_x = x_embed[..., None] / dim_t
  pos_y = y_embed[..., None] / dim_t
  pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
  pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
  return torch.cat((pos_y, pos_x), dim=-1)


def bias_init_with_prob(prior_prob):
  return float(-np.log((1 - prior_prob) / prior_prob))


class SinusoidalPosEmb(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, x):
    half_dim = self.dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
    emb = x[:, None] * emb[None, :]
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


class GridSampleCrossBEVAttention(nn.Module):
  """Each mode query attends to the BEV map at its own trajectory points (bilinear grid_sample), with learned
  per-point weights (zero-initialised -> uniform at start). `max_x` / `max_y` (m) normalise the points to [-1, 1]."""

  def __init__(self, embed_dims, num_heads, max_x, max_y, in_bev_dims=64, num_points=8, swap_xy=False):
    super().__init__()
    self.embed_dims = embed_dims
    self.num_heads = num_heads
    self.num_points = num_points
    self.max_x, self.max_y = float(max_x), float(max_y)
    self.swap_xy = bool(swap_xy)
    self.attention_weights = nn.Linear(embed_dims, num_points)
    self.output_proj = nn.Linear(embed_dims, embed_dims)
    self.dropout = nn.Dropout(0.1)
    self.value_proj = nn.Sequential(nn.Conv2d(in_bev_dims, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
                                    nn.ReLU(inplace=True))
    nn.init.constant_(self.attention_weights.weight, 0)
    nn.init.constant_(self.attention_weights.bias, 0)
    nn.init.xavier_uniform_(self.output_proj.weight)
    nn.init.constant_(self.output_proj.bias, 0)

  def forward(self, queries, traj_points, bev_feature):
    """queries (bs, M, D); traj_points (bs, M, P, 2) in metres (x forward, y right); bev_feature (bs, C, H, W).
    grid_sample takes (col, row) in [-1, 1]. swap_xy=True is the reference layout (col = y, row = x; LEAD's BEV).
    In THIS fork the lidar histogram and therefore the backbone's fused map have col = x (forward), row = y (right)
    (tools/bridgedrive_smoke.py check 1 measures it on the input BEV), so the fork uses swap_xy=False."""
    bs, num_queries, num_points, _ = traj_points.shape
    normalized_trajectory = traj_points.clone()
    normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.max_x
    normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.max_y
    if self.swap_xy:
      normalized_trajectory = normalized_trajectory[..., [1, 0]]
    attention_weights = self.attention_weights(queries).view(bs, num_queries, num_points).softmax(-1)
    value = self.value_proj(bev_feature)
    sampled = F.grid_sample(value, normalized_trajectory.view(bs, num_queries, num_points, 2).to(value.dtype),
                            mode='bilinear', padding_mode='zeros', align_corners=False)   # bs, C, M, P
    out = (attention_weights.unsqueeze(1) * sampled).sum(dim=-1).permute(0, 2, 1).contiguous()
    out = self.output_proj(out)
    return self.dropout(out) + queries


# ---------------------------------------------------------------------------------------------------------- scheduler
def append_dims(x, target_dims):
  for _ in range(target_dims - x.ndim):
    x = x.unsqueeze(-1)
  return x


class DDBMScheduler:
  """Denoising diffusion bridge (VP schedule) between the clean trajectory x0 and the anchor xT."""

  def __init__(self, beta_d=DEFAULT_BETA_D, beta_min=DEFAULT_BETA_MIN, T=DEFAULT_T):
    self.beta_d, self.beta_min, self.T = beta_d, beta_min, T

  def vp_logs(self, t):
    t = torch.as_tensor(t)
    return -0.25 * t**2 * self.beta_d - 0.5 * t * self.beta_min

  def vp_logsnr(self, t):
    t = torch.as_tensor(t)
    return -torch.log((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1)

  def get_abc(self, t):
    logsnr_t, logsnr_T = self.vp_logsnr(t), self.vp_logsnr(self.T)
    logs_t, logs_T = self.vp_logs(t), self.vp_logs(self.T)
    a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
    b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
    c_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t / 2).exp()
    return a_t, b_t, c_t

  def add_noise(self, t, x0, xT, noise, t_normalize=T_NORMALIZE):
    t = append_dims(t, x0.ndim) / t_normalize
    a_t, b_t, c_t = self.get_abc(t)
    return a_t * xT + b_t * x0 + c_t * noise

  def sample_step(self, t, t_prev, xt, x0, xT, t_normalize=T_NORMALIZE):
    is_T = ((t / t_normalize) == self.T).all()
    t = append_dims(t, x0.ndim) / t_normalize
    t_prev = append_dims(t_prev, x0.ndim) / t_normalize
    a_t, b_t, c_t = self.get_abc(t)
    a_t_prev, b_t_prev, c_t_prev = self.get_abc(t_prev)
    xt_prev = a_t_prev * xT + b_t_prev * x0
    if is_T:
      xt_prev = xt_prev + c_t_prev * torch.randn_like(xt_prev)
    else:
      xt_prev = xt_prev + (c_t_prev / c_t) * (xt - a_t * xT - b_t * x0)
    return xt_prev


# ------------------------------------------------------------------------------------------------------ decoder layers
class ModulationLayer(nn.Module):

  def __init__(self, embed_dims, condition_dims=DEFAULT_TIME_EMBED_DIMS):
    super().__init__()
    self.scale_shift_mlp = nn.Sequential(nn.Mish(), nn.Linear(condition_dims, embed_dims * 2))

  def forward(self, traj_feature, time_embed):
    scale, shift = self.scale_shift_mlp(time_embed).chunk(2, dim=-1)
    return traj_feature * (1 + scale) + shift


class RefinementModule(nn.Module):
  """Regression (per mode: P poses x 3, the third channel unused) + classification (per mode: 1 logit)."""

  def __init__(self, embed_dims, ego_fut_ts):
    super().__init__()
    self.ego_fut_ts = ego_fut_ts
    self.plan_cls_branch = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2), nn.Linear(embed_dims, 1))
    self.plan_reg_branch = nn.Sequential(nn.Linear(embed_dims, embed_dims), nn.ReLU(), nn.Linear(embed_dims, embed_dims), nn.ReLU(),
                                         nn.Linear(embed_dims, ego_fut_ts * 3))
    nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init_with_prob(0.01))

  def forward(self, traj_feature):
    bs, ego_fut_mode, _ = traj_feature.shape
    plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
    plan_reg = self.plan_reg_branch(traj_feature).reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)
    return plan_reg, plan_cls


class RefinementModuleCls(nn.Module):

  def __init__(self, embed_dims):
    super().__init__()
    self.plan_cls_branch = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2), nn.Linear(embed_dims, 1))
    nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init_with_prob(0.01))

  def forward(self, traj_feature):
    return self.plan_cls_branch(traj_feature).squeeze(-1)


class DiffDecoderLayer(nn.Module):
  """Two parallel branches (noisy trajectory / anchor), each: BEV grid attention -> ego cross-attention -> FFN; concat;
  time modulation; refinement. Returns the denoised trajectory (noisy points + regressed delta) and a logit per mode."""

  def __init__(self, num_poses, d_model, d_ffn, num_heads, dropout, in_bev_dims, max_x, max_y, swap_xy=False):
    super().__init__()
    self.dropout1 = nn.Dropout(DEFAULT_DROPOUT_RATE)
    self.dropout1_T = nn.Dropout(DEFAULT_DROPOUT_RATE)
    self.cross_bev_attention = GridSampleCrossBEVAttention(d_model, num_heads, max_x, max_y, in_bev_dims=in_bev_dims, num_points=num_poses, swap_xy=swap_xy)
    self.cross_bev_attention_T = GridSampleCrossBEVAttention(d_model, num_heads, max_x, max_y, in_bev_dims=in_bev_dims, num_points=num_poses, swap_xy=swap_xy)
    self.cross_ego_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
    self.cross_ego_attention_T = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
    self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model))
    self.ffn_T = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model))
    self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
    self.norm1_T, self.norm2_T, self.norm3_T = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
    self.time_modulation = ModulationLayer(d_model * 2, DEFAULT_TIME_EMBED_DIMS)
    self.task_decoder = RefinementModule(embed_dims=d_model * 2, ego_fut_ts=num_poses)

  def forward(self, traj_feature, noisy_traj_points, plan_anchor, bev_feature, ego_query, time_embed):
    traj_feature_T = self.cross_bev_attention_T(traj_feature, plan_anchor, bev_feature)
    traj_feature = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_feature)
    traj_feature = self.norm1(traj_feature)
    traj_feature_T = self.norm1_T(traj_feature_T)
    traj_feature = self.norm2(traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query, ego_query)[0]))
    traj_feature_T = self.norm2_T(traj_feature_T + self.dropout1_T(self.cross_ego_attention_T(traj_feature_T, ego_query, ego_query)[0]))
    traj_feature = self.norm3(self.ffn(traj_feature))
    traj_feature_T = self.norm3_T(self.ffn_T(traj_feature_T))
    traj_feature = self.time_modulation(torch.cat([traj_feature, traj_feature_T], dim=-1), time_embed)
    poses_reg, poses_cls = self.task_decoder(traj_feature)
    poses_reg = poses_reg[..., :2] + noisy_traj_points
    return poses_reg, poses_cls


class DiffDecoderLayerCls(nn.Module):
  """Anchor classification branch: BEV grid attention at the anchor points -> ego cross-attention -> FFN -> logit."""

  def __init__(self, num_poses, d_model, d_ffn, num_heads, dropout, in_bev_dims, max_x, max_y, swap_xy=False):
    super().__init__()
    self.dropout1 = nn.Dropout(DEFAULT_DROPOUT_RATE)
    self.cross_bev_attention = GridSampleCrossBEVAttention(d_model, num_heads, max_x, max_y, in_bev_dims=in_bev_dims, num_points=num_poses, swap_xy=swap_xy)
    self.cross_ego_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
    self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model))
    self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
    self.task_decoder_cls = RefinementModuleCls(embed_dims=d_model)

  def forward(self, traj_feature, anchor_points, bev_feature, ego_query):
    traj_feature = self.norm1(self.cross_bev_attention(traj_feature, anchor_points, bev_feature))
    traj_feature = self.norm2(traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query, ego_query)[0]))
    traj_feature = self.norm3(self.ffn(traj_feature))
    return self.task_decoder_cls(traj_feature)


# ------------------------------------------------------------------------------------------------------------ the head
class BridgeDriveHead(nn.Module):
  """forward(ego_query (bs, Q, D), bev_feature (bs, C, H, W), gt (bs, P, 2) or None).
  Training (gt given): dict(poses_cls_list=[(bs, K) x layers], best_reg=(bs, P, 2)) -- the regression of the mode
  the classifier picks, computed from ONE noisy sample at a random bridge time t ~ U{1..1000}.
  Inference: (bs, P, 2) -- 20-step bridge sampling from the anchors, mode picked by the classifier."""

  def __init__(self, num_poses, d_model, d_ffn, num_heads, dropout, in_bev_dims, max_x, max_y, anchor_path, stats_path,
               step_num=20, num_layers=2, zero_last_y=True, swap_xy=False, num_anchors=60):
    super().__init__()
    self._num_poses = num_poses
    self.step_num = int(step_num)
    self.zero_last_y = bool(zero_last_y)
    # Anchors and statistics live in the state_dict (parameter / buffers), so a trained checkpoint restores them on any
    # machine; the files are only needed to START training. Missing files -> placeholders of the configured shape
    # (num_anchors), to be overwritten by load_state_dict (the agent on the other machine loads strict).
    if anchor_path and os.path.exists(anchor_path):
      plan_anchor = np.load(anchor_path)                    # (K, P, 2) metres, ego frame
    else:
      print(f'BridgeDriveHead: anchor file {anchor_path!r} not found -> zero placeholder ({num_anchors}, {num_poses}, 2); must be loaded from a checkpoint', flush=True)
      plan_anchor = np.zeros((num_anchors, num_poses, 2), dtype=np.float32)
    assert plan_anchor.ndim == 3 and plan_anchor.shape[1] == num_poses and plan_anchor.shape[2] == 2, plan_anchor.shape
    self.plan_anchor = nn.Parameter(torch.tensor(plan_anchor, dtype=torch.float32), requires_grad=False)
    if stats_path and os.path.exists(stats_path):
      stats = np.load(stats_path)                            # (2, P, 2): [mean, std] per pose, per coordinate
    else:
      print(f'BridgeDriveHead: stats file {stats_path!r} not found -> placeholder (mean 0, std 1); must be loaded from a checkpoint', flush=True)
      stats = np.stack([np.zeros((num_poses, 2)), np.ones((num_poses, 2))]).astype(np.float32)
    assert stats.shape == (2, num_poses, 2), stats.shape
    self.register_buffer('mean', torch.tensor(stats[0], dtype=torch.float32)[None, None])   # (1, 1, P, 2)
    self.register_buffer('std', torch.tensor(stats[1], dtype=torch.float32)[None, None])
    self.diffusion_scheduler = DDBMScheduler()
    self.plan_anchor_encoder = nn.Sequential(*linear_relu_ln(d_model, 1, 1, 64 * num_poses), nn.Linear(d_model, d_model))
    self.time_mlp = nn.Sequential(SinusoidalPosEmb(d_model), nn.Linear(d_model, d_model * 4), nn.Mish(), nn.Linear(d_model * 4, d_model))
    layer = DiffDecoderLayer(num_poses, d_model, d_ffn, num_heads, dropout, in_bev_dims, max_x, max_y, swap_xy)
    layer_cls = DiffDecoderLayerCls(num_poses, d_model, d_ffn, num_heads, dropout, in_bev_dims, max_x, max_y, swap_xy)
    self.diff_decoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
    self.diff_decoder_cls = nn.ModuleList([copy.deepcopy(layer_cls) for _ in range(num_layers)])

  # normalisation of trajectories (per pose, per coordinate)
  def norm_odo(self, x):
    return (x - self.mean) / self.std

  def denorm_odo(self, x):
    return x * self.std + self.mean

  def encode_points(self, pts):
    """(bs, K, P, 2) metres -> (bs, K, d_model)."""
    bs, k = pts.shape[:2]
    emb = gen_sineembed_for_position(pts, hidden_dim=64).flatten(-2)
    return self.plan_anchor_encoder(emb).view(bs, k, -1)

  def run_decoder(self, traj_feature, traj_points, plan_anchor, bev_feature, ego_query, time_embed):
    poses_reg_list, poses_cls_list = [], []
    for mod in self.diff_decoder:
      poses_reg, poses_cls = mod(traj_feature, traj_points, plan_anchor, bev_feature, ego_query, time_embed)
      poses_reg_list.append(poses_reg)
      poses_cls_list.append(poses_cls)
      traj_points = poses_reg.clone().detach()
    return poses_reg_list, poses_cls_list

  def run_decoder_cls(self, ak_feature, plan_anchor, bev_feature, ego_query):
    return [mod(ak_feature, plan_anchor, bev_feature, ego_query) for mod in self.diff_decoder_cls]

  def forward(self, ego_query, bev_feature, gt_checkpoints=None):
    if self.training:
      assert gt_checkpoints is not None, 'BridgeDriveHead in train mode needs the checkpoint labels'
      return self.forward_train(ego_query, bev_feature, gt_checkpoints)
    return self.forward_test(ego_query, bev_feature)

  def forward_train(self, ego_query, bev_feature, gt_checkpoints):
    bs, device = ego_query.shape[0], ego_query.device
    plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)                       # (bs, K, P, 2)
    x_0 = gt_checkpoints[:, None, :self._num_poses, :2].repeat(1, plan_anchor.shape[1], 1, 1)
    timesteps = torch.randint(1, 1001, (bs,), device=device)
    noise = torch.randn(x_0.shape, device=device)
    noisy = self.diffusion_scheduler.add_noise(t=timesteps, x0=self.norm_odo(x_0), xT=self.norm_odo(plan_anchor), noise=noise).float()
    noisy = self.denorm_odo(noisy)
    traj_feature = self.encode_points(noisy)
    time_embed = self.time_mlp(timesteps).view(bs, 1, -1)
    ak_feature = self.encode_points(plan_anchor).to(dtype=torch.float32)
    poses_reg_list, _ = self.run_decoder(traj_feature, noisy, plan_anchor, bev_feature, ego_query, time_embed)
    poses_cls_list = self.run_decoder_cls(ak_feature, plan_anchor, bev_feature, ego_query)
    mode_idx = poses_cls_list[-1].argmax(dim=-1)[..., None, None, None].repeat(1, 1, self._num_poses, 2)
    best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
    return {'poses_cls_list': poses_cls_list, 'best_reg': best_reg}

  @torch.no_grad()
  def forward_test(self, ego_query, bev_feature):
    bs, device = ego_query.shape[0], ego_query.device
    ts = torch.arange(0, 1001, 1000 // self.step_num, device=device)
    plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
    ak_feature = self.encode_points(plan_anchor)
    poses_cls_list = self.run_decoder_cls(ak_feature, plan_anchor, bev_feature, ego_query)
    mode_idx = poses_cls_list[-1].argmax(dim=-1)[..., None, None, None].repeat(1, 1, self._num_poses, 2)
    xt_prev = plan_anchor
    for i in range(self.step_num, 0, -1):
      xt = xt_prev
      t = ts[i] * torch.ones([bs], device=device)
      t_prev = ts[i - 1] * torch.ones([bs], device=device)
      traj_feature = self.encode_points(xt)
      time_embed = self.time_mlp(t).view(bs, 1, -1)
      poses_reg_list, _ = self.run_decoder(traj_feature, xt, plan_anchor, bev_feature, ego_query, time_embed)
      x0_mean = poses_reg_list[-1]
      if self.zero_last_y:
        x0_mean[..., -1, -1] = 0
      xt_prev = self.denorm_odo(self.diffusion_scheduler.sample_step(t=t, t_prev=t_prev, xt=self.norm_odo(xt.clone()),
                                                                     x0=self.norm_odo(x0_mean), xT=self.norm_odo(plan_anchor.clone())))
      if self.zero_last_y:
        xt_prev[..., -1, -1] = 0
    return torch.gather(xt_prev, 1, mode_idx).squeeze(1)

  # ------------------------------------------------------------------------------------------------------------ loss
  def compute_loss(self, out, gt_checkpoints):
    """Reference LossComputer applied per classification layer (so the L1 is counted once per layer, as in the
    reference loop): focal loss (gamma 2, alpha 0.25, mean over bs x K) on the one-hot of the anchor nearest to the
    label (mean point distance), plus L1(best_reg, label)."""
    target = gt_checkpoints[:, :self._num_poses, :2].float()
    plan_anchor = self.plan_anchor[None].float()
    dist = torch.linalg.norm(target.unsqueeze(1) - plan_anchor, dim=-1).mean(dim=-1)    # (bs, K)
    cls_target = dist.argmin(dim=-1)
    loss_cls, loss_reg = 0.0, 0.0
    for poses_cls in out['poses_cls_list']:
      onehot = torch.zeros_like(poses_cls).scatter_(1, cls_target.unsqueeze(1), 1.0)
      loss_cls = loss_cls + sigmoid_focal_loss(poses_cls.float(), onehot, gamma=2.0, alpha=0.25)
      loss_reg = loss_reg + F.l1_loss(out['best_reg'].float(), target)
    return loss_cls, loss_reg


def sigmoid_focal_loss(pred, target, gamma=2.0, alpha=0.25):
  pred_sigmoid = pred.sigmoid()
  target = target.type_as(pred)
  pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
  focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
  return (F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight).mean()
