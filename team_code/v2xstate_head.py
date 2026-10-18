"""Original V2XState planner (VTC2026 paper: intent-aware spatial basis attention) as a TF++ checkpoint head.

Port of V2XState_Real/codriving/models/planning_v2xstate_attention_learnable_gate.py (WaypointPlannerV2XStateLearnableGate,
config end2end_v2xstate_attention_learnable_gate_4.yaml) onto the TF++ backbone, ego sensing only (user decision 2026-10-18,
LOCAL 18:57 handover section; basis_source='concat' by user decision). It replaces TF++'s GRU checkpoint decoder; TF++'s
target-speed head stays (longitudinal control), as for BridgeDrive.

What is the same as the original (structure, widths, losses, hyper-parameters of the _4 config):
  occupancy branch conv_pre_1/2 (6 -> 32), feature branch conv_pre_1/2_f (C -> 32), K = 4 basis maps from basis_extractor,
  sigmoid(basis / T) x softmax(state_gate(state) / T), attention = clamp(sum, 0, 1), feature x (1 + attention),
  concat with occupancy -> conv_pre_*_f2 -> three stride-2 stages -> global mean -> [feature 256, target 128, state 96]
  -> MLP(512, 512, dropout 0.1) -> 10 points; aux losses = KL of each gated basis map to its hand-written sector mask
  (near / left / forward / right, weighted by command and steer) and the command -> basis gate prior.
What is adapted (each forced by the TF++ inputs, none changes the mechanism):
  - BEV feature = TF++'s bev_feature_grid (64 ch, 64 x 64, 1 m / px, +-32 m around the ego) instead of the cooperative
    fused 128-ch map; layout of this fork: column = x forward, row = y right (measured, tools/bridgedrive_smoke.py check 1).
  - Occupancy channels from TF++'s own outputs, detached (they were frozen-perception inputs in the original):
    [0] objects = max over car / walker / emergency of the CenterNet centre heatmap (transposed: the detection labels use
    column = y, row = x, see tfpp_occupancy_inputs), [1] ego box (2.45 x 1.0 m half
    extents, as the original render), [2] target-point 1 x 1 m box (clipped into the grid), [3, 4] forward / lateral
    coordinate in metres, [5] drivable = P(road) + P(road line) + P(broken road line) of the BEV semantic head inside the
    camera's visible area (0 outside: TF++ has no semantics there).
  - Single frame (TF++ seq_len = 1): the two temporal Conv3D (kernel 3 over 5 frames) become 1 x 1 convolutions + BN +
    ReLU at the same places (a kernel-1 Conv3D over one frame).
  - State = [speed m/s, steer, 7-way command]: TF++'s 6-way one-hot is the original's commands 1..6 (void slot 0 empty).
    Steer is zero unless v2xstate_use_steer (TF++ deliberately does not feed back the previous control).
  - Output = TF++'s 10 route checkpoints (x forward, y right, metres), L1 as TF++'s loss_checkpoint.
  - Mask grids are the TF++ BEV (forward -32..32, lateral -32..32) instead of forward -12..36 / lateral -12..12.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

COMMAND_MAP = dict(void=0, left=1, right=2, straight=3, lanefollow=4, change_left=5, change_right=6)


class _MLP(nn.Module):
  """codriving.models.planning_v2xstate.MLP: Linear-ReLU stack (no dropout here)."""

  def __init__(self, in_feat, out_feat, hid_feat=(16, 64)):
    super().__init__()
    dims = (in_feat,) + tuple(hid_feat) + (out_feat,)
    self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

  def forward(self, x):
    for layer in self.layers[:-1]:
      x = F.relu(layer(x))
    return self.layers[-1](x)


class _TemporalK1(nn.Module):
  """The original Conv3D(kernel (3,1,1)) + BN3d + ReLU over 5 frames, at one frame: a kernel-1 convolution."""

  def __init__(self, c_in, c_out):
    super().__init__()
    self.conv = nn.Conv2d(c_in, c_out, kernel_size=1)
    self.bn = nn.BatchNorm2d(c_out)

  def forward(self, x):
    return F.relu(self.bn(self.conv(x)))


class V2XStateHead(nn.Module):
  """forward(bev_feature (B, C, H, W), occupancy (B, 6, H, W), target_point (B, 2), speed (B, 1), command6 (B, 6),
  steer (B, 1) or None) -> dict(checkpoints (B, P, 2), gate (B, K), attention (B, H, W), basis (B, K, H, W),
  loss_mask, loss_gate) -- losses un-weighted (weights live in config.detailed_loss_weights)."""

  def __init__(self, feature_dim=64, height=64, width=64, min_x=-32.0, max_x=32.0, min_y=-32.0, max_y=32.0,
               num_basis=4, basis_temperature=0.5, state_hidden_dim=96, num_commands=7, traj_points=10,
               mlp_hidden_dims=(512, 512), mlp_dropout=0.1, gate_prior_alpha=0.9, gate_prior_sigma=0.7,
               basis_source='concat', occupancy_channels=6):
    super().__init__()
    assert basis_source in ('state', 'feature', 'concat'), basis_source
    self.basis_source, self.num_basis, self.num_commands, self.traj_points = basis_source, num_basis, num_commands, traj_points
    self.basis_temperature, self.gate_prior_alpha, self.gate_prior_sigma = basis_temperature, gate_prior_alpha, max(gate_prior_sigma, 1e-3)
    self.height, self.width = int(height), int(width)
    self._mask_sharpness, self._near_radius, self._lane_change_offset = 4.0, 4.0, 3.5
    self.max_range = float(max(abs(min_x), abs(max_x), abs(min_y), abs(max_y)))
    # metric grids of the fork's BEV layout: column -> x forward, row -> y right (pixel centres)
    xs = min_x + (torch.arange(self.width, dtype=torch.float32) + 0.5) * (max_x - min_x) / self.width
    ys = min_y + (torch.arange(self.height, dtype=torch.float32) + 0.5) * (max_y - min_y) / self.height
    fwd, lat = xs[None, :].expand(self.height, -1), ys[:, None].expand(-1, self.width)
    self.register_buffer('grid_forward', fwd.contiguous(), persistent=False)
    self.register_buffer('grid_lateral', lat.contiguous(), persistent=False)
    self.cell_x, self.cell_y = (max_x - min_x) / self.width, (max_y - min_y) / self.height
    self.register_buffer('ego_map', self.box_map(torch.zeros(1, 2), 2.45, 1.0)[0], persistent=False)

    self.fusion_feat_dim = 32
    self.conv_pre_1 = nn.Conv2d(occupancy_channels, 32, kernel_size=3, stride=1, padding=1)
    self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    self.bn_pre_1, self.bn_pre_2 = nn.BatchNorm2d(32), nn.BatchNorm2d(32)
    self.conv_pre_1_f = nn.Conv2d(feature_dim, 32, kernel_size=3, stride=1, padding=1)
    self.conv_pre_2_f = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    self.bn_pre_1_f, self.bn_pre_2_f = nn.BatchNorm2d(32), nn.BatchNorm2d(32)
    self.conv_pre_1_f2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
    self.conv_pre_2_f2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    self.bn_pre_1_f2, self.bn_pre_2_f2 = nn.BatchNorm2d(32), nn.BatchNorm2d(32)
    self.temporal_1, self.temporal_2 = _TemporalK1(64, 64), _TemporalK1(128, 128)
    self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
    self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
    self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
    self.bn1_1, self.bn1_2 = nn.BatchNorm2d(64), nn.BatchNorm2d(64)
    self.bn2_1, self.bn2_2, self.bn3_1 = nn.BatchNorm2d(128), nn.BatchNorm2d(128), nn.BatchNorm2d(256)

    self.state_to_feature = nn.Sequential(nn.Linear(state_hidden_dim, state_hidden_dim), nn.ReLU(inplace=True),
                                          nn.Linear(state_hidden_dim, state_hidden_dim), nn.ReLU(inplace=True),
                                          nn.Linear(state_hidden_dim, self.height * self.width))
    basis_in = self.fusion_feat_dim * (2 if basis_source == 'concat' else 1)
    self.basis_extractor = nn.Sequential(nn.Conv2d(basis_in, self.fusion_feat_dim, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(self.fusion_feat_dim), nn.ReLU(inplace=True),
                                         nn.Conv2d(self.fusion_feat_dim, num_basis, kernel_size=1))
    state_in = 2 + num_commands
    self.state_gate = nn.Sequential(nn.Linear(state_in, state_hidden_dim), nn.ReLU(inplace=True), nn.Linear(state_hidden_dim, num_basis))
    self.state_context = nn.Sequential(nn.Linear(state_in, state_hidden_dim), nn.ReLU(inplace=True))
    self.target_encoder = _MLP(2, 128, hid_feat=(16, 64))
    layers, d = [], 256 + 128 + state_hidden_dim
    for h in mlp_hidden_dims:
      layers += [nn.Linear(d, h), nn.ReLU(inplace=True)] + ([nn.Dropout(mlp_dropout)] if mlp_dropout > 0 else [])
      d = h
    layers.append(nn.Linear(d, traj_points * 2))
    self.mlp_head = nn.Sequential(*layers)

  # ------------------------------------------------------------------------------------------------------------ inputs
  def box_map(self, centre, half_x, half_y):
    """(B, H, W) area fraction of each cell covered by an axis-aligned box (half extents in metres) at centre (B, 2)."""
    fx, fy = self.grid_forward[None], self.grid_lateral[None]
    cx, cy = centre[:, 0, None, None], centre[:, 1, None, None]
    ox = (torch.minimum(fx + self.cell_x / 2, cx + half_x) - torch.maximum(fx - self.cell_x / 2, cx - half_x)).clamp(min=0)
    oy = (torch.minimum(fy + self.cell_y / 2, cy + half_y) - torch.maximum(fy - self.cell_y / 2, cy - half_y)).clamp(min=0)
    return ox * oy / (self.cell_x * self.cell_y)

  def occupancy(self, objects, drivable, target_point):
    """objects / drivable (B, H, W) in [0, 1] -> (B, 6, H, W) in the original channel order."""
    bs = objects.shape[0]
    lim = self.max_range - 0.5
    tgt = self.box_map(target_point.float().clamp(-lim, lim), 0.5, 0.5)
    ego = self.ego_map[None].expand(bs, -1, -1)
    return torch.stack([objects, ego, tgt, self.grid_forward[None].expand(bs, -1, -1),
                        self.grid_lateral[None].expand(bs, -1, -1), drivable], dim=1)

  def state(self, speed, command6, steer=None):
    bs = speed.shape[0]
    steer = torch.zeros(bs, 1, device=speed.device, dtype=torch.float32) if steer is None else steer.reshape(bs, 1).float()
    cmd = torch.cat([torch.zeros(bs, 1, device=speed.device), command6.float()], dim=1)[:, :self.num_commands]
    return torch.cat([speed.reshape(bs, 1).float(), steer, cmd], dim=1)

  # ----------------------------------------------------------------------------------------------------------- forward
  def forward(self, bev_feature, occupancy, target_point, state):
    bs, _, h, w = bev_feature.shape
    assert (h, w) == (self.height, self.width), f'BEV {h}x{w} != head grid {self.height}x{self.width}'
    occ = F.relu(self.bn_pre_1(self.conv_pre_1(occupancy)))
    occ = F.relu(self.bn_pre_2(self.conv_pre_2(occ)))
    feature = F.relu(self.bn_pre_1_f(self.conv_pre_1_f(bev_feature)))
    feature = F.relu(self.bn_pre_2_f(self.conv_pre_2_f(feature)))
    state_context = self.state_context(state)
    state_map = self.state_to_feature(state_context).view(bs, 1, h, w).expand(-1, self.fusion_feat_dim, -1, -1)
    if self.basis_source == 'state':
      basis_in = state_map
    elif self.basis_source == 'feature':
      basis_in = feature
    else:
      basis_in = torch.cat([feature, state_map], dim=1)
    temp = max(self.basis_temperature, 1e-6)
    basis = torch.sigmoid(self.basis_extractor(basis_in) / temp)                 # (B, K, H, W)
    gate = F.softmax(self.state_gate(state) / temp, dim=1)                        # (B, K)
    attention = (basis * gate[:, :, None, None]).sum(1, keepdim=True).clamp(0.0, 1.0)
    x = torch.cat((occ, feature * (attention + 1.0)), dim=1)
    x = F.relu(self.bn_pre_1_f2(self.conv_pre_1_f2(x)))
    x = F.relu(self.bn_pre_2_f2(self.conv_pre_2_f2(x)))
    x = F.relu(self.bn1_1(self.conv1_1(x)))
    x = self.temporal_1(F.relu(self.bn1_2(self.conv1_2(x))))
    x = F.relu(self.bn2_1(self.conv2_1(x)))
    x = self.temporal_2(F.relu(self.bn2_2(self.conv2_2(x))))
    x = F.relu(self.bn3_1(self.conv3_1(x)))
    context = torch.cat((x.mean(dim=(2, 3)), self.target_encoder(target_point.float()), state_context), dim=1)
    checkpoints = self.mlp_head(context).view(bs, self.traj_points, 2)
    # occupancy / state are returned for logging only (closed-loop consistency dumps); nothing reads them back
    out = dict(checkpoints=checkpoints, gate=gate, attention=attention[:, 0], basis=basis, occupancy=occupancy, state=state)
    if self.training:
      with torch.autocast(device_type=bev_feature.device.type, enabled=False):   # logs / KL in fp32
        g = gate.detach().float()   # as the original: the mask loss shapes the basis maps only
        out['loss_mask'] = self.mask_loss(basis.float() * g[:, :, None, None], state, g)
        out['loss_gate'] = self.gate_prior_loss(gate.float(), state)
    return out

  # ------------------------------------------------------------------------------------------------ auxiliary losses
  @staticmethod
  def _soft_between(v, lo, hi, s):
    return torch.sigmoid((v - lo) * s) * torch.sigmoid((hi - v) * s)

  def state_masks(self, state):
    """The original's hand-written sector masks and weights; grid_x = lateral (+ right), y_forward = forward."""
    bs = state.shape[0]; s = self._mask_sharpness; mr = self.max_range
    gx = self.grid_lateral[None].expand(bs, -1, -1); yf = self.grid_forward[None].expand(bs, -1, -1)
    speed = state[:, 0:1].clamp(min=0.0); steer = state[:, 1:2]; cmd = state[:, 2:2 + self.num_commands]
    v = lambda t: t.view(bs, 1, 1)
    forward_band = self._soft_between(yf, -2.0, v(torch.clamp(10.0 + 3.5 * speed, max=mr)), s)
    radius = torch.sqrt(gx ** 2 + yf ** 2)
    radius_gate = torch.sigmoid((v(torch.clamp(12.0 + 3.5 * speed, min=8.0, max=mr)) - radius) * s)
    angle = torch.atan2(gx, yf.clamp(min=1e-3))
    angle_gate = torch.sigmoid((v(torch.deg2rad(torch.clamp(35.0 + 6.0 * speed, min=25.0, max=60.0))) - angle.abs()) * s)
    forward_mask = forward_band * radius_gate * angle_gate
    turn_max = v(torch.deg2rad(torch.clamp(65.0 + 4.0 * speed, min=45.0, max=80.0)))
    turn_gate = torch.sigmoid((v(torch.clamp(10.0 + 3.0 * speed, min=6.0, max=mr)) - radius) * s)
    left_mask = forward_band * turn_gate * self._soft_between(-angle, 0.0, turn_max, s)
    right_mask = forward_band * turn_gate * self._soft_between(angle, 0.0, turn_max, s)
    near_mask = torch.sigmoid((v(torch.clamp(self._near_radius - 0.6 * speed, min=1.5, max=self._near_radius)) - radius) * s)
    c = lambda name: cmd[:, COMMAND_MAP[name]:COMMAND_MAP[name] + 1]
    steer_left, steer_right = steer.clamp(min=0.0), (-steer).clamp(min=0.0)   # original sign convention (steer unused by default)
    masks = [near_mask, left_mask, forward_mask, right_mask]
    weights = [torch.clamp(1.0 - speed / 4.0, 0.0, 1.0),
               torch.clamp(c('left') + steer_left * 5.0, max=1.0),
               torch.clamp(c('straight') + c('lanefollow') + torch.clamp(1.0 - steer.abs() / 0.2, min=0.0), max=1.0),
               torch.clamp(c('right') + steer_right * 5.0, max=1.0)]
    if self.num_basis >= 5:
      ys = self._soft_between(yf, 0.0, v(torch.clamp(15.0 + 2.0 * speed, max=mr)), s)
      masks.append(ys * torch.exp(-0.5 * ((gx + self._lane_change_offset) / 1.8) ** 2)); weights.append(c('change_left'))
    if self.num_basis >= 6:
      masks.append(ys * torch.exp(-0.5 * ((gx - self._lane_change_offset) / 1.8) ** 2)); weights.append(c('change_right'))
    return masks[:self.num_basis], weights[:self.num_basis]

  def mask_loss(self, gated_basis, state, gate):
    masks, weights = self.state_masks(state)
    eps = 1e-6; total = gated_basis.new_zeros(()); wsum = gated_basis.new_zeros(())
    norm = lambda t: t / t.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
    for k, (m, wt) in enumerate(zip(masks, weights)):
      w_k = wt.squeeze(-1) * gate[:, k]
      log_p = torch.log(norm(gated_basis[:, k].clamp(min=0.0) + eps).flatten(1) + eps)
      q = norm(m.clamp(min=0.0) + eps).flatten(1)
      kl = F.kl_div(log_p, q, reduction='none').sum(dim=1)
      total = total + (kl * w_k).sum(); wsum = wsum + w_k.sum()
    return total / (wsum + eps)   # 0 when no basis carries weight (total is 0 then)

  def gate_prior_loss(self, gate, state):
    cmd_idx = state[:, 2:2 + self.num_commands].argmax(dim=1)
    to_basis = torch.zeros(self.num_commands, device=gate.device, dtype=torch.long)
    for name, b in (('straight', 2), ('lanefollow', 2), ('left', 1), ('right', 3), ('change_left', 1), ('change_right', 3)):
      to_basis[COMMAND_MAP[name]] = b if self.num_basis > b else 0
    idx = torch.arange(self.num_basis, device=gate.device, dtype=gate.dtype)
    prior = torch.exp(-0.5 * ((idx[None] - to_basis[cmd_idx][:, None].to(gate.dtype)) / self.gate_prior_sigma) ** 2)
    prior = prior / (prior.sum(dim=1, keepdim=True) + 1e-6)
    alpha = float(min(max(self.gate_prior_alpha, 0.0), 1.0))
    prior = (1.0 - alpha) / self.num_basis + alpha * prior
    return (prior * (torch.log(prior + 1e-6) - torch.log(gate + 1e-6))).sum(dim=1).mean()


def tfpp_occupancy_inputs(pred_bounding_box, pred_bev_semantic, valid_bev_pixels, out_hw):
  """TF++ outputs -> (objects, drivable) at the head grid, detached. pred_bounding_box[0] = sigmoid centre heatmap
  (B, 5, H, W), classes car 0 / walker 1 / traffic light 2 / stop sign 3 / emergency 4; pred_bev_semantic = logits
  (B, 11, 256, 256) already multiplied by the visible-pixel mask; classes road 1, road line 3, broken road line 4."""
  heat = pred_bounding_box[0].detach().float()
  # The CenterNet labels (and so the predicted heatmap) are drawn column = y right, row = x forward
  # (t_u.bb_vehicle_to_image_system), i.e. TRANSPOSED w.r.t. the lidar histogram and the BEV-semantic labels (column = x
  # forward, row = y right) that define the head's grid -- measured by tools/bev_layout_probe.py (2026-10-18).
  objects = heat[:, [0, 1, 4]].amax(dim=1).transpose(-1, -2)
  if objects.shape[-2:] != tuple(out_hw):
    objects = F.adaptive_max_pool2d(objects[:, None], out_hw)[:, 0]
  prob = F.softmax(pred_bev_semantic.detach().float(), dim=1)
  drv = (prob[:, 1] + prob[:, 3] + prob[:, 4]) * valid_bev_pixels[:, 0].float()
  drivable = F.adaptive_avg_pool2d(drv[:, None], out_hw)[:, 0]
  return objects, drivable
