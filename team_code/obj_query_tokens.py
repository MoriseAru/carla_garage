"""Implicit object-query tokens for the TF++ planner decoder (LOCAL HANDOVER 10947; LEAD-style, without radar).

N learnable queries cross-attend to the detection BEV feature grid (`bev_feature_grid`, the input of the CenterNet
head; 64 channels, 64 x 64 for the B2D config, optionally average-pooled by `pool`) with a 2-D sine position encoding,
plus the planner's ego velocity/command token (HANDOVER 10947 amendment: the planner's 8 x 8 BEV tokens do not hold
precise vehicle positions). Each query predicts one vehicle (x/32, y/32, v/10) and an existence logit, supervised with
Hungarian matching against the K nearest vehicles. What the planner receives is NOT the decoded state: it is
Linear(query feature) + a sine encoding of the PREDICTED (x, y), appended to the decoder memory where the explicit
state tokens of `use_v2x` would go. Nothing is detached, so the planning loss can reshape the query features.
Inference needs no extra input.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment


def sine_xy(xy_m, dim, scale=2 * math.pi, extent=32.0, temperature=10000.0):
  """(..., 2) metric positions -> (..., dim) DETR-style sine/cosine encoding (dim/2 per coordinate)."""
  half = dim // 2
  pos = (xy_m / extent + 1.0) * 0.5 * scale            # [-extent, extent] -> [0, scale]
  i = torch.arange(half // 2, device=xy_m.device, dtype=xy_m.dtype)
  freq = temperature ** (2 * i / half)
  out = []
  for c in range(2):
    a = pos[..., c:c + 1] / freq
    out += [torch.sin(a), torch.cos(a)]
  return torch.cat(out, dim=-1)


class ObjQueryTokens(nn.Module):

  def __init__(self, d_bev, d_ego, n_query=16, d_model=256, n_layers=4, n_heads=8, d_ff=1024, dropout=0.1, d_out=None,
               pool=1):
    super().__init__()
    self.n_query, self.pool, self.d_model = n_query, int(pool), d_model
    self.query = nn.Parameter(torch.randn(1, n_query, d_model))   # N(0, 1) as nn.Embedding / DETR: with std 0.02 the queries start near-identical and only dropout noise separates them (smoke 2026-11-19: eval collapse)
    self.bev_in = nn.Conv2d(d_bev, d_model, kernel_size=1)
    self.ego_in = nn.Identity() if d_ego == d_model else nn.Linear(d_ego, d_model)
    layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, activation=nn.GELU(), batch_first=True)
    self.decoder = nn.TransformerDecoder(layer, n_layers, norm=nn.LayerNorm(d_model))
    self.state_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, 3))
    self.exist_head = nn.Linear(d_model, 1)
    d_out = d_out or d_model
    self.to_planner = nn.Linear(d_model, d_out)
    self.d_out = d_out

  def grid_pos(self, h, w, device, dtype):
    """(h*w, d_model) sine encoding of the normalised cell centres (row, col) in [-1, 1]."""
    r = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h * 2 - 1
    c = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w * 2 - 1
    rc = torch.stack(torch.meshgrid(r, c, indexing='ij'), dim=-1).reshape(-1, 2)
    return sine_xy(rc, self.d_model, extent=1.0)

  def forward(self, bev_grid, ego_token):
    """bev_grid (bs, d_bev, H, W), ego_token (bs, 1, d_ego) -> tokens (bs, N, d_out), pred dict(state (bs, N, 3)
    normalised, exist_logit (bs, N))."""
    bs = bev_grid.shape[0]
    if self.pool > 1:
      bev_grid = F.avg_pool2d(bev_grid, self.pool)
    m = self.bev_in(bev_grid)
    h, w = m.shape[-2:]
    m = m.flatten(2).transpose(1, 2) + self.grid_pos(h, w, m.device, m.dtype)[None]
    memory = torch.cat((m, self.ego_in(ego_token)), dim=1)
    self.last_memory_shape = (int(h), int(w), int(memory.shape[1]))   # logging only
    feat = self.decoder(self.query.expand(bs, -1, -1), memory)
    state = self.state_head(feat)
    exist = self.exist_head(feat).squeeze(-1)
    tok = self.to_planner(feat) + sine_xy(state[..., :2] * 32.0, self.d_out)
    return tok, {'state': state, 'exist_logit': exist}


def hungarian_loss(pred, gt_state, gt_mask, w_l1=5.0, w_exist=1.0):
  """pred: dict from ObjQueryTokens; gt_state (bs, K, 3) normalised (x/32, y/32, v/10); gt_mask (bs, K).
  Matching cost = w_l1 * L1(state) - w_exist * sigmoid(exist). Loss = w_l1 * mean matched L1 (per matched coordinate,
  averaged over matched pairs in the batch) + w_exist * BCE(exist) over all queries. Returns (loss, stats)."""
  st, ex = pred['state'], pred['exist_logit']
  bs, n, _ = st.shape
  tgt_exist = torch.zeros_like(ex)
  l1_terms = []
  with torch.no_grad():
    cost_all = w_l1 * torch.cdist(st.float(), gt_state.float(), p=1) - w_exist * torch.sigmoid(ex.float())[..., None]   # (bs, N, K)
  pos_err = []
  for b in range(bs):
    valid = torch.nonzero(gt_mask[b] > 0.5).squeeze(1)
    if valid.numel() == 0:
      continue
    c = cost_all[b][:, valid].cpu().numpy()
    qi, gj = linear_sum_assignment(c)
    qi = torch.as_tensor(qi, device=st.device); gj = valid[torch.as_tensor(gj, device=st.device)]
    tgt_exist[b, qi] = 1.0
    l1_terms.append(torch.abs(st[b, qi] - gt_state[b, gj]))
    pos_err.append(torch.linalg.norm((st[b, qi, :2] - gt_state[b, gj, :2]).detach() * 32.0, dim=-1))
  l_l1 = torch.cat(l1_terms).mean() if l1_terms else st.sum() * 0.0
  l_ex = F.binary_cross_entropy_with_logits(ex, tgt_exist)
  stats = {'matched': int(sum(t.shape[0] for t in l1_terms)),
           'pos_err_m': float(torch.cat(pos_err).mean()) if pos_err else float('nan'),
           'l1': float(l_l1.detach()), 'bce': float(l_ex.detach())}
  return w_l1 * l_l1 + w_exist * l_ex, stats


def labels_from_boxes(boxes, k=16, radius=32.0, y_augmentation=0.0, yaw_augmentation=0.0):
  """K nearest 'car' boxes within radius (same selection/augmentation as v2x_features.coop_states_from_boxes):
  (k, 3) [x/32, y/32, speed/10] and (k,) mask."""
  import v2x_features   # pylint: disable=import-outside-toplevel
  states, mask, *_ = v2x_features.coop_states_from_boxes(boxes, k=k, y_augmentation=y_augmentation,
                                                         yaw_augmentation=yaw_augmentation, radius=radius)
  speed = np.hypot(states[:, 2], states[:, 3])            # already /10
  return np.stack([states[:, 0], states[:, 1], speed], axis=1).astype(np.float32), mask
