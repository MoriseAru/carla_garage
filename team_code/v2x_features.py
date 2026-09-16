"""V2XState plug-in for TransFuser++: cooperative vehicle states as extra decoder memory tokens.

Premise (same as the DiffusionDrive plug-in): connected vehicles broadcast their own ground-truth state; nothing at the
sensor/feature level is shared. Which vehicles are connected is a deterministic hash of the actor id, so a penetration
rate `rate` keeps a fixed random subset. Slot vector (K nearest connected vehicles, nearest first):
    [x/32, y/32, vx/10, vy/10, cos(yaw), sin(yaw), length/5]   in the TF++ ego/box frame (CARLA: x forward, y right).
Training reads the states from the recorded `boxes/*.json.gz` (positions already relative to the ego, same augmentation
as the detection labels); closed-loop reads them from the CARLA world through the privileged CarlaDataProvider.
"""
import math
import zlib

import numpy as np
import torch

STATE_DIM = 7
BUCKETS = 1000


def bucket_of(track_id) -> int:
    return zlib.crc32(str(track_id).encode("utf-8")) % BUCKETS


def _state(x, y, yaw, speed, length):
    return [x / 32.0, y / 32.0, speed * math.cos(yaw) / 10.0, speed * math.sin(yaw) / 10.0, math.cos(yaw), math.sin(yaw), length / 5.0]


def coop_states_from_boxes(boxes, k=16, y_augmentation=0.0, yaw_augmentation=0.0, radius=64.0, hidden_pts=5, hazard_range=30.0,
                           hazard_lat=12.0, moving=1.0):
    """From one frame's recorded boxes (list of dicts). Returns (states (k,7) f32, mask (k,) f32, bucket (k,) i64,
    hidden (k,) f32, hazard (k,) f32). hidden: <= hidden_pts ego-lidar points inside the box (the ego's own sensors do not
    see it); hazard: moving vehicle ahead within hazard_range m and |lateral| <= hazard_lat m -- the case where a shared
    state carries information the ego cannot get itself.
    Applies the same rotation/translation augmentation as CARLA_Data.get_bbox_label so the tokens stay aligned with the
    augmented lidar BEV. No visibility filter: hidden connected vehicles are exactly what the plug-in adds."""
    aug = math.radians(yaw_augmentation)
    c, s = math.cos(aug), math.sin(aug)
    cands = []
    for b in boxes:
        if b.get("class") != "car":
            continue
        px, py = float(b["position"][0]), float(b["position"][1]) - float(y_augmentation)
        x, y = c * px + s * py, -s * px + c * py          # rotation_matrix.T @ (position - translation)
        dist = math.hypot(x, y)
        if dist > radius:
            continue
        yaw = float(b.get("yaw", 0.0)) - aug
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        spd = b.get("speed", 0.0)
        spd = 0.0 if spd is None or (isinstance(spd, float) and math.isnan(spd)) else float(spd)
        ext = b.get("extent", [2.0, 1.0, 0.8])
        npts = b.get("num_points")
        hid = 1.0 if (npts is not None and 0 <= int(npts) <= hidden_pts) else 0.0
        haz = 1.0 if (0.0 < x <= hazard_range and abs(y) <= hazard_lat and spd > moving) else 0.0
        cands.append((dist, _state(x, y, yaw, spd, 2.0 * float(ext[0])), bucket_of(b.get("id", len(cands))), hid, haz))
    cands.sort(key=lambda t: t[0])
    states = np.zeros((k, STATE_DIM), dtype=np.float32)
    mask = np.zeros((k,), dtype=np.float32)
    bucket = np.full((k,), BUCKETS, dtype=np.int64)     # padded slots never cooperate
    hidden = np.zeros((k,), dtype=np.float32)
    hazard = np.zeros((k,), dtype=np.float32)
    for j, (_, vec, bk, hid, haz) in enumerate(cands[:k]):
        states[j], mask[j], bucket[j], hidden[j], hazard[j] = vec, 1.0, bk, hid, haz
    return states, mask, bucket, hidden, hazard


def apply_rate(states, mask, bucket, rate):
    """Penetration gating on batched tensors: keep slot iff valid and hash bucket < rate * BUCKETS."""
    if rate >= 1.0:
        return states, mask
    if rate <= 0.0:
        return states, torch.zeros_like(mask)
    keep = (bucket < int(rate * BUCKETS)).to(mask.dtype)
    return states, mask * keep


def apply_random_rate(states, mask, bucket, p_zero=0.25, generator=None):
    """Training-time penetration dropout: per sample, with probability p_zero no vehicle cooperates (rate 0), otherwise the
    rate is drawn uniformly from (0, 1]. Slots are kept iff bucket < rate * BUCKETS, so a given vehicle's participation is
    still consistent within a sample. Returns (states, mask, rates)."""
    bs = mask.shape[0]
    rates = torch.rand(bs, device=mask.device, generator=generator)
    rates = torch.where(torch.rand(bs, device=mask.device, generator=generator) < p_zero, torch.zeros_like(rates), rates)
    keep = (bucket < (rates[:, None] * BUCKETS).long()).to(mask.dtype)
    return states, mask * keep, rates


def apply_visibility_dropout(states, mask, hidden, keep_visible=0.5, generator=None):
    """Scheme D3: tokens of vehicles the ego's own sensors already see are dropped at random (kept with prob keep_visible,
    per slot), tokens of HIDDEN vehicles are never dropped. The cooperative head therefore learns (i) tokens for hidden
    vehicles are fully reliable -> it may depend on them, which is where the closed-loop gain comes from; (ii) visible
    vehicles may be missing from the tokens -> it must keep reading them from the sensors, so partial penetration does
    not blind it to unconnected-but-visible traffic. Returns (states, mask)."""
    keep = (torch.rand(mask.shape, device=mask.device, generator=generator) < keep_visible).to(mask.dtype)
    keep = torch.where(hidden > 0.5, torch.ones_like(keep), keep)
    return states, mask * keep


def coop_states_from_world(ego_matrix, ego_yaw, vehicles, k=16, rate=1.0, radius=64.0, relative_transform=None):
    """Closed-loop counterpart. `vehicles` = iterable of (actor_id, actor_matrix(4x4), yaw_rad, speed, length);
    `relative_transform(ego_matrix, actor_matrix)` must be team_code.transfuser_utils.get_relative_transform so the
    frame matches the recorded boxes exactly. Returns torch tensors (1,k,7), (1,k)."""
    cands = []
    for aid, mat, yaw, spd, length in vehicles:
        if rate < 1.0 and bucket_of(aid) >= int(rate * BUCKETS):
            continue
        rel = relative_transform(ego_matrix, mat)
        x, y = float(rel[0]), float(rel[1])
        dist = math.hypot(x, y)
        if dist > radius:
            continue
        ryaw = math.atan2(math.sin(yaw - ego_yaw), math.cos(yaw - ego_yaw))
        cands.append((dist, _state(x, y, ryaw, float(spd), float(length))))
    cands.sort(key=lambda t: t[0])
    states = np.zeros((k, STATE_DIM), dtype=np.float32)
    mask = np.zeros((k,), dtype=np.float32)
    for j, (_, vec) in enumerate(cands[:k]):
        states[j], mask[j] = vec, 1.0
    return torch.from_numpy(states)[None], torch.from_numpy(mask)[None]
