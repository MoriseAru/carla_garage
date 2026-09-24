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


def bucket_of(track_id, salt=0) -> int:
    """Deterministic penetration bucket of one actor. salt=0 reproduces the historical hash (every result so far);
    salt>0 draws a different silent set -- so that eval seeds also sample WHICH vehicles stay silent (2026-09-24 concern:
    with the unsalted hash the silent set may be identical across eval seeds and the SE of the penetration curve understated)."""
    key = str(track_id) if salt == 0 else f"{salt}:{track_id}"
    return zlib.crc32(key.encode("utf-8")) % BUCKETS


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


def rasterize_states(states, mask, min_x=-32.0, max_x=32.0, min_y=-32.0, max_y=32.0, pixels_per_meter=4.0, width=2.0):
    """Late fusion: draw the received vehicle states into the ego BEV grid used by the lidar histogram.
    states (bs, K, 7) normalised as _state(); mask (bs, K). Returns (bs, 3, H, W): [occupancy, vx/10, vy/10] inside each
    vehicle's oriented box (length from the message, width assumed `width` m). Pixel layout is the one of
    CARLA_Data.lidar_to_histogram_features: [C, iy, ix] with iy indexing CARLA y (right) from min_y and ix indexing
    CARLA x (front) from min_x. Empty mask -> all zeros, so the channel is exactly silent without messages."""
    bs, k, _ = states.shape
    dev, dt = states.device, states.dtype
    H = int(round((max_y - min_y) * pixels_per_meter)); W = int(round((max_x - min_x) * pixels_per_meter))
    xs = min_x + (torch.arange(W, device=dev, dtype=dt) + 0.5) / pixels_per_meter          # (W,) CARLA x of column centres
    ys = min_y + (torch.arange(H, device=dev, dtype=dt) + 0.5) / pixels_per_meter          # (H,) CARLA y of row centres
    Yg, Xg = torch.meshgrid(ys, xs, indexing="ij")                                          # (H, W)
    cx = states[..., 0] * 32.0; cy = states[..., 1] * 32.0; vx = states[..., 2] * 10.0; vy = states[..., 3] * 10.0
    c = states[..., 4]; sn = states[..., 5]; half_l = states[..., 6] * 5.0 / 2.0
    dx = Xg[None, None] - cx[..., None, None]; dy = Yg[None, None] - cy[..., None, None]       # (bs, K, H, W)
    u = dx * c[..., None, None] + dy * sn[..., None, None]                                    # along the vehicle's heading
    v = -dx * sn[..., None, None] + dy * c[..., None, None]                                   # across
    inside = (u.abs() <= half_l[..., None, None]) & (v.abs() <= width / 2.0) & (mask[..., None, None] > 0.5)   # (bs, K, H, W)
    inside_f = inside.to(dt)
    occ = inside_f.amax(dim=1)
    # velocity channels: where boxes overlap take the nearest vehicle's (slots are distance-sorted, so the first valid wins)
    first = inside_f.cumsum(dim=1) <= 1.0
    sel = (inside_f * first.to(dt))
    vx_map = (sel * (vx / 10.0)[..., None, None]).sum(dim=1); vy_map = (sel * (vy / 10.0)[..., None, None]).sum(dim=1)
    return torch.stack((occ, vx_map, vy_map), dim=1)


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


def rotation_angle(track_id, salt=0):
    """Deterministic angle in [0, 2pi) for one actor: the same vehicle gets the same rotation for the whole route, so a
    randomised token set is temporally coherent (a ghost car driving a rotated trajectory) instead of flickering noise."""
    return 2.0 * math.pi * (zlib.crc32(f"{track_id}:{salt}".encode("utf-8")) / 2 ** 32)


def rotate_about_ego(x, y, yaw, angle):
    """Rotate a reported state rigidly about the ego. Distance, speed and size are preserved; only WHERE the vehicle is
    (and which way it points) becomes wrong -- the control for 'does the content of this message matter, or just its
    presence?'."""
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y, math.atan2(math.sin(yaw + angle), math.cos(yaw + angle))


def unmatched_detections(sender_xy, det_xy, match_dist=2.5):
    """Deployable estimate of the vehicles that did NOT send: ego-perceived vehicle detections (previous frame, ego frame,
    metres) that lie farther than `match_dist` from every received message. Vehicles that neither send nor are seen are
    invisible to this estimate, so the derived availability sending/(sending+unmatched) is biased UPWARD vs the oracle."""
    if len(det_xy) == 0:
        return 0
    det = np.asarray(det_xy, dtype=np.float64).reshape(-1, 2)
    if len(sender_xy) == 0:
        return int(len(det))
    snd = np.asarray(sender_xy, dtype=np.float64).reshape(-1, 2)
    d2 = ((det[:, None, :] - snd[None, :, :]) ** 2).sum(-1)
    return int((d2.min(axis=1) > match_dist ** 2).sum())


def route_use_base(history, thresh):
    """Availability router: True = fall back to the V2X-free base model. `history` is a deque of per-frame
    kept/in_radius ratios (frames with no vehicle in radius contribute nothing). Undecidable (empty history) -> False,
    i.e. keep using the plug-in. Averaging over the window keeps the decision from flapping frame to frame."""
    if thresh <= 0.0 or not len(history):
        return False
    return (sum(history) / len(history)) < thresh


def points_in_box(rel_pos, rel_yaw, extent, lidar):
    """Ego-lidar hits inside a vehicle's box. Same computation as data_agent.get_points_in_bbox (the recorded
    `num_points`): `rel_pos` (3,) and `rel_yaw` are the vehicle's pose in the ego frame, `lidar` (N,3) ego-frame points."""
    if lidar is None or len(lidar) == 0:
        return -1
    c, s = math.cos(rel_yaw), math.sin(rel_yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    v = (rot.T @ (np.asarray(lidar, dtype=np.float64)[:, :3] - np.asarray(rel_pos, dtype=np.float64)).T).T
    x, y, z = float(extent[0]), float(extent[1]), float(extent[2])
    return int(((v[:, 0] < x) & (v[:, 0] > -x) & (v[:, 1] < y) & (v[:, 1] > -y) & (v[:, 2] < z) & (v[:, 2] > -z)).sum())


def select_delayed(history, latency_frames):
    """`history` = per-actor deque of per-frame records (newest last). Returns the record `latency_frames` frames old, or
    None if the message has not "arrived" yet (fewer records than the latency)."""
    if latency_frames <= 0:
        return history[-1] if len(history) else None
    idx = len(history) - 1 - int(latency_frames)
    return history[idx] if idx >= 0 else None


def coop_states_from_world(ego_matrix, ego_yaw, vehicles, k=16, rate=1.0, radius=64.0, relative_transform=None,
                           lidar=None, hidden_pts=5, tokens="all", pos_noise=0.0, vel_noise=0.0, drop=0.0, rng=None,
                           randomize="none", salt=0, bucket_salt=0, stats=None):
    """Closed-loop counterpart. `vehicles` = iterable of (actor_id, actor_matrix(4x4), yaw_rad, speed, length[, extent(3,)]);
    `relative_transform(ego_matrix, actor_matrix)` must be team_code.transfuser_utils.get_relative_transform so the
    frame matches the recorded boxes exactly. Returns torch tensors (1,k,7), (1,k).
    Inference-time message model (all default to the training condition):
      tokens: 'all' | 'hidden' (only vehicles with <= hidden_pts ego-lidar hits in their box, needs `lidar` + extents)
              | 'visible' (the complement) -- does the gain come from vehicles the ego cannot see?
      pos_noise / vel_noise: Gaussian noise (m, m/s) on the reported position / speed;  drop: per-message loss probability.
      randomize: 'none' | 'all' | 'hidden' | 'visible' -- that subset keeps its token (count, distance, speed and size
              unchanged) but is rigidly rotated about the ego by a per-actor constant angle, so the message is present
              and plausible but no longer describes where the vehicle actually is. Separates 'the content of this
              message matters' from 'a token being there matters'.
    `stats` (dict) receives counts: total, in_radius (all vehicles within radius), sending (whose message arrived),
    hidden, kept (tokens given to the model), dropped, randomized, and sender_xy (ego-frame positions of the arrived
    messages). bucket_salt: 0 = historical hash; >0 = a different silent set per eval seed."""
    rng = rng if rng is not None else np.random.default_rng()
    cands = []
    # Counters (all vehicles within `radius`, whether or not they send):
    #   in_radius = vehicles physically present within the radius  (oracle denominator for the router)
    #   sending   = of those, the ones whose message arrived (passed the penetration gate and the drop model)
    #   kept      = tokens actually handed to the model (after the hidden/visible filter and the K-slot cap)
    # The penetration gate must come AFTER the in_radius count -- 2026-09-20 it came before, so in_radius only counted
    # senders and sending/in_radius was 1.0 at every penetration rate; the router never fired.
    st = dict(total=0, in_radius=0, sending=0, hidden=0, kept=0, dropped=0, randomized=0)
    sender_xy = []
    need_vis = (tokens != "all") or (randomize in ("hidden", "visible"))
    for veh in vehicles:
        aid, mat, yaw, spd, length = veh[:5]
        extent = veh[5] if len(veh) > 5 else (length / 2.0, 1.0, 0.8)
        st["total"] += 1
        rel = relative_transform(ego_matrix, mat)
        x, y = float(rel[0]), float(rel[1])
        dist = math.hypot(x, y)
        if dist > radius:
            continue
        st["in_radius"] += 1
        if rate < 1.0 and bucket_of(aid, bucket_salt) >= int(rate * BUCKETS):
            continue
        if drop > 0.0 and rng.random() < drop:
            st["dropped"] += 1
            continue
        st["sending"] += 1
        sender_xy.append((x, y))     # true position of every arrived message, before the K cap (router proxy matches against it)
        ryaw = math.atan2(math.sin(yaw - ego_yaw), math.cos(yaw - ego_yaw))
        hid = False
        if need_vis:
            n = points_in_box(np.asarray(rel[:3], dtype=np.float64), ryaw, extent, lidar)
            hid = (0 <= n <= hidden_pts)
            st["hidden"] += int(hid)
            if (tokens == "hidden" and not hid) or (tokens == "visible" and hid):
                continue
        if randomize != "none" and (randomize == "all" or (randomize == "hidden") == hid):
            x, y, ryaw = rotate_about_ego(x, y, ryaw, rotation_angle(aid, salt))
            st["randomized"] += 1
        if pos_noise > 0.0:
            x += float(rng.normal(0.0, pos_noise)); y += float(rng.normal(0.0, pos_noise))
        spd = float(spd) + (float(rng.normal(0.0, vel_noise)) if vel_noise > 0.0 else 0.0)
        cands.append((dist, _state(x, y, ryaw, max(0.0, spd), float(length))))
    cands.sort(key=lambda t: t[0])
    states = np.zeros((k, STATE_DIM), dtype=np.float32)
    mask = np.zeros((k,), dtype=np.float32)
    for j, (_, vec) in enumerate(cands[:k]):
        states[j], mask[j] = vec, 1.0
    st["kept"] = int(mask.sum())
    if stats is not None:
        stats.update(st)
        stats["sender_xy"] = sender_xy
    return torch.from_numpy(states)[None], torch.from_numpy(mask)[None]
