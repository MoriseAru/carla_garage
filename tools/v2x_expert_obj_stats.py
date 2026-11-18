"""Read-only: does the TF++ training set store the expert's speed-reducing object (LOCAL HANDOVER 10371)?
Same 20,000 random frames (seed 0, setting 'all') as the token probes. For each frame reads the current measurement
(speed_reduced_by_obj_{type,id,distance}, vehicle_affecting_id, walker_affecting_id) and the current boxes file, and
checks whether the id matches a box id (and its class), and whether it is among the K=16 nearest vehicles within 64 m
(the token candidate set; distance from the box 'distance' field). Writes a json summary.
"""
import argparse, collections, gzip, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "team_code"))
import config as cfgmod, data as datamod  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/work/gn21/n21001/carla_garage_data_root")
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--out", default="/work/gn21/n21001/V2XState_Real/tmp/v2x_expert_obj_stats.json")
a = ap.parse_args()

cfg = cfgmod.GlobalConfig()
cfg.initialize(root_dir=[a.root], setting="all", use_v2x=1, use_v2x_bev=0)
ds = datamod.CARLA_Data(root=cfg.data_roots, config=cfg, estimate_class_distributions=False, estimate_sem_distribution=False,
                        shared_dict=None, rank=0)
idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[: a.n].tolist()
cur = cfg.seq_len - 1
print("dataset", len(ds), "frames; seq_len", cfg.seq_len, flush=True)


def rd(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


C = collections.Counter(); types = collections.Counter(); match_cls = collections.Counter(); dist_m = collections.defaultdict(list)
ex = []
for j, i in enumerate(idx):
    mfile = str(ds.measurements[i][0], encoding="utf-8") + f"/{ds.sample_start[i] + cur:04}.json.gz"
    bfile = str(ds.boxes[i][cur], encoding="utf-8")
    m = rd(mfile); b = rd(bfile)
    if j == 0:
        print("files", mfile, bfile, flush=True)
    C["frames"] += 1
    if "speed_reduced_by_obj_type" not in m:
        C["field_missing"] += 1; continue
    t, oid, d = m["speed_reduced_by_obj_type"], m["speed_reduced_by_obj_id"], m["speed_reduced_by_obj_distance"]
    C["vehicle_affecting_id_set"] += m.get("vehicle_affecting_id") is not None
    C["walker_affecting_id_set"] += m.get("walker_affecting_id") is not None
    if t is None and oid is None:
        C["empty"] += 1; continue
    C["nonempty"] += 1
    tt = str(t).split(".")[0] if t is not None else "None"
    types[str(t)] += 1
    byid = {x.get("id"): x for x in b if x.get("id") is not None}
    hit = byid.get(oid)
    key = "matched" if hit is not None else "unmatched"
    C[f"{tt}_{key}"] += 1
    if d is not None:
        dist_m[f"{tt}_{key}"].append(float(d))
    if hit is not None:
        match_cls[f"{tt}->{hit.get('class')}"] += 1
        if hit.get("class") == "car":
            cars = sorted((x for x in b if x.get("class") == "car" and x.get("distance", 1e9) < 64.0), key=lambda x: x["distance"])[:16]
            C[f"{tt}_in_token_set"] += any(x.get("id") == oid for x in cars)
    elif len(ex) < 10:
        ex.append(dict(file=mfile, type=t, id=oid, distance=d, box_ids=sorted(byid)[:20]))
    if j % 2000 == 0:
        print(j, dict(C), flush=True)

res = dict(counts=dict(C), type_counts=dict(types), matched_box_class=dict(match_cls),
           distance_quantiles={k: np.percentile(v, [5, 25, 50, 75, 95]).round(2).tolist() for k, v in dist_m.items()},
           unmatched_examples=ex)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1)); print("STATS_DONE")
