"""Step C: feature-group ablations with the distribution model (dist_hgb), dual CV.

Each run = base 321 features + one group (or a replacement), same classifier, class edges and
decision as the reference model; reported plain + calibrated, with per-bin contributions.
Groups: dyn, spec, maxmin, maxmin_replace (maxmin instead of the first-2-min *_rel), eye.

Usage: python scripts/ablation_c.py [B|B2] [group ...]
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from causal_features import CausalFeatureStream, core_index, feature_names, PER_CHANNEL
from dist_hgb import VARIANTS, run_clf
from dual_cv import OUT, dual_report, fmt, seen_splits, unseen_splits
from report_bins import contributions
from stream_utils import RANGE_NAMES, load_window_table
from train_hgb import SAVE

WINDOW_GROUPS = ("spec", "eye")
ABLATIONS = {  # name: (window groups, stream groups, drop first-2-min *_rel)
    "dyn": ((), ("dyn",), False),
    "spec": (("spec",), (), False),
    "maxmin": ((), ("maxmin",), False),
    "maxmin_replace": ((), ("maxmin",), True),
    "eye": (("eye",), (), False),
    "dyn_maxmin": ((), ("dyn", "maxmin"), False),
    "dyn_maxmin_spec": (("spec",), ("dyn", "maxmin"), False),
    "dyn_maxmin_spec_eye": (("spec", "eye"), ("dyn", "maxmin"), False),
}


def build_table_ext(window_groups=(), stream_groups=(), drop_rel=False):
    zb = np.load(SAVE / "features_base.npz")
    ze = np.load(SAVE / "features_ext.npz")
    assert (zb["timeline"] == ze["timeline"]).all() and (zb["stop"] == ze["stop"]).all()
    ch = list(zb["ch_names"])
    base_names = [f"{c}_{f}" for c in ch for f in PER_CHANNEL]
    enames = list(ze["names"])
    cols = [i for i, n in enumerate(enames)
            if ("spec" in window_groups and not n.startswith(("AF7-AF8", "blink")))
            or ("eye" in window_groups and n.startswith(("AF7-AF8", "blink")))]
    F = np.concatenate([zb["feats"], ze["feats"][:, cols]], axis=1).astype(np.float64)
    names = feature_names(ch, stream_groups, base_names + [enames[i] for i in cols])
    meta = pd.DataFrame({"timeline": zb["timeline"], "stop": zb["stop"], "split": zb["split"]})
    order = np.lexsort((meta.stop.to_numpy(), meta.timeline.to_numpy()))
    full = np.empty((len(meta), len(names)), dtype=np.float32)
    stream = CausalFeatureStream(5.0, stream_groups, core_index(ch))
    prev = None
    for i in order:
        if meta.timeline.iat[i] != prev:
            stream.reset()
            prev = meta.timeline.iat[i]
        full[i] = stream.update(F[i])
    df = meta.merge(load_window_table(), on=["split", "timeline", "stop"], how="left", validate="1:1")
    keep = np.array([not (drop_rel and n.endswith("_rel")) for n in names])
    return df, full[:, keep], [n for n, k in zip(names, keep) if k]


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "B"
    todo = sys.argv[2:] or list(ABLATIONS)
    edges = VARIANTS[variant]
    summary_path = OUT / f"ablation_{variant}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in todo:
        wg, sg, drop = ABLATIONS[name]
        t0 = time.perf_counter()
        df, X, names = build_table_ext(wg, sg, drop)
        m = (df.split != "test").to_numpy()
        d, X = df[m].reset_index(drop=True), X[m]
        protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
        print(f"=== {name}: {X.shape[1]} features (built in {time.perf_counter() - t0:.0f} s)", flush=True)
        runs = {proto: run_clf(X, d, splits, edges) for proto, splits in protocols.items()}
        preds = {proto: runs[proto]["dist_continuous"][0] for proto in protocols}
        info = {f"n_iter_{p}": runs[p]["dist_continuous"][1] for p in protocols}
        res = dual_report(f"{variant}+{name}", d, protocols, preds,
                          dict(n_features=X.shape[1], **info, time_s=time.perf_counter() - t0))
        res["contrib_plain"] = {p: dict(zip(RANGE_NAMES, contributions(d, protocols[p], preds[p])[0].tolist()))
                                for p in protocols}
        for p in protocols:
            np.savez(OUT / f"abl_{variant}_{name}_{p}_preds.npz", *preds[p])
        summary[name] = res
        summary_path.write_text(json.dumps(summary, indent=2))
        print(fmt(res), f"({res['time_s']:.0f} s)", flush=True)
        for p in protocols:
            c = res["contrib_plain"][p]
            print(f"    {p:6s} contrib: " + "  ".join(f"{k} {v:.2f}" for k, v in c.items()), flush=True)
