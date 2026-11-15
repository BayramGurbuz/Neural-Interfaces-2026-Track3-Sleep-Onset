"""B2: sequence models of mix3 (GRU-uw x10, TCN x10, GRU-emb x3) with B2 groups added to the per-window inputs,
dual CV, deterministic GPU settings (repro.set_deterministic).

Per-window feature matrix = the 62 window features of mix3 + the B2 window groups, then the same causal inputs as
gru_cv.seq_inputs (features, features minus the first-2-min running mean, log1p elapsed):
  bip        : + 72 bipolar spectral features                      -> 2 x 134 + 1 = 269 inputs
  bip+epoch  : + 72 bipolar + 113 epoch-aligned (features_b2)      -> 2 x 247 + 1 = 495 inputs
The channel-gain augmentation acts on the original 62 columns only (zero on the new ones).
GRU-emb reuses the saved per-fold CNN encoder outputs (cnn_cnn3_<proto>_f<k>_s<s>.npz, same as cnn3_emb).
Training schedules as before: gru = CONFIGS["uw_n10"] (40 epochs), tcn = CONFIGS["tcn_n10"] (30), gemb = uw with
run_epochs 40, 3 seeds; epoch per fold chosen LOFO on the other folds' mean curve.

  n2det      : + the 5 fold-specific CNN N2-detector summaries (n2det_feats.py) as per-window extras (like the CNN
               embeddings: fold k uses OUT/n2det_<det>_<proto>_f<k>.npy, out-of-fold for every row)
  n2c        : + the 5 summaries of the multi-class detector's closeness (3-seed mean, n2det_feats.fold_features_mc; Work 3)
  old        : no extra group (deterministic rerun of the original inputs)
Usage: python scripts/b2_seq_cv.py VARIANT ARCH [--crop]   (VARIANT old | bip | bip+epoch | n2det | bip+n2det ...;
       ARCH gru | tcn | gemb; --crop = stream-start augmentation 50 / 35 / 15 % (gru_cv.run_fold cfg["crop"]), name suffix _crop)
  writes OUT/gru_<name>_{unseen,seen}_{preds,proba}.npz, name = uw_n10_<v> | tcn_n10_<v> | cnn3_emb_<v>
  (<v> = VARIANT with "+" -> "_")
"""
import json
import sys
import time

from repro import set_deterministic

set_deterministic()

import numpy as np  # noqa: E402

from ablation_c import ABLATIONS, build_table_ext  # noqa: E402
from b2_ablation import b2_table  # noqa: E402
from dist_hgb import class_stats, decide  # noqa: E402
from dual_cv import OUT, seen_splits, unseen_splits  # noqa: E402
from gru_cv import CONFIGS, EDGES, load_inputs, run_fold  # noqa: E402
from n2det_feats import fold_features, fold_features_mc  # noqa: E402
from stream_utils import summarize  # noqa: E402
from train_hgb import SAVE  # noqa: E402

ARCH = {"gru": ("uw_n10", CONFIGS["uw_n10"], 10), "tcn": ("tcn_n10", CONFIGS["tcn_n10"], 10),
        "gemb": ("cnn3_emb", dict(CONFIGS["uw"], run_epochs=40, n_seeds=3), 3)}


def extended_inputs(variant, include_test=False):
    """-> d, recs, F2 (N, 62 + extra), gain; rows as gru_cv.load_inputs."""
    d, recs, F, gain = load_inputs(include_test=include_test)
    wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
    df, _, _ = build_table_ext(wg, sg)
    m = np.ones(len(df), bool) if include_test else (df.split != "test").to_numpy()
    assert (df[m].timeline.to_numpy() == d.timeline.to_numpy()).all() and (df[m].stop.to_numpy() == d.stop.to_numpy()).all()
    parts = [F]
    groups = variant.split("+")
    if "bip" in groups:
        parts.append(np.load(SAVE / "features_b2.npz")["feats_bip"][m].astype(np.float32))
    if "epoch" in groups:
        parts.append(b2_table(("epoch",), df)[0][m].astype(np.float32))
    return d, recs, np.concatenate(parts, axis=1), gain


if __name__ == "__main__":
    variant, arch = sys.argv[1], sys.argv[2]
    base, cfg, n_seeds = ARCH[arch]
    name = f"{base}_{variant.replace('+', '_')}"
    if "--crop" in sys.argv:
        cfg = dict(cfg, crop=(0.5, 0.35, 0.15))
        name += "_crop"
    d, recs, F2, gain = extended_inputs(variant)
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    print(f"{name}: window features {F2.shape[1]} -> inputs {2 * F2.shape[1] + 1 + (64 if arch == 'gemb' else 0)}, "
          f"{n_seeds} seeds, cfg {cfg}", flush=True)
    t_all = time.perf_counter()
    info = {}
    for proto, splits in protocols.items():
        curves, cands, probs = [], [], []
        for k, (tr, te) in enumerate(splits):
            t0 = time.perf_counter()
            runs = []
            if k == 0:
                n2 = [fold_features(proto, j, recs) for j in range(len(splits))] if "n2det" in variant else None
                n2c = [fold_features_mc(proto, j, recs) for j in range(len(splits))] if "n2c" in variant.split("+") else None
            for s in range(n_seeds):
                ex = []
                if arch == "gemb":
                    ex.append(np.load(OUT / f"cnn_cnn3_{proto}_f{k}_s{s}.npz")["emb"].astype(np.float32))
                if "n2det" in variant.split("+"):
                    ex.append(n2[k])
                if "n2c" in variant.split("+"):
                    ex.append(n2c[k])
                extra = np.concatenate(ex, axis=1) if ex else None
                runs.append(run_fold(d, recs, F2, gain, tr, te, cfg, s, extra=extra))
            dte = d[te].reset_index(drop=True)
            _, omega = class_stats(d[tr].reset_index(drop=True), EDGES)
            Pk = [np.mean([r[j] for r in runs], axis=0) for j in range(len(runs[0]))]
            q = [decide(P.astype(np.float64), None, omega, True, EDGES).astype(np.float32) for P in Pk]
            curves.append([summarize(dte, x)["wbmae"] for x in q])
            cands.append(q)
            probs.append(Pk)
            print(f"  {proto} fold {k}: {time.perf_counter() - t0:.0f} s, curve " + " ".join(f"{x:.1f}" for x in curves[-1]),
                  flush=True)
        C = np.array(curves)
        sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
        np.savez(OUT / f"gru_{name}_{proto}_preds.npz", *[cands[k][j] for k, j in enumerate(sel)])
        np.savez(OUT / f"gru_{name}_{proto}_proba.npz", *[probs[k][j] for k, j in enumerate(sel)])
        info[proto] = dict(epochs=[cfg["eval_every"] * (j + 1) for j in sel],
                           wbmae=float(np.mean([C[k, j] for k, j in enumerate(sel)])))
    info["dual"] = (info["unseen"]["wbmae"] + info["seen"]["wbmae"]) / 2
    info["time_s"] = time.perf_counter() - t_all
    path = OUT / "b2_seq.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary[name] = info
    path.write_text(json.dumps(summary, indent=2))
    print(f"{name}: unseen {info['unseen']['wbmae']:.2f} seen {info['seen']['wbmae']:.2f} dual {info['dual']:.2f} "
          f"epochs {info['unseen']['epochs']} / {info['seen']['epochs']} ({info['time_s']:.0f} s)", flush=True)
