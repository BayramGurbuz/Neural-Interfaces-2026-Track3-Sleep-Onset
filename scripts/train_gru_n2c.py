"""Candidate 2 sequence row "GRU crop + closeness": final GRU-uw, 10 seeds, stream-start crop augmentation, with the 5 summaries
of the multi-class detector's closeness as extra inputs (value and difference to the first-2-min mean, seq_inputs), re-streamed
from the first window the model sees (a crop restarts them, as at inference). Deterministic GPU settings.

Training rows' closeness = out-of-fold (n2det_final[_tv]/close_oof.npy, stage detmc), as in the dual CV (b2_seq_cv.py n2c gru
--crop). Epochs = median of that CV's LOFO choices (OUT/b2_seq.json "uw_n10_n2c_crop").
--trainval: 500 train+val recordings, detector n2det_final_tv -> SAVE/gru_n2c_tv; default all 540 -> SAVE/gru_n2c_all.
Saved like a mix3 bundle's GRU part (gru_states, gru_mu, gru_sd, gru_omega_b) + gru_n2c = True, read by
train_n2det_final.py mix --gru-from=<dir> (sets n2det_into = ("gru",)).

--arch=tcn (16 Nov, Work B): TCN instead of the GRU (CONFIGS["tcn_n10"], epochs from "tcn_n10_n2c_crop"), saved as tcn_states /
tcn_mu / tcn_sd / tcn_dilations + tcn_n2c -> SAVE/tcn_n2c_{all,tv}.

--det=ctx (17 Nov, Work C): the context detector (train_n2det_final.py detctx: SAVE/n2det_ctx_final[_tv]) instead; a crop at window k
gives window k + j the closeness computed with min(j + 1, 6) windows of context (close_oof_L.npy, n2det_feats.crop_closeness), as at
inference; epochs from "uw_n10_ctx6_crop" -> SAVE/gru_ctx_{all,tv}.

--cropE (18 Nov, Work E): crop mix gru_cv.CROP_E (10 % none, 50 % U[2,5] min, 40 % U[0,10] min) -> SAVE/<arch>_<det>E_{all,tv}.
--seeds=0,10,11,... / --out=<dir name> (20 Nov, Work K): other seeds (same recipe) into another folder; default seeds 0-9, folder as above.
Usage: python scripts/train_gru_n2c.py [--trainval] [--arch=tcn] [--det=ctx] [--cropE] [--seeds=...] [--out=...]
"""
import json
import sys
import time

from repro import set_deterministic

set_deterministic()

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from dist_hgb import VARIANTS, class_stats  # noqa: E402
from dual_cv import OUT  # noqa: E402
from gru_cv import CROP_E, CONFIGS, load_inputs, run_fold  # noqa: E402
from n2det_feats import crop_closeness, summaries  # noqa: E402
from train_hgb import SAVE  # noqa: E402

TV = "--trainval" in sys.argv
ARCH = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--arch=")), "gru")
DET = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--det=")), "mc")
CROPE = "--cropE" in sys.argv  # Work E (18 Nov): crop mix gru_cv.CROP_E -> SAVE/<arch>_ctxE_{all,tv}
SEEDS = [int(x) for x in next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--seeds=")), ",".join(map(str, range(10)))).split(",")]
OUT_NAME = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)  # Work K (20 Nov)

if __name__ == "__main__":
    if DET == "ctx":
        src = SAVE / ("n2det_ctx_final_tv" if TV else "n2det_ctx_final")
        close_L = np.load(src / "close_oof_L.npy").astype(np.float64)
    else:
        src = SAVE / ("n2det_final_tv" if TV else "n2det_final")
        close = np.load(src / "close_oof.npy").astype(np.float64)
    d, recs, F, gain = load_inputs(include_test=True)
    tr = (d.split != "test").to_numpy() if TV else np.ones(len(d), bool)
    cv = json.loads((OUT / "b2_seq.json").read_text())[("uw_n10_ctx6_crop" if DET == "ctx" else "uw_n10_n2c_crop") if ARCH == "gru" else "tcn_n10_n2c_crop"]
    epochs = int(np.median(cv["unseen"]["epochs"] + cv["seen"]["epochs"]))
    cfg = dict(CONFIGS["uw"] if ARCH == "gru" else CONFIGS["tcn_n10"], crop=CROP_E if CROPE else (0.5, 0.35, 0.15))
    _, omega_b = class_stats(d[tr].reset_index(drop=True), VARIANTS["B"])

    def extra(r):
        v = crop_closeness(close_L, r) if DET == "ctx" else close[r]
        return summaries(v, [np.arange(len(r))])

    t0 = time.perf_counter()
    states = []
    for s in SEEDS:
        model, mu, sd = run_fold(d, recs, F, gain, tr, None, cfg, s, stop_epoch=epochs, extra=extra)
        states.append({k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    tag = ("ctx" if DET == "ctx" else "n2c") + ("E" if CROPE else "")
    out = SAVE / (OUT_NAME or (f"{ARCH}_{tag}_tv" if TV else f"{ARCH}_{tag}_all"))
    out.mkdir(exist_ok=True)
    if ARCH == "gru":
        joblib.dump(dict(gru_states=states, gru_mu=mu, gru_sd=sd, gru_omega_b=omega_b, gru_n2c=True, epochs=epochs), out / "model.joblib")
    else:
        joblib.dump(dict(tcn_states=states, tcn_mu=mu, tcn_sd=sd, tcn_dilations=(1, 2, 4, 8, 16, 32), gru_omega_b=omega_b, tcn_n2c=True,
                         epochs=epochs), out / "model.joblib")
    (out / "results.json").write_text(json.dumps(dict(arch=ARCH, det=DET, trainval=TV, epochs=epochs, seeds=list(SEEDS), crop=cfg["crop"], deterministic=True,
                                                      n_inputs=int(len(mu)), time_s=time.perf_counter() - t0), indent=2))
    print(f"saved {out}: {len(states)} seeds x {epochs} epochs, {len(mu)} inputs ({time.perf_counter() - t0:.0f} s)")
