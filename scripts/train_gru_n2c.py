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

Usage: python scripts/train_gru_n2c.py [--trainval] [--arch=tcn]
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
from gru_cv import CONFIGS, load_inputs, run_fold  # noqa: E402
from n2det_feats import summaries  # noqa: E402
from train_hgb import SAVE  # noqa: E402

TV = "--trainval" in sys.argv
ARCH = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--arch=")), "gru")
SEEDS = range(10)

if __name__ == "__main__":
    src = SAVE / ("n2det_final_tv" if TV else "n2det_final")
    close = np.load(src / "close_oof.npy").astype(np.float64)
    d, recs, F, gain = load_inputs(include_test=True)
    tr = (d.split != "test").to_numpy() if TV else np.ones(len(d), bool)
    cv = json.loads((OUT / "b2_seq.json").read_text())["uw_n10_n2c_crop" if ARCH == "gru" else "tcn_n10_n2c_crop"]
    epochs = int(np.median(cv["unseen"]["epochs"] + cv["seen"]["epochs"]))
    cfg = dict(CONFIGS["uw"] if ARCH == "gru" else CONFIGS["tcn_n10"], crop=(0.5, 0.35, 0.15))
    _, omega_b = class_stats(d[tr].reset_index(drop=True), VARIANTS["B"])

    def extra(r):
        return summaries(close[r], [np.arange(len(r))])

    t0 = time.perf_counter()
    states = []
    for s in SEEDS:
        model, mu, sd = run_fold(d, recs, F, gain, tr, None, cfg, s, stop_epoch=epochs, extra=extra)
        states.append({k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    out = SAVE / (f"{ARCH}_n2c_tv" if TV else f"{ARCH}_n2c_all")
    out.mkdir(exist_ok=True)
    if ARCH == "gru":
        joblib.dump(dict(gru_states=states, gru_mu=mu, gru_sd=sd, gru_omega_b=omega_b, gru_n2c=True, epochs=epochs), out / "model.joblib")
    else:
        joblib.dump(dict(tcn_states=states, tcn_mu=mu, tcn_sd=sd, tcn_dilations=(1, 2, 4, 8, 16, 32), gru_omega_b=omega_b, tcn_n2c=True,
                         epochs=epochs), out / "model.joblib")
    (out / "results.json").write_text(json.dumps(dict(arch=ARCH, trainval=TV, epochs=epochs, seeds=list(SEEDS), crop=cfg["crop"], deterministic=True,
                                                      n_inputs=int(len(mu)), time_s=time.perf_counter() - t0), indent=2))
    print(f"saved {out}: {len(states)} seeds x {epochs} epochs, {len(mu)} inputs ({time.perf_counter() - t0:.0f} s)")
