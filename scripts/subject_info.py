"""Diagnostics on subject-specific information (not a submission model).

1. Sleep-onset latency (= stream length, first N2) on train+val: distribution, between/within-subject
   variance, ICC(1) over subjects with >= 2 nights; how well "median of the subject's other nights"
   predicts a night versus the global median (MAE of the latency itself, not the metric).
2. Subject identification from the signal: per recording, the mean of the 62 window features over the
   first 60 s / 120 s (and the whole recording as an upper bound), standardised; leave-one-recording-out
   nearest neighbour among all other train+val recordings (correlation distance). Top-1 accuracy =
   nearest recording belongs to the same subject; chance = (n_subject_recs - 1) / (n_recs - 1) averaged.

Usage: python scripts/subject_info.py
"""
import numpy as np
import pandas as pd

from gru_cv import load_inputs

if __name__ == "__main__":
    d, recs, F, _ = load_inputs()
    rec = d.groupby("timeline").agg(subject=("subject", "first"), onset=("stop", "max"),
                                    n_win=("stop", "size")).reset_index()
    print(f"{len(rec)} recordings, {rec.subject.nunique()} subjects")
    q = rec.onset.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(0).to_dict()
    print(f"onset latency s: mean {rec.onset.mean():.0f}, sd {rec.onset.std():.0f}, quantiles {q}")
    for col, name in (("onset", "latency"), ("log_onset", "log latency")):
        if col == "log_onset":
            rec[col] = np.log(rec.onset)
        g = rec.groupby("subject")[col]
        multi = rec[rec.subject.map(g.size()) >= 2]
        gm = multi.groupby("subject")[col]
        k = gm.size().to_numpy()
        n_sub = len(k)
        grand = multi[col].mean()
        ssb = float((k * (gm.mean().to_numpy() - grand) ** 2).sum())
        ssw = float(((multi[col] - multi.subject.map(gm.mean())) ** 2).sum())
        msb, msw = ssb / (n_sub - 1), ssw / (len(multi) - n_sub)
        k0 = (len(multi) - (k ** 2).sum() / len(multi)) / (n_sub - 1)
        icc = (msb - msw) / (msb + (k0 - 1) * msw)
        print(f"{name}: subjects with >=2 nights {n_sub} ({len(multi)} recs, {k.min()}-{k.max()} nights); "
              f"between-subject SD {np.sqrt(max(msb - msw, 0) / k0):.2f}, within SD {np.sqrt(msw):.2f}, "
              f"ICC(1) = {icc:.3f}")
    # leave-one-night-out: subject median of other nights vs global median
    glob = rec.onset.median()
    rows = []
    for s, grp in rec.groupby("subject"):
        if len(grp) < 2:
            continue
        for i, r in grp.iterrows():
            others = grp.onset.drop(i)
            rows.append((abs(r.onset - others.median()), abs(r.onset - glob), len(others)))
    a = np.array(rows)
    print(f"latency MAE (s) over {len(a)} nights of multi-night subjects: subject-median-of-others "
          f"{a[:, 0].mean():.0f} vs global median {a[:, 1].mean():.0f}; "
          f"by n_other nights: " + ", ".join(f"{n}: {a[a[:, 2] == n, 0].mean():.0f} (n={int((a[:, 2] == n).sum())})"
                                            for n in sorted(set(a[:, 2]))))

    # subject identification
    tl = rec.timeline.to_numpy()
    sub = rec.subject.to_numpy()
    n_sub_recs = pd.Series(sub).map(pd.Series(sub).value_counts()).to_numpy()
    multi = n_sub_recs >= 2
    chance = ((n_sub_recs[multi] - 1) / (len(rec) - 1)).mean()
    idx_by_tl = {d.timeline[r[0]]: r for r in recs}
    for horizon, name in ((12, "first 60 s"), (24, "first 120 s"), (None, "whole recording")):
        E = np.stack([F[idx_by_tl[t][:horizon]].mean(0) for t in tl])
        E = (E - E.mean(0)) / (E.std(0) + 1e-9)
        E = E - E.mean(1, keepdims=True)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        S = E @ E.T
        np.fill_diagonal(S, -np.inf)
        nn = S.argmax(1)
        top1 = (sub[nn] == sub)[multi].mean()
        # top-5 (any of 5 nearest from the same subject)
        top5 = np.array([(sub[np.argsort(-S[i])[:5]] == sub[i]).any() for i in range(len(rec))])[multi].mean()
        print(f"identification ({name}, 62 feats): top-1 {top1:.3f}, top-5 {top5:.3f}, chance {chance:.3f} "
              f"(queries = {multi.sum()} nights of multi-night subjects)")
