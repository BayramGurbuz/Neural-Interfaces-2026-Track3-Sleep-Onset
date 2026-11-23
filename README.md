# Muse Sleep-Onset — EEG/EMG Foundation Challenge 2026, Track 03

Our entry to **Track 03 (Sleep Onset)** of the [EEG/EMG Foundation Challenge 2026](https://neural-interfaces26.github.io/)
(NeurIPS 2026 Brain & Body Workshop, [Codabench competition 17983](https://www.codabench.org/competitions/17983/)).

**Final result:** sealed W-bMAE (macro) **32.92 s**, **26th** at the close of the sealed phase.

## Task

A Muse headband EEG recording (4 channels: TP9, AF7, AF8, TP10, 128 Hz, µV) is streamed in 5-second windows. For
every window the model predicts the **seconds remaining until the first N2 sleep epoch**, capped at 600 s. Rules:

- **Strictly causal.** The model sees only the current and past windows, and its state is reset for every recording.
- **Scoring is W-bMAE.** It is a binned MAE weighted by the true remaining time: [0,40) ×10, [40,90) ×5, [90,300) ×3,
  [300,600] ×1. It is computed per recording, then averaged. Seen-subject and unseen-subject recordings are weighted
  equally.
- **Sealed streams start mid-recording.** In the sealed evaluation the opening minutes are not given to the model, so
  models must be robust to the stream starting part-way through.

## Approach

The final model (`mixI_540`) is a fixed 50/50 mix of two submissions that had each been proven on the sealed set. The
components' metric-weighted remaining-time distributions are averaged, and the prediction is their continuous weighted
median.

```
raw 5-s window ──► causal features (817: band powers, spectral shape, eye movements, EMAs, slopes,
                   5-min variance / lag-1 autocorrelation, running max/min, first-2-min baselines, elapsed time)
               └─► context N2 detector (small CNN + GRU over the last 6 windows, 3 seeds) ─► causal "closeness" summaries

HGB ctx   (4 HistGradientBoosting distribution members, start-crop augmented table)   ┐ 0.6 (½ + ½)
HGB E-s3p (same members, wider crop augmentation, partial 3-seed averaging)           ┘
GRU ctx   (causal GRU on features + closeness, start-crop augmented, 10 seeds)  ⅔ ┐
TCN       (causal dilated conv, 10 seeds)                                        ⅙ ├ 0.4
GRU-emb   (CNN window embedding + GRU, 3 seeds)                                  ⅙ ┘
                         ──► mixed distribution ──► metric-weighted median ──► seconds to N2
```

Key ingredients:

- **Remaining-time distributions, not point regression.** Each component outputs class probabilities over
  remaining-time bins. The decision is a weighted median using the metric's effective window weights.
- **In-domain N2 detector.** It is trained on the post-onset EEG of the training recordings, which the raw files contain
  but the stream pipeline cuts off. Its causal summaries are extra inputs.
- **Stream-start augmentation.** Training copies are cropped at random starts, and every causal feature, baseline and
  detector context is rebuilt from the crop. This was the single largest sealed-set gain (−2.7 s).
- **Validation.** We used a subject-wise dual CV (unseen-subject 5-fold + seen-subject 5-fold). After the November 16
  sealed update we also used a **shifted** CV, in which held-out streams start 2–5 min into the recording.
- **Compliance.** No leakage of recording length, recording end, subject identity or labelling regularities. A 30-s
  onset grid trick was removed after the November 13 FAQ update. No external data is used in any submitted model.

## Results (sealed, after the November 16 rescoring)

| # | Submission | Idea | Sealed W-bMAE |
|---|---|---|---|
| 1 | final540 | HGB + GRU | 87.99 |
| 4 | mix3_n2det_cropg_540 | + N2 detector, GRU start-crop | 47.42 |
| 5 | cand2_cropGc_540 | multi-class detector, crop-augmented HGB | 34.41 |
| 8 | ctx_c3_540 | context detector, GRU only in sequence part | 33.09 |
| 10 | ctxE_c2hs3p_540 | wider crop HGB, seed averaging | 33.13 |
| **11** | **mixI_540** | **½ ctx_c3 + ½ ctxE_c2hs3p** | **32.92** |
| 12 | mixK_540 | mixI + extra seeds | 32.99 |

All 12 submissions are listed in [`docs/submissions_log.md`](docs/submissions_log.md), with SHA-256 hashes, local
Docker runs and timings.

**What did not help:**

- an HMC-trained sleep stager (it did not transfer to Muse);
- decision-layer tweaks (temperature, per-range mixing weights);
- a countdown/Bayesian temporal filter;
- an end-to-end raw-EEG CNN→GRU model, which scored 39.8 on its own but added nothing to the ensemble;
- re-training with wider crops, which helped locally (−0.2) but not on the sealed set.

What remains is largely an information limit: telling 1.5–5 min from 5–10 min before onset from frontal/temporal EEG.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Feature extraction, CV protocols, model training, packaging and tests (≈120 scripts). |
| `scripts/build_hgb_submission.py` | Packager. It writes a self-contained `submission.py` (causal features, numpy tree inference, torch models) plus `model.joblib`. |
| `submissions/<name>/submission.py` | Generated submission wrappers. Weights (`*.joblib`) and ZIPs are not in git. |
| `docs/method.md` | **Final method write-up**: model, full production chain, seeds and epochs, reproducibility, compute, external-data declaration. |
| `docs/submissions_log.md` | Every submission: recipe, commands, checks, local and sealed scores. |
| `docs/status.md` | Chronological project log, including all experiments. |
| `*_RAPORU.md` | Detailed experiment reports (decision layer, phase B). |

The detailed documents are written in **Turkish**. Code, comments and commit messages are in English.

## Reproducing the final model

Requirements: Python 3.12 and the packages in `requirements.txt` (torch 2.11 + CUDA, scikit-learn 1.9.1, numpy, scipy,
joblib, neuralbench 0.4.0). Data is the Interaxon2026Muse dataset ([NEMAR nm000287](https://github.com/nemarDatasets/nm000287),
CC-BY-NC-SA-4.0), prepared via `neuralbench eeg _sleep_onset_stream`.

From `scripts/`:

```bash
python extract_features.py && python extract_raw.py && python n2det_extract.py
python train_n2det_final.py detctx                       # context N2 detector (3 seeds)
python train_n2det_final.py hgbmc --det=ctx              # HGB ctx
python train_gru_n2c.py --det=ctx                        # GRU ctx (10 seeds)
python train_mix3_final.py --all --deterministic --tag=det   # TCN, CNN encoder, GRU-emb
python train_n2det_final.py mix --alpha=0.5 --ens=ens_c_all_ctx_crop --seq=mix3det_all --gru-from=gru_ctx_all --out=ctx_c2_all
python train_n2det_final.py hgbmc --det=ctx --cropE --seeds=0,1,2   # HGB E
python train_n2det_final.py mix --alpha=0.4 --ens=ens_c_all_ctxE_s3_crop --seq=mix3det_all --gru-from=gru_ctx_all --out=ctxE_c2hs3_all
python derive_bundles.py                                 # ctx_c3, ctxE_c2hs3p, mixI bundles
python build_hgb_submission.py --model ~/neuralbench/save/mixI_all/model.joblib --name mixI_540 --smooth-a 0
```

Final training takes about 3 hours on a single RTX 3050 laptop GPU plus CPU. All neural components are trained
deterministically, and same-seed retraining was verified to be bit-identical. Epoch and tree counts, and the CV runs
that selected them, are documented in [`docs/method.md`](docs/method.md).

## Acknowledgements

Data: Interaxon2026Muse (Interaxon / NEMAR nm000287, CC-BY-NC-SA-4.0). Benchmark and starter kit:
[NeuralBench](https://facebookresearch.github.io/neuroai/neuralbench/) and the challenge organisers.
