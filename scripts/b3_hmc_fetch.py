"""B3: fetch HMC Sleep Staging (PhysioNet hmc-sleep-staging 1.1, CC BY 4.0) recording by recording and keep only
the sleep-onset part: lights off -> first N2 epoch + 10 min; channels F4-M1, C3-M2, C4-M1, E1-M2, E2-M2;
resampled to 128 Hz, microvolts, float32; plus the 30 s hypnogram labels of that span. At most one raw
recording is on disk at any time; every downloaded file is checked against PhysioNet's SHA256SUMS.txt and
the raw EDFs are deleted right after cropping.

Selection: N recordings drawn with numpy default_rng(SEED) from the recordings listed in SHA256SUMS.txt.
Output: ~/neuralbench/data/hmc/crops/SNxxx.npz, ~/neuralbench/data/hmc/fetch_log.json

Usage: nice -n 19 python scripts/b3_hmc_fetch.py [N] [SEED]   (default 30, 0)
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import mne  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

BASE = "https://physionet.org/files/hmc-sleep-staging/1.1/"
ROOT = Path.home() / "neuralbench" / "data" / "hmc"
CROPS = ROOT / "crops"
TMP = ROOT / "tmp"
CHANNELS = ["EEG F4-M1", "EEG C3-M2", "EEG C4-M1", "EOG E1-M2", "EOG E2-M2"]
SFREQ = 128.0
AFTER_N2_S = 600.0
STAGES = {"Sleep stage W": "W", "Sleep stage N1": "N1", "Sleep stage N2": "N2", "Sleep stage N3": "N3",
          "Sleep stage R": "R"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rel, sums, tries=6):
    """Download one file to TMP (retries with backoff on server / connection errors), check its SHA-256."""
    dst = TMP / Path(rel).name
    t0 = time.perf_counter()
    for attempt in range(1, tries + 1):
        try:
            with requests.get(BASE + rel, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if sha256(dst) == sums[rel]:
                break
            err = "checksum mismatch"
        except (requests.RequestException, OSError) as e:
            err = repr(e)
        dst.unlink(missing_ok=True)
        if attempt == tries:
            raise RuntimeError(f"{rel}: giving up after {tries} attempts ({err})")
        print(f"  {rel}: attempt {attempt} failed ({err}); retrying in {30 * attempt} s", flush=True)
        time.sleep(30 * attempt)
    return dst, time.perf_counter() - t0, dst.stat().st_size


def crop_one(sn, sums):
    rec = {"recording": sn}
    sig, dt_sig, size_sig = fetch(f"recordings/{sn}.edf", sums)
    ann_path, dt_ann, _ = fetch(f"recordings/{sn}_sleepscoring.edf", sums)
    rec.update(download_s=round(dt_sig + dt_ann, 1), raw_mb=round(size_sig / 1e6, 1))
    try:
        t0 = time.perf_counter()
        ann = mne.read_annotations(ann_path)
        desc = [str(d) for d in ann.description]
        lo = [o for o, d in zip(ann.onset, desc) if "lights off" in d.lower()]
        lights_off = float(lo[0]) if lo else 0.0
        stage_ev = [(float(o), float(du), STAGES[d]) for o, du, d in zip(ann.onset, ann.duration, desc) if d in STAGES]
        n2 = [o for o, _, s in stage_ev if s == "N2" and o >= lights_off]
        if not n2:
            rec["skipped"] = "no N2 after lights off"
            return rec
        raw = mne.io.read_raw_edf(sig, include=CHANNELS, preload=False, verbose="error")
        missing = [c for c in CHANNELS if c not in raw.ch_names]
        if missing:
            rec["skipped"] = f"missing channels {missing}"
            return rec
        sf0 = float(raw.info["sfreq"])
        t_end = min(n2[0] + AFTER_N2_S, raw.times[-1])
        raw.crop(tmin=lights_off, tmax=t_end).load_data(verbose="error")
        raw.pick(CHANNELS).resample(SFREQ, verbose="error")
        data = (raw.get_data() * 1e6).astype(np.float32)  # V -> uV
        ep = [(o - lights_off, du, s) for o, du, s in stage_ev if lights_off - 1e-6 <= o < t_end]
        out = CROPS / f"{sn}.npz"
        np.savez(out, data=data, ch_names=np.array(CHANNELS), sfreq=SFREQ, orig_sfreq=sf0,
                 lights_off_s=lights_off, lights_off_marker=bool(lo), n2_onset_s=n2[0] - lights_off,
                 stage_onset=np.array([e[0] for e in ep]), stage_dur=np.array([e[1] for e in ep]),
                 stage=np.array([e[2] for e in ep]), annotations=np.array(sorted(set(desc))))
        rec.update(process_s=round(time.perf_counter() - t0, 1), orig_sfreq=sf0, lights_off_marker=bool(lo),
                   lights_off_s=lights_off, n2_onset_after_lights_off_s=round(n2[0] - lights_off, 1),
                   kept_s=round(data.shape[1] / SFREQ, 1), saved_mb=round(out.stat().st_size / 1e6, 2),
                   median_abs_uv=[round(float(v), 1) for v in np.median(np.abs(data), axis=1)])
        return rec
    finally:
        sig.unlink(missing_ok=True)
        ann_path.unlink(missing_ok=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    CROPS.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    local = ROOT / "SHA256SUMS.txt"  # PhysioNet checksum list, cached (43 KB)
    if not local.exists() or local.stat().st_size == 0:
        for attempt in range(6):
            r = requests.get(BASE + "SHA256SUMS.txt", timeout=60)
            if r.ok and "recordings/SN001.edf" in r.text:
                local.write_text(r.text)
                break
            time.sleep(30 * (attempt + 1))
        else:
            raise RuntimeError("could not fetch SHA256SUMS.txt")
    lines = [ln.split() for ln in local.read_text().splitlines() if ln.strip()]
    assert all(len(x) == 2 and len(x[0]) == 64 for x in lines), "unexpected SHA256SUMS.txt format"
    sums = {x[1]: x[0] for x in lines}
    all_recs = sorted(Path(k).stem for k in sums if k.startswith("recordings/SN") and k.endswith(".edf")
                      and "_sleepscoring" not in k)
    chosen = sorted(np.random.default_rng(seed).choice(all_recs, size=n, replace=False).tolist())
    print(f"{len(all_recs)} recordings listed; chosen (seed {seed}): {chosen}", flush=True)
    log_path = ROOT / "fetch_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    log["selection"] = dict(seed=seed, n=n, n_listed=len(all_recs), chosen=chosen, source=BASE, license="CC BY 4.0")
    t_all = time.perf_counter()
    for sn in chosen:
        if (CROPS / f"{sn}.npz").exists() and sn in log.get("records", {}):
            continue
        rec = crop_one(sn, sums)
        log.setdefault("records", {})[sn] = rec
        log_path.write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)
    print(f"total {time.perf_counter() - t_all:.0f} s; tmp dir now: {sorted(p.name for p in TMP.iterdir())}", flush=True)
