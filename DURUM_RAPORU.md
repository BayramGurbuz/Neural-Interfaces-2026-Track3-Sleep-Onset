# DURUM RAPORU — Track 03 (Sleep Onset, Muse)

Rapor tarihi: 2026-11-11. Depo HEAD: `15b7ddc`. Bu rapor salt okunur bir denetimle hazırlandı; mevcut hiçbir dosya değiştirilmedi, eğitim başlatılmadı, Codabench'e bir şey gönderilmedi.

## Özet

- **En iyi model:** `mix3_540.zip` = (GRU ×10 + TCN ×10 + CNN-gömmeli GRU ×3) / 3 dizi karışımı, 4 üyeli HGB dağılım topluluğu ile α = 0.7 oranında karıştırılmış; 540 kaydın tamamıyla eğitildi.
- **Yerel skor (çift CV, 500 kayıt):** 39.40 s (unseen 40.73 ± 1.77, seen 38.07 ± 1.50). **Mühürlü skor:** 38.66 s.
- **Yerel–mühürlü farkı:** −0.74 s (mühürlü daha iyi). Önceki gönderim `final540`'ta fark −0.72 s idi (40.28 → 39.56); iki gönderim arasındaki iyileşme yerelde 0.88 s, mühürlüde 0.90 s.
- **İlk 3 eşiğine uzaklık:** 38.66 − 33.06 = 5.60 s. 11 Kasım'da denenen ~25 varyantın en iyisi yerelde 0.88 s kazandırdı.
- **Eksik 1 — [300,600] aralığı:** model bu aralıkta sistematik düşük tahmin ediyor (havuzlanmış yanlılık unseen −283 s, seen −254 s); aralığın W-bMAE'ye katkısı (~10.4 s) hiçbir deneyle değişmedi.
- **Eksik 2 — kullanılmayan bilgi/teknikler:** 30 sn ızgarası kullanılmıyor; tahminler arası yumuşatma kodda var ama kapalı; gönderilen modellerde kayıp metrik ağırlıklı değil (ağırlık yalnız karar aşamasında); kanal dropout / sol-sağ değişimi / kayıt başı kırpma yok.
- **Eksik 3 — sağlamlık ve süreç:** `mix3_540`'ta NaN içeren tek bir pencere kaydın kalan tüm tahminlerini 0'a kilitliyor (çalıştırılarak doğrulandı; açık veride NaN pencere yok). Yerel Docker testleri `v3` imajıyla yapıldı, `v4-sealed` ile yapılmadı. mix3'ün mühürlü sonucu depo dokümanlarına işlenmemiş.

**Kanıt işaretleri:** **[Ç]** = bu denetimde çalıştırılıp doğrulandı · **[K]** = koddan okundu · **[L]** = mevcut log / sonuç dosyasından okundu · **[D]** = proje dokümanındaki beyan, bağımsız doğrulanmadı · **[B]** = görev tanımında verilen bağlam bilgisi.

Doğrulama betikleri proje dışında, geçici klasörde çalıştırıldı:
`/tmp/claude-1000/-home-bayram-projects-muse-sleep-onset/79618ccf-b42c-4536-8b59-7e58206c819a/scratchpad/verify1.py` (veri, CV yapısı, metrik, hata analizi), `.../verify2.py` (ZIP içeriği, parametre sayıları, reset/deepcopy/nedensellik, süre), `.../verify3.py` (NaN / düz pencere davranışı). Hepsi `PYTHONDONTWRITEBYTECODE=1` ile, yalnız okuma yaparak koştu.

---

## 1. Proje haritası

### Klasör yapısı [Ç: `ls`, `git ls-files`]

| Yol | İçerik | Git'te mi |
|---|---|---|
| `CLAUDE.md` | proje kuralları, görev, kabul kuralı | evet |
| `docs/` | `status.md`, `submissions_log.md`, `final_plan.md`, `gru_design.md`, kural PDF'leri, `worker_packages.txt` | evet |
| `scripts/` | 43 Python betiği + 1 kabuk betiği (aşağıda) | evet |
| `submissions/<ad>/` | üretilmiş `submission.py` + `model.joblib`; yanında `<ad>.zip` | `submission.py` evet (final540, mix3_540, mix3_tv hariç: izlenmiyor); `.joblib` ve `.zip` `.gitignore`'da |
| `requirements.txt` | sabitlenmiş ortam | evet |
| `~/neuralbench/data/` | ham veri (1.2 GB) | depo dışı |
| `~/neuralbench/save/` | öznitelik önbellekleri, CV tahminleri (`cv/dual/`), eğitilmiş modeller (2.3 GB) | depo dışı |
| `~/neuralbench/compet/` | yerel Docker tekrar-oynatma alanı | depo dışı |
| `~/projects/2026-competition` | yarışma deposu klonu (HEAD `8734b2f`, 2026-11-10) | ayrı depo |

### Önemli betikler [K]

| Betik | İşlevi |
|---|---|
| `stream_utils.py` | pencere tablosu (NeuralBench'ten), yerel W-bMAE (`per_recording_scores`, `summarize`), metrik pencere ağırlıkları (`window_weights`) |
| `causal_features.py` | pencere öznitelikleri + nedensel akış öznitelikleri (`CausalFeatureStream`); `submission.py` içine birebir gömülür |
| `extract_features.py`, `extract_raw.py` | 62 pencere özniteliğini ve ham pencereleri bir kez çıkarıp önbelleğe yazar |
| `cv_utils.py`, `dual_cv.py` | kişi bazlı 5 kat (unseen) + gece dışarıda 5 kat (seen), LOFO seçim, kalibrasyon |
| `dist_hgb.py`, `ablation_c.py`, `ensemble_dist.py` | dağılım sınıflandırıcısı (HGB), öznitelik grubu ablasyonları, 4 üyeli topluluk (c) |
| `gru_cv.py`, `seeds_cv.py`, `seq_cv.py` | GRU / LSTM / TCN çift CV, seed sayısı çalışması |
| `cnn_encoder.py`, `gru_cnn.py` | ham pencere CNN kodlayıcısı ve gömmeli GRU |
| `mix_gru_hgb.py`, `mix_seq.py`, `mix_calib.py` | dizi modeli + HGB karışımı, α seçimi, sıcaklık / bölgesel α deneyleri |
| `range_breakdown.py`, `report_bins.py` | aralık bazında döküm, kat eşleştirmeli karşılaştırma |
| `train_ens_c.py`, `train_mix_final.py`, `train_mix3_final.py` | final eğitimleri (`--all` = 540 kayıt) |
| `build_hgb_submission.py`, `tree_numpy.py` | `submission.py` + ZIP üretimi; HGB ağaçlarını numpy dizilerine çevirir |
| `check_stream_mix3.py`, `compare_final.py` | akış ↔ toplu eşdeğerlik; 500 ↔ 540 ve tekrar koşusu karşılaştırması |
| `docker_test_submission.sh` | Codabench ingestion + scoring'i yerelde `v3` imajında oynatır |
| `subject_info.py`, `oracle_subject.py`, `long_features.py`, `warmup_metrics.py`, `bmae_decision.py` | teşhis ve yan deneyler |

### Git geçmişi [Ç: `git log`]

30 commit, 5–11 Kasım 2026. Aşamalar: 5 Kasım ortam, veri, tabanlar, ilk HGB → 8 Kasım kişi bazlı CV, kalibrasyon, v2 → 9 Kasım çift CV, dağılım HGB, öznitelik grupları, topluluk (c), GRU → 10 Kasım GRU+HGB karışımı, seed çalışması, `final540` → 11 Kasım mühürlü sonuç kaydı, sıcaklık/α, LSTM/TCN, uzun bağlam, kişi bilgisi, CNN kodlayıcı, mix3 (`15b7ddc`). `scripts/` HEAD ile aynı (fark yok).

### Ortam ve donanım [Ç]

| Bileşen | Değer |
|---|---|
| Python | 3.12.3 (`.venv`) |
| torch | 2.11.0+cu128 (CUDA 12.8) |
| neuralbench / neuralset | 0.4.0 / 0.4.0 |
| numpy / scipy / scikit-learn / joblib | 2.5.4 / 1.18.1 / 1.9.1 / 1.6.0 |
| mne / mne-bids / braindecode | 1.13.2 / 0.20.0 / 2.0.0 |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4.0 GiB |
| CPU / RAM | Intel i5-11400H, 12 iş parçacığı / 11 GiB (WSL2) |
| Yerel Docker imajları | `lionelyneuro/neural-compet:v3` (7.03 GB); `v4-sealed` yerelde **yok** |

---

## 2. Veri

| Soru | Bulgu | Kanıt |
|---|---|---|
| Nerede | `~/neuralbench/data/Interaxon2026Muse/download/nm000287` (BIDS, BrainVision `.vhdr/.vmrk`), 1.2 GB | [Ç] `ls`, `dataset_description.json` |
| Kaç kayıt / kişi | 540 kayıt, 203 kişi, 80 975 pencere | [Ç] verify1 |
| Bölünme | train 412 kayıt / 170 kişi / 61 303 pencere · val 88 / 33 / 13 803 · test 40 / 40 / 5 869 | [Ç] verify1 |
| Kişi örtüşmesi | test kişilerinin 40/40'ı train+val'da var; val kişilerinin 0/33'ü train'de | [Ç] verify1 |
| ≥ 2 geceli kişi | 113 kişi, 410 kayıt | [Ç] verify1 |
| Nasıl okunuyor | NeuralBench `_sleep_onset_stream` boru hattı (`Experiment(**cfg).data.prepare()`), elle MNE-BIDS okuma yok | [K] `extract_features.py:24-26`, `stream_utils.py:32-33` |
| Ölçek | görev yapılandırması `scale_factor: 1.0e+6  # V -> uV`, filtre/scaler/clamp `null`; ham pencerelerde medyan \|x\| = 5.45 µV, %99.9 = 846 µV, kanal ortalamaları ≈ 0 | [K] neuralbench `tasks/eeg/_sleep_onset_stream/config.yaml`; [L] `extract_raw.log`; [Ç] verify1 (örneklemde medyan 5.50 µV) |
| Pencere biçimi | (80 975, 4, 640) float32; kanal sırası TP9, AF7, AF8, TP10 | [Ç] verify1/2 |
| Hedef | `clip(n2_onset − stop, 0, 600)`; 24 479 pencere hedef = 600, 463 pencere hedef = 0 | [K] `stream_utils.py:45`; [Ç] verify1 |
| N2 ızgarası | 463/540 (%85.7) onset 30 sn'nin katı; aralık 60–1500 sn | [Ç] verify1 (`sessions.tsv`) |
| Akışın sonu | pencere akışında son pencerenin bitişi = N2 onset (463 kayıtta fark tam 0; kalanlarda 2–3.6 sn önce); ilk pencere bitişi her kayıtta 5.0 sn; kayıt başına 12–300 pencere (medyan 138) | [Ç] verify1 |
| NaN / düz pencere | 80 975 pencerede NaN içeren 0, sıfır varyanslı kanal içeren 0 | [Ç] verify3 |
| Dış veri | kullanılmadı; betiklerde başka veri kümesi, `from_pretrained`, HF indirmesi yok | [Ç] `grep` (eşleşme yok) |
| Bütünlük denetimi | `verify_manifest.py` mevcut; çıktısı kayıtlı değil | [K]; sonuç **bulunamadı** |

Not: CLAUDE.md'deki "N2 başlangıcı kayıt bitişinden tam 300 sn önce" ifadesi ham dosya uzunluğuna dairdir; ham dosya uzunlukları bu denetimde ölçülmedi. Modele gelen pencere akışı N2 onset'te bitiyor (yukarıdaki satır).

---

## 3. İki gönderimin tam içeriği

### 3.1 Ortak yapı [K, Ç]

Her iki ZIP iki dosya içerir: `submission.py` ve `model.joblib`; açılmış klasörlerle SHA-256 özeti birebir aynı [Ç].

| | final540.zip | mix3_540.zip |
|---|---|---|
| Boyut (zip) | 9.02 MB | 13.23 MB |
| SHA-256 (zip) | `24dba3f7…b917b` | `16fdba0d…66377` |
| `submission.py` | 21 261 bayt; `mix_gru10_hgb/submission.py` ile aynı | 32 174 bayt |
| `kind` | `mix` | `mix3` |
| İçe aktarmalar | joblib, numpy, torch, `benchmark_utils.base_solver`, `scipy.signal.find_peaks/welch` | aynı |

**Girdi öznitelikleri (her iki modelde aynı):**
- Pencere düzeyi, 62 öznitelik: kanal başına 5 log bant gücü (delta 0.5–4, theta 4–8, alpha 8–12, sigma 12–15, beta 15–30 Hz; Welch, 2 sn segment), log theta/alpha, log (delta+theta)/(alpha+beta), log varyans, log çizgi uzunluğu, log maks \|x\| (40) + "spec": alfa/beta tepe frekansı, SEF95, aperiodik eğim, spektral entropi (20) + "eye": AF7−AF8 0.2–0.6 Hz log gücü, kırpma sayısı (2). [K] `causal_features.py:33-98`
- HGB girdisi, 817 öznitelik: 62 × (anlık + 30 sn / 2 dk / 5 dk EMA) = 248, bunların kaydın ilk 2 dakikasının ortalamasından farkı (248), geçen süre (1), "dyn" (28 çekirdek × 30 sn/1/2/5 dk eğim, 5 dk varyans, lag-1 otokorelasyon, 10 dk EMA = 196), "maxmin" (30 sn EMA − o ana kadarki maks/min = 124). [K] `causal_features.py:123-140, 234-258`; [Ç] bundle `feature_names` uzunluğu 817
- Dizi modeli girdisi, 125: 62 pencere özniteliği + ilk-2-dk ortalamasından fark (62) + `log1p(min(geçen süre, 1500))`. Eğitim kümesi ortalama/std ile standardize. [K] `gru_cv.py:87-97, 179`

**HGB(c) kısmı (her iki modelde aynı tarif, aynı dosyadan: `~/neuralbench/save/ens_c_all/`):** 4 `HistGradientBoostingClassifier`, `random_state=0`, `max_features=0.5`, ağırlıksız. [K] `train_ens_c.py:56-62`, `ensemble_dist.py:32-39`; [Ç] bundle

| Üye | Sınıf | lr / yaprak / min yaprak örneği / L2 | İterasyon | Ağaç sayısı |
|---|---|---|---|---|
| B | 11 (kenarlar 0,10,20,30,40,60,90,150,225,300,450,600) | 0.05 / 15 / 500 / 5 | 225 | 2 475 |
| B_h1 | 11 | 0.03 / 31 / 300 / 2 | 200 | 2 200 |
| B_h2 | 11 | 0.05 / 7 / 1000 / 10 | 550 | 6 050 |
| B2 | 21 (30 sn dilimler + 600'de nokta sınıf) | 0.05 / 15 / 500 / 5 | 200 | 4 200 |

İterasyon sayıları her üyenin 10 çift-CV LOFO seçiminin medyanı (25'e yuvarlanmış) [K] `train_ens_c.py:57-58`. Üye olasılıkları 25 hücreli ortak ızgaraya yayılıp ortalanır.

**Karar kuralı (her iki modelde):** her bileşenin sınıf olasılıkları, metriğin etkin pencere ağırlığıyla (ω) çarpılıp normalize edilir; karışım dağılımının sürekli medyanı tahmindir. [K] `dist_hgb.py:39-60`, `mix3_540/submission.py:337-346, 542-545, 649-654`. Normalize ω_B (bundle): `[1, 1, 1, 1, 0.40, 0.40, 0.06, 0.059, 0.057, 0.013, 0.006]` [Ç].

### 3.2 final540.zip (gönderim #1, 10 Kasım, mühürlü 39.56)

| Başlık | İçerik | Kanıt |
|---|---|---|
| Model | GRU (10 seed ortalaması) + HGB(c); karışım `α·q_GRU + (1−α)·W_HGB`, α = 0.6 | [Ç] bundle `mix_alpha=0.6`; [K] `final540/submission.py:449` |
| GRU mimarisi | Linear(125→64) + GELU + LayerNorm → GRU(64, 2 katman, tek yönlü) → Linear(64→11) | [K] `gru_cv.py:100-109` |
| Parametre | GRU: 58 827 × 10 seed = 588 270. HGB: 14 925 ağaç (parametrik değil) | [Ç] verify2 |
| Eğitim verisi | 540 kayıt (train+val+test), 80 975 pencere; açık test 40 kaydı **dahil** | [L] `mix_uw_n10_all/results.json` (`all540: true`, `n_train_windows: 80975`); [K] `train_mix_final.py:41` |
| Seed | GRU 0–9; HGB `random_state=0` | [L] `results.json`; [K] |
| Kayıp | GRU: ağırlıksız çapraz entropi, 11 sınıf (`weighted=False`); HGB: ağırlıksız çok sınıflı log-kayıp | [K] `gru_cv.py:52, 171, 206`; `train_ens_c.py:62` |
| Pencere ağırlığı | kayıpta yok; yalnız karar aşamasında ω | [K] |
| Optimizasyon | AdamW lr 2e-3, wd 1e-2, kosinüs (T_max 150), 15. epoch'ta durdurma, batch 32 kayıt, gradyan kırpma 1.0, dropout 0.2 | [K] `gru_cv.py:41-49, 185-186, 209`; [L] `n_epochs: 15` |
| Artırma | kayıt başına kanal kazancı (log10 σ = 0.1) + öznitelik gürültüsü (σ = 0.05) | [K] `gru_cv.py:46-47, 199-201` |
| α seçimi | protokol başına LOFO; 10 seçimin 9'u 0.6, 1'i 0.5 → 0.6 | [L] `mix_gru_hgb.json` (`GRU-uw_n10+HGB(c)`) |
| Epoch seçimi | LOFO, seçimler 10–15 → 15 | [L] `gru.json` (`GRU-uw_n10`) |
| Kalibrasyon / yumuşatma | kapalı (`CALIB_X = None`, `SMOOTH_A = 0.0`) | [K] `final540/submission.py:286, 289` |
| Üretim | `train_ens_c.py --all` → `train_mix_final.py uw 15 0.6 0,1,…,9 _n10 --all` → `build_hgb_submission.py --model …/mix_uw_n10_all/model.joblib --name final540 --smooth-a 0` | [D] `submissions_log.md:129-131` |
| Commit | `6695885` ("Train final 540-recording mix (final540)…") | [Ç] `git log` |
| Tekrar üretilebilirlik | aynı seed'lerle ikinci 540 koşusu bit düzeyinde aynı: GRU ağırlıkları, binleme sınırları, µ/σ, olasılıklar, tahminler için maks fark 0.0 | [L] `mix_uw_n10_all/compare.json` → `repeat` |

### 3.3 mix3_540.zip (gönderim #2, 11 Kasım, mühürlü 38.66)

**"mix3" tam olarak:** üç dizi bileşeninin her biri kendi seed ortalamasını alır, ω_B ile ağırlıklanıp normalize edilir, ortak ızgaraya yayılır; üçü **eşit ağırlıkla (1/3)** ortalanır (`q3`); sonra `0.7·q3 + 0.3·W_HGB`. [K] `train_mix3_final.py:42, 124-126, 145-148`; `mix3_540/submission.py:583-584, 653`

| Bileşen | Mimari | Parametre (tek model) | Seed | Epoch | Girdi |
|---|---|---|---|---|---|
| GRU-uw | final540'taki GRU ile aynı | 58 827 | 10 (0–9) | 15 | 125 |
| TCN | Linear(125→64)+GELU+LN → 6 nedensel dilate Conv1d (çekirdek 3, dilasyon 1–32, alıcı alan 127 pencere ≈ 10.6 dk), artık bağlantı + LN → Linear(64→11) | 83 787 | 10 (0–9) | 10 | 125 |
| CNN kodlayıcı | ham (4, 640) pencere; 4 × Conv1d(çekirdek 7, adım 2) + 2 dilate blok → ort+maks havuz → Linear(64→32) → 11 sınıf; girdi `clamp(x/20, ±15)` | 35 323 | 3 (0–2) | 12 | ham pencere |
| GRU-emb | GRU-uw + 32-b gömme + gömmenin ilk-2-dk farkı | 62 923 | 3 (0–2), seed s ↔ kodlayıcı s | 10 | 189 |
| HGB(c) | final540 ile aynı dosya (`ens_c_all`) | 14 925 ağaç | `random_state=0` | – | 817 |

Toplam sinir ağı parametresi: 588 270 + 837 870 + 105 969 + 188 769 = **1 720 878** [Ç verify2].

| Başlık | İçerik | Kanıt |
|---|---|---|
| Eğitim verisi | 540 kayıt, 80 975 pencere; açık test 40 kaydı **dahil** | [L] `mix3_all.log` ("all 540: 80975 windows, 540 recordings"), `mix3_all/results.json` |
| Kayıp | tüm sinir bileşenleri ağırlıksız çapraz entropi (11 sınıf); pencere ağırlığı yalnız kararda | [K] `gru_cv.py:52, 65`; `cnn_encoder.py:98` |
| Artırma | GRU/TCN/GRU-emb: yukarıdaki kazanç + gürültü; CNN: pencere ve kanal başına kazanç (log10 σ 0.1) + gürültü (0.1 ölçekli birim ≈ 2 µV) | [K] `cnn_encoder.py:40, 106-107` |
| α seçimi | iç içe LOFO: dışarıdaki katın kayıtları maskelenip diğer 9 katın (iki protokol) toplam kaybı en aza indirilir; 10 katın 10'unda 0.7 | [K] `mix_seq.py:57-59`, `mix_calib.py:66-87`; [L] `mix3_gru_tcn_cnnemb.log` ("alphas [0.7 ×10]") |
| Dizi ağırlıkları | seçilmedi; sabit 1/3. Serbest GRU/TCN/HGB ağırlık ızgarası (GRU-emb'siz) ayrıca denendi: 40.06 | [K]; [D] `status.md:117-120` |
| Epoch'lar | CV'den sabit: GRU 15, TCN 10, kodlayıcı 12 (LOFO 9–15), GRU-emb 10 | [K] `train_mix3_final.py:43-44`; [L] `cnn_cnn3.json`, `gru.json` |
| Gömme koşulu | GRU-emb eğitimde örneklem-içi gömmeler görür; çıkarımda gömmeler örneklem-dışıdır. CV'de aynı koşul geçerli (eğitim kayıtları içi, dışarıdaki kayıtlar dışı) | [K] `train_mix3_final.py:79-90`, `gru_cnn.py:1-3` |
| Üretim | `train_mix3_final.py --all` → `build_hgb_submission.py --model …/mix3_all/model.joblib --name mix3_540` | [D] `submissions_log.md:31-34`; [L] `mix3_540_build.log` |
| Commit | `15b7ddc` | [Ç] `git log` |
| Tekrar üretilebilirlik | Seed'ler sabit (`torch.manual_seed`, numpy `default_rng`, GPU üreteci). mix3 için ağırlık düzeyinde tekrar koşusu karşılaştırması **bulunamadı**. Dolaylı kanıt: GRU-uw bileşeni final540 koşusuyla aynı örneklem-içi test skorunu veriyor (39.8588, 15+ hane aynı); Docker'da aynı ZIP iki koşuda aynı skor (çıkarım determinizmi). TCN ve CNN'in (cuDNN evrişim) yeniden eğitimde bit düzeyinde aynı çıkıp çıkmadığı test edilmedi | [K] `gru_cv.py:165-166`, `cnn_encoder.py:89-90`; [L] `mix3_all/results.json` ↔ `mix_uw_n10_all/results.json`; `mix3_540_build.log` |

### 3.4 Yerel ve mühürlü skorlar (saniye)

| | Çift CV makro | unseen (± kat SS) | seen (± kat SS) | Açık test, 500-kayıt modeli | Açık test, 540-modeli (örneklem içi) | Mühürlü | Mühürlü − çift CV |
|---|---|---|---|---|---|---|---|
| final540 | 40.28 | 41.80 ± 1.91 | 38.76 ± 1.41 | 46.91 | 31.53 | 39.56 | −0.72 |
| mix3_540 | 39.40 | 40.73 ± 1.77 | 38.07 ± 1.50 | 47.06 | 32.74 | 38.66 | −0.74 |
| fark (mix3 − final540) | −0.88 | −1.07 | −0.69 | +0.15 | – | −0.90 | – |

Kaynaklar: çift CV [Ç] verify1 (kayıtlı OOF tahminleri yeniden skorlandı); açık test [L] `mix_uw_n10/results.json`, `mix3/results.json`, `mix3_all/results.json`, `mix_uw_n10_all/results.json`; mühürlü final540 [D] `submissions_log.md:43` (W-bMAE makro 39.56, W-bMAE 39.91, bMAE 101.36, MAE 198.26, süre 1249.51 sn, 23/60); mühürlü mix3 [B] — depo dokümanlarında kayıtlı değil (`status.md:50` ve `submissions_log.md:24` hâlâ "onay bekliyor" yazıyor).

Çift CV 500 kayıtla (train+val) ölçülmüştür; gönderilen modeller 540 kayıtla eğitildiği için birebir aynı model için fold-dışı skor yoktur.

---

## 4. Doğrulama şeması

| Soru | Bulgu | Kanıt |
|---|---|---|
| Katılımcı bazlı gruplama | var: `GroupKFold(n_splits=5, shuffle=True, random_state=0)`, grup = subject | [K] `cv_utils.py:10-16` |
| Unseen rejimi | 5 kat; dışarıdaki kayıt 107/110/99/98/86, kişi 41/41/41/40/40; dışarıdaki kişilerin eğitimde kaydı 0 | [Ç] verify1 |
| Seen rejimi | 5 kat; her katta ≥ 2 geceli 113 kişinin birer gecesi dışarıda (113 kayıt), diğer geceleri eğitimde; dışarıdaki kayıt eğitimde 0. 5 katta toplam 565 kat-kaydı, 410 farklı kayıt (katlar kayıt paylaşıyor) | [K] `dual_cv.py:41-51`; [Ç] verify1 |
| Ayrı ölçüm + makro | var: `dual = (unseen ortalaması + seen ortalaması) / 2` | [K] `dual_cv.py:128` |
| Kapsam | yalnız train+val (500 kayıt); test 40 kaydı CV dışında | [K] `dual_cv.py:54-57` |
| Kat / tohum sayısı | 5 + 5 kat; CV bölünmesi tek tohumla (0), tekrarlı CV yok. Model tohumları: GRU/TCN 10, CNN/GRU-emb 3, HGB 1 | [K] |
| Katlar arası SS (ddof = 1) | mix3: unseen 1.77, seen 1.50 · final540 tarifi: 1.91, 1.41 | [Ç] verify1 |
| Seed gürültüsü | 3 seed'li karışımda farklı seed üçlüleri, açık test: 48.22 / 48.70 / 47.90 / 47.58 → σ ≈ 0.48 | [D] `submissions_log.md:143-146` |
| Hiperparametre seçimi | epoch / ağaç sayısı: LOFO (kat k için diğer katların ortalama eğrisi). α: final540'ta protokol içi LOFO, mix3'te iç içe LOFO (kayıt maskeli). Epoch LOFO'sunda seen katları arasında kayıt maskelemesi yok | [K] `dual_cv.py:60-69`, `gru_cv.py:243`, `mix_calib.py:74-83` |
| Kat içi istatistikler | binleme sınırları, µ/σ, ω yalnız katın eğitim satırlarından | [K] `dist_hgb.py:76-77`, `gru_cv.py:179`, `mix_seq.py:37-39` |

### Yerel metrik ↔ resmi metrik

Resmi kod: `~/projects/2026-competition/benchmark_utils/metrics.py:10-42` (`binned_mae`) ve `tracks/sleep_onset/objective.py:93-103` (`group_scores` ile kayıt başına, sonra ortalama). Yerel kod: `scripts/stream_utils.py:56-77`.

| Özellik | Resmi | Yerel | Sonuç |
|---|---|---|---|
| Bin sınırları | `np.digitize(y, edges[1:-1], right=False)`; son bin 600'ü içerir | `searchsorted(EDGES, y, side="right") − 1`, 0–3'e kırpılı | hedefler 0, 39.999, 40, 89.999, 90, 299.999, 300, 600 için ikisi de `[0,0,1,1,2,2,3,3]` [Ç] |
| Boş bin | boş olmayan binler üzerinden `Σ w·MAE / Σ w` | `notna` maskesiyle aynı formül | oyuncak örnek: ikisi de 79.487179 [Ç] |
| Kayıt ortalaması | kayıt başına skor, eşit ağırlıklı ortalama | `groupby(timeline)`, ortalama | aynı [K] |
| Sayısal fark | 4 model × 10 katın fold-dışı tahminleri iki kodla skorlandı | | **maks \|fark\| = 1.4 × 10⁻¹⁴ sn** [Ç] verify1 |

Ek çapraz kontroller: yerel akış skoru 32.7446 ↔ Docker (resmi ingestion + scoring) 32.74462 [L] `mix3_540_build.log`; final540 yerel 31.53379 ↔ Docker 31.53373; EEGNet 3 seed için yerel hesap ↔ NeuralBench log değeri 6. haneye kadar aynı [L] `warmup_metrics.json` → `checks`.

Seen/unseen makro ortalamasının resmi mühürlü kodu yarışma deposunda yok ("ships with the sealed evaluation data", `objective.py:20-21`); bu kısım birebir karşılaştırılamadı.

---

## 5. Deney tablosu

Tüm skorlar W-bMAE, saniye, düşük iyi. "Açık 40" = test 40 kayıt, 500-kayıt modeliyle. Çift CV sütunları kalibrasyonsuz ("plain") değerlerdir; kalibrasyonlu değer ayrıca belirtilmiştir. Kaynaklar: [L] `~/neuralbench/save/cv/dual/*.json` ve `*.log` (verify2 ile okundu), [D] `docs/status.md`, `docs/submissions_log.md`.

### Tabanlar ve HGB hattı

| Yaklaşım | unseen | seen | makro | Açık 40 | Mühürlü | Sonuç |
|---|---|---|---|---|---|---|
| Sabit 35 sn | – | – | – | 60.74 | – | taban |
| Geçen süre kuralı | – | – | – | 61.25 | – | taban |
| EEGNet (NeuralBench, 3 seed) | – | – | – | 60.03 | – | taban; işe yaramadı |
| HGB v1 regresyon (L1, metrik ağırlıklı, 321 özn.) | 48.88 | 46.68 | 47.78 | 54.83 | – | başlangıç |
| HGB c5 regresyon | 48.13 | 46.39 | 47.26 | – | – | küçük kazanç |
| c5 + izotonik kalibrasyon (`hgb_v2_s0`) | 47.27 | 45.57 | 46.42 | 54.94 | – | küçük kazanç |
| Dağılım sınıflandırıcı B, ayrık karar | 45.94 | 43.56 | 44.75 | – | – | işe yaradı |
| Dağılım B, sürekli medyan | 45.25 | 42.68 | 43.97 | – | – | işe yaradı (en büyük sıçrama) |
| Dağılım B2 (30 sn dilimler), sürekli | 45.44 | 43.01 | 44.22 | – | – | B ile benzer |
| B + dyn | 45.17 | 42.15 | 43.66 (kalib. 43.30) | 52.48 (`hgb_v3`) | – | işe yaradı |
| B + spec | 45.27 | 42.47 | 43.87 | – | – | küçük |
| B + maxmin | 44.91 | 42.54 | 43.73 | – | – | küçük |
| B + maxmin (rel yerine) | 45.39 | 42.85 | 44.12 | – | – | işe yaramadı |
| B + eye | 45.07 | 42.69 | 43.88 | – | – | küçük |
| B + dyn + maxmin | 45.45 | 42.18 | 43.81 | – | – | – |
| B + dyn + maxmin + spec | 44.69 | 41.46 | 43.08 | – | – | işe yaradı |
| B + dyn + maxmin + spec + eye (817) | 44.94 | 41.47 | 43.20 (kalib. 42.92) | – | – | seçilen öznitelik kümesi |
| Topluluk (a) B + B2 | 44.36 | 41.08 | 42.72 | – | – | işe yaradı |
| Topluluk (b) B + B_h1 + B_h2 | 44.50 | 41.14 | 42.82 | – | – | işe yaradı |
| Topluluk (c) 4 üye (`hgb_ens_c`) | 44.28 | 41.02 | 42.65 (kalib. 42.50) | 52.18 (kalib. 52.86) | – | işe yaradı; karışımların HGB kısmı |
| B + "long" grubu (10/20 dk, 929 özn.) | 44.59 | 41.66 | 43.12 | – | – | işe yaramadı (Δ −0.08, 5/10 kat) |
| Çıktı yumuşatma, yakın-onset analizi | – | – | **bulunamadı** | – | – | eklendi (`5527be1`, `003fed7`), sonuç dosyası bulunamadı; gönderimlerde kapalı |

### Dizi modelleri (tek başına)

| Yaklaşım | unseen | seen | makro | Açık 40 | Sonuç |
|---|---|---|---|---|---|
| GRU-w (metrik ağırlıklı CE, 3 seed) | 43.59 | 41.29 | 42.44 | – | ağırlıksızdan kötü |
| GRU-uw (ağırlıksız CE, 3 seed) | 42.77 | 40.56 | 41.66 | 48.46 | işe yaradı |
| GRU u64_d2 / u64_d4 / u32_d2 / u32_d4 / w64_d4 | 43.89 / 45.42 / 45.77 / 46.22 / 43.74 | 42.03 / 42.53 / 44.71 / 44.97 / 42.01 | 42.96 / 43.97 / 45.24 / 45.60 / 42.88 | – | işe yaramadı |
| GRU-uw 5 seed | 42.40 | 40.18 | 41.29 | – | – |
| GRU-uw 10 seed | 43.13 | 39.73 | 41.43 | 47.44 | karışım bileşeni |
| LSTM 10 seed | 44.49 | 41.32 | 42.90 | – | işe yaramadı |
| TCN 10 seed (alıcı alan 10.6 dk) | 42.60 | 40.42 | 41.51 | 47.10 | karışım bileşeni |
| TCN-long 10 seed (alıcı alan 42.6 dk) | 43.03 | 40.60 | 41.81 | – | işe yaramadı |
| CNN kodlayıcı tek başına (3 seed) | 50.46 | 50.51 | 50.49 | – | tek başına zayıf |
| CNN split-half tek başına | 53.04 | 51.54 | 52.29 | – | işe yaramadı |
| GRU + CNN gömme (3 seed) | 44.94 | 43.42 | 44.18 | 57.40 | tek başına zayıf; karışım bileşeni |
| GRU + CNN logit (split-half) | 42.91 | 42.23 | 42.57 | – | işe yaramadı |

### Karışımlar (dizi + HGB(c))

| Yaklaşım | unseen | seen | makro | Δ (ref 40.28) | İyi kat | Açık 40 | Mühürlü | Sonuç |
|---|---|---|---|---|---|---|---|---|
| GRU(3) + HGB(c), α 0.5 (`mix_gru_hgb`) | 41.81 | 39.06 | 40.43 | +0.15 | – | 48.22 | – | yedek |
| GRU(5) + HGB(c) | 41.62 | 38.96 | 40.29 | +0.01 | – | – | – | – |
| **GRU(10) + HGB(c), α 0.6 (`final540` tarifi)** | 41.80 | 38.76 | 40.28 | ref | – | 46.91 | **39.56** | gönderildi (#1) |
| Sabit α 0.6 | – | – | 40.25 | −0.03 | 1/10 | – | – | fark yok |
| Sıcaklık τ global / aralığa göre | – | – | 40.24 / 40.26 | −0.03 / −0.02 | 5/10 / 7/10 | – | – | işe yaramadı |
| α aralığa göre / geçen süreye göre | – | – | 40.22 / 40.40 | −0.06 / +0.12 | 7/10 / 3/10 | – | – | işe yaramadı |
| (α, τ) bölge başına ortak; α bölge + τ global (2 varyant) | – | – | 40.28 / 40.26 / 40.29 | 0.00 / −0.02 / +0.01 | 6/10 / 7/10 / 5/10 | – | – | işe yaramadı |
| Karışım + izotonik kalibrasyon | 41.94 | 39.11 | 40.53 | +0.25 | – | – | – | zarar verdi |
| LSTM(10) + HGB(c) | 42.60 | 39.45 | 41.03 | +0.75 | 1/10 | – | – | işe yaramadı |
| TCN(10) + HGB(c) | 41.79 | 39.10 | 40.44 | +0.16 | 3/10 | – | – | işe yaramadı |
| (GRU + TCN)/2 + HGB(c), α 0.7 | – | – | 40.00 | −0.28 | 8/10 | 46.55 | – | eşiğin 0.02 altında; gönderilmedi |
| GRU / TCN / HGB serbest ağırlık | – | – | 40.06 | −0.22 | 8/10 | – | – | işe yaramadı |
| TCN-long + HGB(c) | 41.72 | 39.32 | 40.52 | +0.24 | 4/10 | – | – | işe yaramadı |
| (GRU + TCN-long)/2 + HGB(c) | 41.38 | 38.81 | 40.09 | −0.18 | 7/10 | – | – | işe yaramadı |
| B_long + GRU | 41.74 | 38.86 | 40.30 | −0.07 (kendi ref. 40.37) | 7/10 | – | – | işe yaramadı |
| GRU-emb + HGB(c), α 0.5 | 41.21 | 38.95 | 40.08 | −0.20 | 7/10 | – | – | kuralı geçmedi |
| GRU-logit (split-half) + HGB(c) | 41.51 | 39.31 | 40.41 | +0.13 | 4/10 | – | – | işe yaramadı |
| **(GRU10 + TCN10 + GRU-emb3)/3 + HGB(c), α 0.7 (`mix3`)** | 40.73 | 38.07 | 39.40 | −0.88 | 10/10 | 47.06 | **38.66** | gönderildi (#2) |

### Teşhis deneyleri

| Deney | Bulgu | Kaynak |
|---|---|---|
| Kişi bilgisi oracle (seen katlar; deneğin diğer gecelerinin medyan süresi öznitelik olarak) | B: 41.47 → 40.87; B + GRU: 38.82 → 38.52 (5/5 kat); kazanç [300,600]'den | [D] `status.md:158-164` |
| Uykuya dalma süresi varyansı | ICC(1) = 0.38; denek içi SS 273 sn, denekler arası 214 sn | [D] `status.md:145-148` |
| Sinyalden denek tanıma | ilk 120 sn top-1 %17.3, tüm kayıt %39.5 (şans %0.6) | [D] `status.md:150-152` |
| bMAE-optimal karar karşılaştırması | `bmae_decision.py` mevcut; sonuç dosyası **bulunamadı** | [K] |
| İğcik / K-kompleks detektörleri, bipolar derivasyonlar (AF8−TP10, AF7−TP9, TP9−TP10) | **denenmedi** (kodda yok; "ertelenenler" listesinde) | [K] `causal_features.py`; [D] `status.md:309` |
| Ham sinyalden uçtan uca CNN → GRU, 2D CNN, 10 seed'li kodlayıcı | **çalıştırılmadı** | [D] `status.md:173-186, 218-219` |

---

## 6. Kontrol listesi

| # | Kontrol | Durum | Kanıt |
|---|---|---|---|
| a | Kayıp metriğe hizalı pencere ağırlığı kullanıyor mu | **kısmen** | Ağırlık formülü mevcut ve doğru: `w = bin ağırlığı / (bin'deki pencere sayısı × kayıttaki mevcut bin ağırlıkları toplamı × kayıt sayısı)` [K] `stream_utils.py:80-93`. İlk HGB regresörlerinde kayıpta kullanıldı (`dual_cv.py:79`). **Gönderilen iki modelde kayıp ağırlıksız**; ağırlık yalnız karar aşamasında ω olarak uygulanıyor (`dist_hgb.py:39-51`). Ağırlıklı CE'li GRU-w denendi: 42.44, ağırlıksız 41.66 |
| b | Geçen süre özellik mi | **var** | HGB: `elapsed = n_seen × step_s` (`causal_features.py:237, 249`). Dizi modelleri: `log1p(min(t·5, 1500))` (`gru_cv.py:93`; `mix3_540/submission.py:562`). v1'de permütasyon önemi +1.05 [L] `hgb_run.log` |
| c | Nedensel hafıza; `reset_state()` doğru mu | **var** | EMA 30 sn / 2 dk / 5 dk / 10 dk, ilk-2-dk tabanı, koşan maks/min, 60 pencerelik geçmiş, GRU gizli durumu, TCN tap geçmişi. `reset_state` hepsini sıfırlıyor (`mix3_540/submission.py:528-540`, `150-159`, `445-446`). **[Ç] verify2, iki ZIP için:** başka kayıttan sonra `reset_state` → taze kopyayla maks fark 0.0; reset'siz → 250–252 sn fark (durum gerçekten var); `deepcopy(kullanılmış model)` + reset → 0.0 |
| d | Kayıt-içi nedensel normalizasyon | **kısmen** | Var: ilk 2 dakikanın koşan ortalamasından fark (`causal_features.py:244-249`, `gru_cv.py:91-94`); 30 sn EMA − o ana kadarki maks/min; pencere başına ortalama çıkarma (`causal_features.py:36`). Yok: kayıt-içi koşan z-skor / ölçek normalizasyonu. Dizi girdileri eğitim kümesinin sabit µ/σ'sı ile, CNN sabit `x/20` ile ölçekleniyor |
| e | Tahmin [0, 600]'e kırpılıyor mu | **var** | `np.clip(np.nan_to_num(pred, nan=0.0), 0.0, 600.0)` — `final540/submission.py:463`, `mix3_540/submission.py:667`. Karar ızgarası zaten 0–600 |
| f | 30 sn ızgarası bilgisi | **yok** | Tahmini bitiş anını 30'un katına çeken kod yok. B2 üyesi kalan süreyi 30 sn'lik dilimlere böler ama "nothing assumes the onsets lie on a 30 s grid" (`dist_hgb.py:26-27`). Izgara bayrağı yalnız çevrimdışı analizde okunuyor (`report_bins.py:5-6, 22-30`) |
| g | Tahminler arası tutarlılık / yumuşatma | **kısmen (kodda var, kapalı)** | `s_t = a·max(s_{t−1} − 5, 0) + (1−a)·p_t` uygulanmış (`cv_utils.py:19-37`; `mix3_540/submission.py:607-615`), ancak iki gönderimde `SMOOTH_A = 0.0` (`final540:286`, `mix3_540:324`). Örtük tutarlılık yalnız GRU/TCN durumu ve EMA'lardan geliyor. Yumuşatmanın çift CV sonucu **bulunamadı** |
| h | Topluluk var mı; ağırlıklar fold-dışı mı | **var** | HGB 4 üye eşit ağırlık; dizi modelleri seed ortalaması; mix3'te 3 dizi bileşeni eşit (1/3, seçilmemiş). α fold-dışı tahminlerle: final540 LOFO (9/10 kat 0.6), mix3 iç içe LOFO kayıt maskeli (10/10 kat 0.7). Açık test seçimde kullanılmadı [K] `mix_gru_hgb.py:62`, `mix_seq.py:57-59` |
| i | Veri artırma | **kısmen** | Var: kayıt başına kanal kazancı + gürültü (öznitelik düzeyi, `gru_cv.py:199-201`); CNN'de pencere/kanal kazancı + gürültü (`cnn_encoder.py:106-107`). Yok: kanal dropout, sol-sağ değişimi, kayıt başı kırpma (`grep flip\|swap\|crop` eşleşmesi yok). HGB'de artırma yok |

---

## 7. Sızıntı ve akış uyumu denetimi

| Kontrol | Bulgu | Kanıt |
|---|---|---|
| Kayıt uzunluğu / örnek sayısı / dosya sonu | kullanılmıyor. Pencere tablosunda kayıt uzunluğu saklanmıyor; `n_times` yalnız pencere içi Welch segment uzunluğu için | [K] `stream_utils.py:24-25`; [Ç] `grep` (`causal_features.py:37-38` dışında eşleşme yok) |
| Tüm-kayıt istatistikleri | yok. Kayıt düzeyi tek istatistik ilk 2 dakikanın koşan ortalaması (nedensel) | [K] |
| `filtfilt` / sıfır fazlı filtre | yok | [Ç] `grep` eşleşmesi yok |
| Çift yönlü katman | yok; `nn.GRU(...)` varsayılan tek yönlü; TCN yalnız sol dolgulu; CNN yalnız pencere içi | [K] `gru_cv.py:104, 137`; `cnn_encoder.py` |
| Geleceğe bakan özellik | yok. **[Ç] verify2:** akışın yalnız ilk yarısı verildiğinde tahminler tam akıştakiyle birebir aynı (maks fark 0.0), iki ZIP için | [Ç] |
| Toplu ↔ akış eşdeğerliği (mix3_540) | test edildi: 40 kayıt, 5 869 pencere; maks \|Δ\| 0.114 sn, ortalama 0.0033 sn, 401 pencerede > 0.01 sn; W-bMAE akış 32.7446 ↔ toplu 32.7449 | [L] `mix3_540_build.log` (`check_stream_mix3.py`) |
| Toplu ↔ akış (mix3_tv, 500-kayıt modeli) | maks \|Δ\| 0.055 sn; 47.0599 ↔ 47.0597 | [D] `status.md:239` |
| Toplu ↔ akış (final540) | ayrı pencere bazlı karşılaştırma **bulunamadı**; skor düzeyinde yerel toplu 31.53379 ↔ Docker akış 31.53373 | [L] `warmup_metrics.json` |
| HGB numpy ağaçları ↔ sklearn | maks \|ΔP\| = 2.6 × 10⁻¹⁵ | [D] `submissions_log.md:70` |
| Açık test 40 kaydı eğitime girdi mi | **evet, bilerek**: her iki gönderim 540 kayıtla eğitildi. Bu yüzden bu modellerin açık test skoru (31.53, 32.74) örneklem içidir | [L] `results.json` (`all540: true`) |
| Açık test model seçimine girdi mi | kodda girmiyor: epoch, ağaç sayısı, α, öznitelik kümesi CV'den (train+val); final betikleri testi bir kez, seçimden sonra skorluyor | [K] `dual_cv.py:54-57`, `train_mix3_final.py:95` ve sonrası |
| Seçimde kullanılmayan ama görülen | 500-kayıt modellerinin açık test skorları dokümanlara yazılmış ve karşılaştırılmış (ör. mix2 46.55 < mix3 47.06 iken mix3 seçildi → seçim CV'ye göre yapılmış) | [D] `status.md:238` |
| Sarmalayıcı sözleşmesi | yalnız `meta["submission_dir"]`, `meta["sfreq"]`, `meta["ch_names"]` okunuyor; dataset adı, yol, subject id okunmuyor | [K] `mix3_540/submission.py:674-678` |

---

## 8. Çıkarım maliyeti

| Ölçüm | final540 | mix3_540 | Kanıt |
|---|---|---|---|
| Pencere başına süre, yerel CPU (i5-11400H, 6 torch iş parçacığı) | 20.6 ms | 31.7 ms | [Ç] verify2 (66 pencerelik bir kayıt) |
| Pencere başına süre, 40 kayıt / 5 869 pencere | ≈ 21 ms | 32.5 ms | [D] `submissions_log.md:38`; [L] `mix3_540_build.log` |
| Yerel Docker `v3`, CPU, açık test toplam | 122 sn | 225.3 sn / 213.1 sn (2 koşu, aynı skor) | [D] `submissions_log.md:23`; [L] `mix3_540_build.log` |
| Mühürlü süre | 1 249.51 sn | **bulunamadı** | [D] `submissions_log.md:43` |
| Mühürlü süre tahmini (gönderim öncesi) | – | ≈ 2 200–2 300 sn (Docker oranı 1.75–1.85 × 1 249.5) | [D] `submissions_log.md:37-38` |
| Sınır | 3 600 sn | 3 600 sn | [K] `competition_sleep.yaml` (`execution_time_limit: 3600`) |

- **CPU/GPU:** çıkarım tamamen CPU'da. Girdi `.cpu().numpy()`'ye çevriliyor, ağlar hiçbir cihaza taşınmıyor; yalnız çıktı tensörü `X.device`'a konuyor [K] `mix3_540/submission.py:619, 668`. A100 kullanılmıyor.
- **Resmi harness:** `tools/run_docker.py` ve `tools/debug_submission.py` ile koşturulduğuna dair kayıt **bulunamadı**. Yerel test, aynı ingestion ve scoring programlarını çağıran proje betiği `scripts/docker_test_submission.sh` ile yapıldı (`--network none`, salt okunur bağlamalar) [K] satır 27-37; [L] `~/neuralbench/compet/run/ingestion.log`, `score/scores.json`.
- **İmaj:** testler `lionelyneuro/neural-compet:v3` ile [K] `docker_test_submission.sh:19`. `v4-sealed` imajı yerelde yok [Ç] `docker images`; dokümana göre Docker Hub'dan çekilemiyor [D] `status.md:301-303`. **`v4-sealed` ile yerel test yapılmadı.** Yarışma deposundaki yaml hâlâ `docker_image: lionelyneuro/neural-compet:v3` yazıyor [Ç].
- Mühürlü kümenin boyutu açıklanmıyor [D]; bu yüzden pencere başına süreden toplam süre bağımsız hesaplanamadı.

---

## 9. Hata analizi

Kaynak: en iyi modelin (mix3) çift CV fold-dışı tahminleri, `~/neuralbench/save/cv/dual/mix3_gru_tcn_cnnemb_{unseen,seen}_preds.npz`, train+val 500 kayıt [Ç] verify1. Bunlar 500 kayıtla eğitilmiş kat modellerinin tahminleridir; gönderilen 540-modeli için fold-dışı tahmin yoktur.

### Bin bazında MAE (saniye)

| Rejim | Ölçü | [0,40) | [40,90) | [90,300) | [300,600] |
|---|---|---|---|---|---|
| unseen | kayıt ortalaması MAE | 15.05 | 29.77 | 87.49 | 226.67 |
| seen | kayıt ortalaması MAE | 13.89 | 28.42 | 83.92 | 205.24 |
| unseen | havuzlanmış MAE | 15.11 | 29.91 | 87.01 | 283.80 |
| seen | havuzlanmış MAE | 13.89 | 28.42 | 83.35 | 256.05 |
| unseen | havuzlanmış yanlılık (tahmin − gerçek) | +11.8 | −7.1 | −58.5 | −283.1 |
| seen | havuzlanmış yanlılık | +10.2 | −9.2 | −53.8 | −254.4 |
| unseen | pencere sayısı | 4 000 | 5 000 | 20 502 | 45 604 |
| seen | pencere sayısı | 4 520 | 5 650 | 23 196 | 51 213 |

[300,600] aralığında mutlak hata ile yanlılık neredeyse eşit: hatanın tamamına yakını düşük tahminden geliyor.

### W-bMAE'ye katkı (makro, saniye) [L] `mix3_gru_tcn_cnnemb.log`

| | [0,40) | [40,90) | [90,300) | [300,600] | Toplam |
|---|---|---|---|---|---|
| unseen | 8.03 | 7.94 | 13.86 | 10.89 | 40.73 |
| seen | 7.36 | 7.54 | 13.33 | 9.84 | 38.07 |
| makro | 7.69 | 7.74 | 13.60 | 10.37 | 39.40 |
| final540 tarifine göre Δ | −0.22 | −0.27 | −0.47 | +0.09 | −0.88 |

### Onset süresine göre

Pencere akışı N2 onset'te bittiği için modele gelen kayıt uzunluğu = onset süresi (1 065 kat-kaydının 914'ünde fark 0, kalanında birkaç saniye); iki dağılım aynıdır. Kayıt başına W-bMAE ile onset korelasyonu r = 0.314.

| Onset (sn) | W-bMAE seen | W-bMAE unseen | Kayıt (seen / unseen) | MAE [0,40) | MAE [40,90) | MAE [90,300) | MAE [300,600] |
|---|---|---|---|---|---|---|---|
| ≤ 330 | 36.69 | 43.62 | 68 / 59 | 20.0 | 48.7 | 88.6 | 48.4 |
| 330–480 | 27.63 | 29.05 | 88 / 84 | 13.4 | 28.0 | 61.0 | 80.8 |
| 480–690 | 33.95 | 36.26 | 127 / 110 | 14.3 | 26.8 | 70.4 | 177.3 |
| 690–990 | 41.27 | 43.32 | 143 / 122 | 13.1 | 25.3 | 93.6 | 263.9 |
| 990–1265 | 46.82 | 50.26 | 83 / 74 | 15.2 | 28.0 | 110.3 | 297.9 |
| 1265–1500 | 44.36 | 47.10 | 56 / 51 | 12.2 | 24.0 | 100.6 | 323.7 |

Bin MAE sütunları iki rejim birlikte, kayıt ortalamasıdır.

| Alt küme | W-bMAE seen | W-bMAE unseen | Kayıt (seen / unseen) |
|---|---|---|---|
| onset 30 sn ızgarasında | 37.33 | 40.14 | 482 / 432 |
| ızgara dışı | 42.37 | 45.11 | 83 / 68 |

Kayıt başına W-bMAE yüzdelikleri (%10 / %50 / %90 / %99): seen 21.8 / 37.5 / 54.2 / 82.5; unseen 23.0 / 39.7 / 57.8 / 103.6.

### En kötü 10 kayıt (iki rejim birlikte)

| Rejim | Kat | Kayıt | Onset (sn) | W-bMAE | [0,40) | [40,90) | [90,300) | [300,600] | Ort. yanlılık |
|---|---|---|---|---|---|---|---|---|---|
| unseen | 0 | sub-147 / ses-001 | 90 | 238.3 | 232.6 | 249.8 | – | – | +242.2 |
| unseen | 0 | sub-132 / ses-005 | 390 | 122.6 | 100.9 | 124.4 | 186.3 | 139.2 | +132.2 |
| unseen | 1 | sub-185 / ses-001 | 1080 | 113.0 | 80.8 | 157.9 | 77.6 | 316.9 | −207.7 |
| unseen | 1 | sub-020 / ses-001 | 210 | 111.5 | 62.0 | 176.5 | 168.3 | – | +150.0 |
| seen | 3 | sub-185 / ses-001 | 1080 | 107.4 | 80.8 | 152.4 | 88.9 | 203.1 | −122.0 |
| unseen | 1 | sub-015 / ses-002 | 270 | 105.0 | 55.3 | 155.7 | 186.3 | – | +159.2 |
| unseen | 0 | sub-038 / ses-005 | 150 | 103.6 | 38.6 | 182.6 | 188.5 | – | +146.6 |
| unseen | 3 | sub-058 / ses-001 | 300 | 102.4 | 76.6 | 149.6 | 109.7 | – | +108.8 |
| seen | 1 | sub-038 / ses-005 | 150 | 98.1 | 30.0 | 169.6 | 205.8 | – | +146.9 |
| unseen | 1 | sub-015 / ses-003 | 210 | 97.8 | 50.2 | 145.4 | 177.1 | – | +144.8 |

10 kaydın 8'i onset ≤ 390 sn olan kısa kayıtlar ve hepsinde yanlılık pozitif (kalan süre fazla tahmin edilmiş); 2'si aynı kayıt (sub-185 / ses-001, onset 1080 sn), iki rejimde de listede.

---

## 10. Açık sorular ve riskler

### Doğrulanmış kusurlar

1. **mix3_540: NaN içeren pencere kaydı kilitliyor [Ç verify3].** 20. pencereye NaN konduğunda o pencerenin tahmini 0.00 oldu ve sonraki 39 pencerenin 39'u tam 0 çıktı. Neden: ham pencere `nan_to_num` uygulanmadan CNN kodlayıcıya giriyor (`mix3_540/submission.py:551-552`), NaN gömme GRU-emb gizli durumuna ve gömme tabanına yerleşiyor (`:559, :581`), sonda `nan_to_num(nan=0.0)` 0 üretiyor (`:667`). `final540` etkilenmiyor (aynı testte tahmin 239.8, sonraki farklar 2–17 sn). Tamamen sıfır pencere veya tek düz kanal mix3'te soruna yol açmıyor. Açık veride NaN pencere yok; mühürlü veride olup olmadığı bilinmiyor.
2. **Dokümanlar güncel değil.** `status.md:50, 287-288` ve `submissions_log.md:24` mix3_540'ı "onay bekliyor" gösteriyor; 11 Kasım gönderimi, mühürlü skor 38.66 ve mühürlü süre depoda kayıtlı değil. CLAUDE.md "5 hakkın 1'i kullanıldı / kalan 4" yazıyor.
3. **İzlenmeyen üretilmiş dosyalar.** `submissions/final540/`, `mix3_540/`, `mix3_tv/` altındaki `submission.py` dosyaları git'te izlenmiyor (`git status`). `final540/submission.py`, izlenen `mix_gru10_hgb/submission.py` ile birebir aynı [Ç `diff`]; `mix3_540/submission.py`'nin izlenen bir kopyası yok (HEAD'deki builder'dan yeniden üretilebilir).
4. **Yorum / kod tutarsızlıkları.** `cnn_encoder.py:7` "x / 100" diyor, kod `SCALE = 20` (`:38`). Üretilen `submission.py` başlığı hâlâ "HistGradientBoostingRegressor" ve `Solver.name = "HGBCausalFeatures"`. `check_stream_mix3` farkı dokümanda 0.055 sn (mix3_tv için); gönderilen mix3_540 için log 0.114 sn.

### Doğrulanamayan veya test edilmemiş noktalar

5. **`v4-sealed` imajı.** Yerelde yok; tüm Docker testleri `v3` ile. İki gönderim de mühürlü ortamda skor üretti, yani pratikte çalıştı; ancak yeni bir ZIP için gönderim öncesi `v4-sealed` testi mümkün olmadı.
6. **mix3 eğitim tekrarlanabilirliği.** final540 için bit düzeyinde tekrar kanıtlı; mix3'ün TCN ve CNN bileşenleri (cuDNN evrişim) için ikinci koşu karşılaştırması yok. Denetim eşiği olan ±2σ'nın nasıl hesaplanacağı dokümanlarda tanımlı değil.
7. **mix3 mühürlü süresi** kayıtlı değil; gönderim öncesi tahmin 2 200–2 300 sn (sınır 3 600 sn).
8. **300 pencereden uzun kayıtlar.** Açık veride en uzun akış 300 pencere (1500 sn); geçen süre girdisi 1500 sn'de kırpılıyor. Daha uzun kayıtlardaki davranış test edilmedi [D `status.md:307-308`].
9. **Kanal adı bağımlılığı.** "eye" öznitelikleri AF7 ve AF8'i ada göre arıyor (`find_channel`, `causal_features.py:101-107`); `meta["ch_names"]` farklı adlandırılırsa `ValueError` ile çöker. Docker'da `['TP9','AF7','AF8','TP10']` geldiği loglandı.
10. **Sabit 5 sn varsayımı.** Dizi modellerinin geçen süre girdisi `gru_t * 5.0` ile sabit kodlu (`mix3_540/submission.py:562`), HGB akışı ise pencere süresini `x.shape[-1] / sfreq`'den alıyor (`:621`).
11. **CV'nin sınırları.** Tek bölünme tohumu, tekrarlı CV yok; katlar arası SS 1.5–1.9 sn, seed gürültüsü ≈ 0.3–0.5 sn; kabul eşiği 0.3 sn. Epoch LOFO'sunda seen katları arasında kayıt maskelemesi yok. İki gönderimde çift CV, mühürlü skoru 0.72 ve 0.74 sn fazla tahmin etti (iki veri noktası).
12. **Mühürlü seen/unseen ayrımı.** Mühürlü panoda yalnız makro skor görünüyor; rejim bazında mühürlü skor **bulunamadı**. Açık testte unseen kişi yok (40/40 kişi train+val'da).
13. **Mühürlü küme özellikleri.** Boyut, onset dağılımı ve 30 sn ızgara oranı bilinmiyor [D].
14. **Leaderboard süreleri.** > 4 500 sn süreler görülüyor, sınır 3 600 sn; sürenin neyi kapsadığı doğrulanmadı [D `status.md:304-305`].

### Yarım kalmış / ertelenmiş işler [D]

- İğcik ve K-kompleks detektörleri, bipolar derivasyonlar (plan C maddeleri) — kodda yok.
- Göz kırpma eşiği ayarı (pencere başına ortalama 0.15 kırpma "düşük" notu).
- Uçtan uca CNN → GRU, 2D CNN, 10 seed'li kodlayıcı — çalıştırılmadı.
- GRU girdisine "dyn" öznitelikleri, ordinal kayıp — "açık sorular"da listeli, denenmedi.
- Yedek aday (GRU+TCN)/2 + HGB(c) için 540 eğitimi yapılmadı (mix3 onun yerini aldı).
- "Kapatılan yönler" (kullanıcı kararı, 11 Kasım): deneğe özgü bilgi / kNN örtük önsel ve 10 dk'dan uzun bağlam.

### Hak durumu

5 hakkın 2'si kullanıldı (final540 10 Kasım, mix3_540 11 Kasım) [B]; kalan 3. Son tarih 21 Kasım 23:59 CET; günde en fazla 1 gönderim. Depoda hazır ama gönderilmemiş ZIP'ler: `mix_gru_hgb`, `mix_gru10_hgb`, `hgb_ens_c`, `hgb_v3_np`, `hgb_v3`, `hgb_v2_s0`, `mix3_tv` — hepsinin çift CV'si gönderilmiş iki modelden kötü veya onlarla aynı tarif.
