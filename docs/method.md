# Yöntem notu — FİNAL (23 Kasım 2026)

EEG/EMG Foundation Challenge 2026, Track 03 (Sleep Onset). Takımın final skoru = en iyi mühürlü gönderim **mixI_540: W-bMAE makro 32.92**
(W-bMAE 33.58, bMAE 100.64, MAE 191.61; mühürlü süre 2178.67 s), gönderim #11, 20 Kasım 2026. Yarışma sonu sıralama: **26.** Son
gönderim #12 mixK_540 (mixI + ek tohumlar) 32.99 → final skoru değiştirmedi. Bütün gönderimler: `docs/submissions_log.md`.

## 1. Görev, veri, kısıtlar
- Girdi: Muse kafa bandı, 4 kanal (TP9, AF7, AF8, TP10), 128 Hz, µV; 5 sn pencereler kronolojik sırayla, birer birer. Çıktı: her pencere için
  ilk N2 epoch'una kalan saniye, 600 ile sınırlı. Model yalnız o ana kadarki pencereleri görür; durum her kayıtta sıfırlanır.
- **Veri: yalnız yarışma verisi Interaxon2026Muse** (NEMAR nm000287 v1.0.0, CC-BY-NC-SA-4.0; 540 kayıt / 203 kişi). Final model 540 kaydın
  tamamıyla (train + val + açık test) eğitildi. Akış penceresi tablosuna ek olarak ham BrainVision dosyalarının akışta kullanılmayan
  **onset-sonrası 300 sn'si** yalnız N2 dedektörünün eğitim pozitifleri için kullanıldı (aynı veri kümesi, `scripts/n2det_extract.py`).
- **Dış veri beyanı:** HMC Sleep Staging (PhysioNet, CC BY 4.0) 30 kayıtla denendi (12–13 Kasım), Muse'a taşınmadığı için bırakıldı;
  **hiçbir gönderilen modelde kullanılmadı** (ayrıntı `docs/external_data.md`). Sleep-EDF yalnız incelendi, indirilmedi.
- Sızıntı kuralları: kayıt uzunluğu, kayıt sonu, gelecek pencere, dosya adı, denek kimliği hiçbir öznitelikte yok (açık veride her onset kayıt
  sonundan tam 300 sn önce — kullanılmadı). Kayıt genelinde istatistik yok; bütün ön işleme pencere başına ya da nedensel. 13 Kasım FAQ
  güncellemesinden sonra 30 sn onset ızgarası ve epoch-hizalı öznitelikler bütün adaylardan çıkarıldı (etiketleme düzenliliği).
- Mühürlü kurgu (16 Kasım FAQ): kayıtların açılış dakikaları modele verilmiyor, geçen süre ilk verilen pencereden. Buna dayanıklılık eğitimde
  başlangıç kırpma artırmasıyla sağlandı; çıkarımda kayma bilgisi kullanılmaz.

## 2. Final model: mixI_540 = 0.5 · ctx_c3_540 + 0.5 · ctxE_c2hs3p_540
Karar öncesi, metrik ağırlıklı ve normalleştirilmiş dağılımların sabit ağırlıklı karışımı; karar = sürekli ağırlıklı medyan (25 hücrelik ızgara,
B sınıf kenarları [0, 10, 20, 30, 40, 60, 90, 150, 225, 300, 450, 600] ∪ B2 30 sn dilimleri; ω = metriğin etkin pencere ağırlıkları).
Eşdeğer biçim:

    W = 0.4 · ( ⅔ q(GRU ctx) + ⅙ q(TCN) + ⅙ q(GRU-emb) )  +  0.6 · ( ½ norm(P_HGBctx · ω) + ½ norm(P_HGBE-s3p · ω) )

| Bileşen | Ne | Girdi | Tohum / boyut | Eğitim |
|---|---|---|---|---|
| Bağlamlı N2 dedektörü | CNN kodlayıcı (35k param.) + son 6 pencerenin gömmesi üzerinde GRU(32), 5 sınıf ("yakınlık" = beklenen sınıf / 4) | pencerenin ham 4 × 640 örneği (/20, ±15) | 3 tohum (0, 1, 2), epoch 18 / 12 / 16 | pozitif = onset sonrası 60 pencere; akış pencereleri kalan-süre sınıflarıyla (≥ 300 / 150–300 / 60–150 / 0–60 sn) |
| Nedensel özetler | yakınlığın EMA 30 sn / 2 dk, koşan maks., 30 sn artış, değer | dedektör çıktıları (yalnız geçmiş) | – | – |
| HGB ctx | HistGradientBoostingClassifier × 4 üye (B, B_h1, B_h2, B2) | 817 nedensel öznitelik + 5 yakınlık özeti = 822 | random_state 0; n_iter 200 / 200 / 300 / 200 | özgün 80 975 + 64 935 kırpılmış satır (kayıt başına 1 kopya) |
| HGB E-s3p | aynı 4 üye, kırpma E tablosu | 822 | B ve B_h2 × random_state 0, 1, 2; B_h1, B2 × 0; üye ağırlığı ¼ (tohum modeli 1/12) | kırpma E: kayıt başına 2 kopya (U[2,5] dk, U[0,10] dk) + özgün satırların ~%10'u (119 641 satır) |
| GRU ctx | GRU-uw (Linear 135→64 + GELU + LayerNorm, GRU 64 × 2, dropout 0.2) | 62 pencere özniteliği + ilk-2-dk farkı + log geçen süre (125) + yakınlık özetleri ve farkları (10) | 10 tohum (0–9), 20 epoch | kırpma: %50 yok, %35 ilk 1–12 pencere, %15 ilk 13 … (onset − 60 sn) |
| TCN | nedensel dilate konvolüsyon (çekirdek 3, dilasyon 1–32, alıcı alan 127 pencere), 84k param. | 125 | 10 tohum, 10 epoch | artırmasız |
| GRU-emb | CNN kodlayıcı gömmesi (32) + GRU-uw | 125 + gömme + gömme farkı (189) | 3 kodlayıcı/GRU çifti, kodlayıcı 12, GRU 10 epoch | artırmasız |

817 nedensel öznitelik = pencere başına taban (40: kanal başına bant güçleri, varyans, çizgi uzunluğu, maks. genlik…), spektral şekil (20) ve göz
hareketi (2) + nedensel dinamikler (EMA'lar, eğimler, 5 dk varyans / lag-1 otokorelasyon), koşan maks./min'e göre göreli değerler, ilk-2-dk
farkları, geçen süre (`scripts/causal_features.py`, `scripts/ablation_c.py` "dyn_maxmin_spec_eye"). Girdi temizliği: NaN / inf / düz kanal →
kanalın son temiz penceresi ya da son geçerli örnek (sarmalayıcı `_clean`). Kalibrasyon, yumuşatma, ızgara kapalı.

## 3. Üretim zinciri (baştan ZIP'e; hepsi `scripts/` içinden `../.venv/bin/python`)
Ortam: Python 3.12.3, torch 2.11.0+cu128, scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1, joblib 1.6.0, neuralbench 0.4.0; RTX 3050 Laptop (4 GB),
12 çekirdek, 11 GB RAM (WSL2). Veri: `neuralbench eeg _sleep_onset_stream` hazırlığıyla `~/.neuralbench/config.json` DATA_DIR.

| # | Komut | Çıktı (`~/neuralbench/save/`) | Süre |
|---|---|---|---|
| 1 | `extract_features.py` | features_base.npz, features_ext.npz (pencere öznitelikleri) | ~dk |
| 2 | `extract_raw.py` ; `n2det_extract.py` | raw_windows.npy (80 975 × 4 × 640); onset-sonrası pencereler | ~dk |
| 3 | `train_n2det_final.py detctx` | n2det_ctx_final/ (detctx.joblib; eğitim satırları için kişi bazlı 5 iç katlı fold-dışı ve kırpma-kesin yakınlık close_oof_L) | 3663 s GPU |
| 4 | `train_n2det_final.py hgbmc --det=ctx` | ens_c_all_ctx_crop/ (HGB ctx) | 1848 s CPU (+ tablo) |
| 5 | `train_gru_n2c.py --det=ctx` | gru_ctx_all/ (GRU ctx, 10 tohum) | 157 s GPU |
| 6 | `train_mix3_final.py --all --deterministic --tag=det` | mix3det_all/ (yalnız TCN, CNN kodlayıcı, GRU-emb kullanılır) | 322 s GPU |
| 7 | `train_n2det_final.py mix --alpha=0.5 --ens=ens_c_all_ctx_crop --seq=mix3det_all --gru-from=gru_ctx_all --out=ctx_c2_all` | ctx_c2_all/ (gönderim #7) | s |
| 8 | `train_n2det_final.py hgbmc --det=ctx --cropE --seeds=0,1,2` | ens_c_all_ctxE_s3_crop/ (HGB E, 4 üye × 3 tohum) | 4765 s CPU |
| 9 | `train_n2det_final.py mix --alpha=0.4 --ens=ens_c_all_ctxE_s3_crop --seq=mix3det_all --gru-from=gru_ctx_all --out=ctxE_c2hs3_all` | ctxE_c2hs3_all/ | s |
| 10 | `derive_bundles.py` | ctx_c3_all (gönderim #8), ctxE_c2hs3p_all (#10), mixI_all (#11) — 23 Kasım'da yeniden türetildi, paketlenenlerle **bit düzeyinde aynı** | s |
| 11 | `build_hgb_submission.py --model ~/neuralbench/save/mixI_all/model.joblib --name mixI_540 --smooth-a 0` | submissions/mixI_540.zip (20.24 MB, SHA-256 `5ece144bfffd28c67e1b554b2ce7c48d27958525ac817732914c9b16cb2c1a5a`) | s |
| 12 | `mixI_build_tests.sh` (+ `mixI_equiv_test.py`) | akış ↔ toplu, eşdeğerlik, temizlik, 9 arıza, bağlam/reset, Docker v3 ×2 | ~1 sa |

Epoch ve ağaç sayıları çift CV'nin LOFO seçimlerinin medyanlarıdır (25'e yuvarlı) ve 3–9. adımlar bunları CV çıktılarından (`~/neuralbench/save/cv/dual/`)
okur: dedektör `n2det_ctx_inner.py 6 0|1|2`, HGB `ens_crop_cv.py ctx6`, GRU `b2_seq_cv.py` (`b2_seq.json` "uw_n10_ctx6_crop"), kırpma E ve tohum
ortalaması kaymalı CV ile (`shift_cv.py`, İş D/E/G). Sabit değerler yukarıdaki tabloda; CV yeniden koşulmadan da bu değerlerle eğitim yapılabilir. Not: HGB ctx ağaç sayıları eğitim kaydına
(`ens_c_all_ctx_crop/results.json`) göre 200 / 200 / 300 / 200; `submissions_log.md`'deki eski "125/175/300/200" notu hatalıydı.

**Toplam hesaplama (final eğitim):** ≈ 1.0 sa GPU (dedektör) + ≈ 0.1 sa GPU (dizi modelleri) + ≈ 1.9 sa CPU (HGB ctx + HGB E) ≈ **3 saat**, tek
RTX 3050 dizüstü bilgisayarda. Geliştirme sürecindeki CV koşuları bunun çok üstünde (kesin toplam kaydedilmedi). Çıkarım: mühürlü 2178.67 s (sınır
3600 s); yerel ≈ 37–40 ms/pencere (CPU).

## 4. Tekrarlanabilirlik
- **Bütün sinir ağı bileşenleri deterministik eğitildi** (`repro.set_deterministic`: cuDNN deterministic, benchmark kapalı,
  `torch.use_deterministic_algorithms(True)`, CUBLAS_WORKSPACE_CONFIG; sabit tohumlar). HGB: sabit random_state.
- Doğrulamalar: aynı tohumla yeniden eğitim bit düzeyinde aynı — GRU ctx tohum 0 (20 Kasım, ayrı süreçte; ağırlıklar, μ/σ, ω), HGB ctx B ve B_h2
  random_state 0 (20 Kasım; ağaçların bütün düğüm alanları), final540 (10 Kasım; HGB + GRU tam). Paket türetmeleri (adım 10) bit düzeyinde aynı.
  Paketleyici regresyonları: yeni paketleyiciyle yeniden kurulan paketlerin akış tahminleri gönderilenlerle bit düzeyinde aynı.
- **CNN kodlayıcı notu:** ilk mix3_all (11 Kasım) kodlayıcısı deterministik ayar olmadan eğitilmişti (yeniden eğitimde ağırlık farkı maks.
  0.36); final zincirdeki kodlayıcı ve GRU-emb **mix3det_all**'dan (16 Kasım, deterministik) — TCN deterministik ayardan bağımsız olarak aynı
  (40.3806 iki sürümde). İki kodlayıcı arasında GRU-emb açık test farkı 49.56 → 49.07 (yalnız kodlayıcı değişti).
- Bilinen küçük platform farkı: bağlamlı dedektör eğitimde GPU'da, sarmalayıcıda CPU'da koşar (~3e-5 fark; 40 açık test kaydında 4 pencerede bir
  HGB bölmesi eşikte döner, maks. 2.3 s). CPU referansıyla akış ↔ toplu farkı maks. 0.016 s.
- Docker (lionelyneuro/neural-compet:v3) iki koşu bit düzeyinde aynı skor (28.300361165769242). Import satırları: joblib, numpy, torch,
  benchmark_utils, scipy.signal (find_peaks, welch) — final540 ile aynı.

## 5. Model seçimi
- Çift CV = (görülmemiş kişi 5 kat + görülmüş kişi 5 kat) / 2, train + val 500 kayıt; açık test yalnız son kontrol. 16 Kasım'dan sonra birincil
  ölçüt **kaymalı çift CV**: dışarıdaki kayıtlar kaydın ortasından (D = U[2,5] dk; İş D'de 8 gönderimin mühürlü farklarını yeniden üreten dağılım)
  akıtılır.
- mixI: ağırlık w = 0.5 **önceden sabitlendi** (CV'den seçilmedi); CV yalnız akıl sağlığı için (M hiçbir dağılımda iki bileşeninden > 0.10 kötü değil;
  en kötü +0.016). Gerekçe (19 Kasım): 0.2 s altındaki yerel farklar mühürlüye taşınmadı; karışım yalnız mühürlüde kanıtlanmış iki modelden.
- Mühürlü gelişim (16 Kasım yeniden skorlaması sonrası): aday 1 47.42 → aday 2 (kırpma artırmalı HGB + çok sınıflı dedektör) 34.41 → cand3 33.72 →
  bağlamlı dedektör ctx_c2 33.12 / ctx_c3 33.09 → İş E/G 33.29 / 33.13 → **mixI 32.92** → mixK 32.99.

## 6. Olumsuz sonuçlar (kısa)
- 30 sn onset ızgarası: yerelde −1.05 ama etiketleme düzenliliği → 13 Kasım'dan sonra yasak, çıkarıldı (mühürlüde de zarar verdi).
- HMC ile eğitilmiş evreleyici Muse'a taşınmadı (Spearman −0.10, AUC 0.67).
- Kırpma E ile yeniden eğitim (İş E) ve HGB tohum ortalaması (İş G): yerelde −0.18 … −0.22, mühürlüde nötr / zararlı.
- Karar katmanı (sıcaklık, aralığa göre α, tam metrik ağırlığı) ±0.1; geri sayım süzgeci (önceki dağılımın 5 sn kaydırılıp birleştirilmesi) en iyi
  +0.001; ham EEG'den uçtan uca CNN → GRU (İş L) tek başına 39.8 ama karışıma −0.015 (20 Kasım; `docs/status.md`).
- Hata anatomisi (kaymalı CV): kalan hatanın çoğu [90,300) ve [300,600] ayrımında; uzak pencerelerde metrik ağırlıklı medyan sistematik ~270 sn
  eksik tahmin eder (yakın aralıkların ×10 / ×5 ağırlığına karşı korunma) — bilgi sınırı, son işlemle düzelmedi.

# Ek: geliştirme sürecindeki adayların notları (14–20 Kasım)

### N2 dedektörü (Work 2; 14 Kasım'dan itibaren adaylarda)
- Tek pencerelik CNN (`scripts/cnn_encoder.Encoder`, 2 çıkış, ~35k parametre), girdi 4 × 640 ham µV örnek (×1/20, ±15 kırpma).
- **Eğitim verisi yalnız açık eğitim kayıtlarının EEG'si:** pozitif = ilk N2 epoch'undan sonraki 5 sn'lik pencereler
  (ham BrainVision dosyalarının akış boru hattında kullanılmayan onset-sonrası bölümü; `scripts/n2det_extract.py`),
  negatif = onset'e ≥ 300 sn kalan pencereler. Aradaki pencereler eğitimde kullanılmaz.
- **Çıkarımda yalnız o ana kadarki EEG kullanılır:** her pencere için P(N2) o pencerenin ham örneklerinden hesaplanır;
  nedensel özetler (EMA 30 sn / 2 dk, koşan maksimum, 30 sn artış; `scripts/n2det_stream.py`) yalnız geçmiş pencerelerin
  P'lerinden. Gelecek pencere, kayıt uzunluğu, kayıt sonu, dosya adı veya denek kimliği kullanılmaz.
- **Eğitim pencereleri, açıkça:** her açık eğitim kaydı için (a) POZİTİF = ilk N2 epoch'unun başlangıcından sonraki
  60 pencere (onset+0 … onset+300 sn; ham dosyalar bu bölümü içeriyor, akış boru hattı kullanmıyor), (b) NEGATİF = akıştaki,
  onset'e ≥ 300 sn kalan pencereler, (c) onset'e 0–300 sn kalan pencereler eğitimde KULLANILMAZ (etiket belirsizliği).
  Dedektör ikili sınıflandırıcıdır; onset zamanı, kayıt uzunluğu veya pencere indeksi girdi değildir.
- **Çıkarım:** değerlendirici her pencereyi sırayla verir; P(N2) yalnız o pencerenin 640 örneğinden hesaplanır, özetler yalnız
  önceki pencerelerin P değerlerinden; kaydın sonu/uzunluğu/gelecek pencereler hiçbir hesaba girmez.
- Sızıntı kuralı: HGB üyeleri eğitilirken eğitim satırlarının P'si fold-dışıdır (kişi bazlı 5 iç kat); CV'de dedektör her katta
  yalnız o katın eğitim kayıtlarıyla eğitildi.
- Deterministik eğitim: `scripts/repro.set_deterministic` (cuDNN deterministic, `use_deterministic_algorithms`), tohum 0.

### Karar katmanı
- 13 Kasım FAQ güncellemesine göre 30 sn onset ızgarası (π / π(D)) ve epoch-hizalı öznitelikler yeni adaylardan çıkarıldı.
  Karar = sınıf dağılımı × metrik ağırlıkları ile sürekli ağırlıklı medyan (ızgarasız).

### Aday 1 (15 Kasım): mix3_n2det_cropg_540 — HGB(n2det) + GRU (kırpma artırmalı) + TCN + GRU-emb, α 0.6, ızgarasız
- **Bileşenler:** (1) HGB(c): 4 dağılım sınıflandırıcısı (B, B_h1, B_h2, B2), 817 nedensel öznitelik + 5 dedektör özeti = 822;
  (2) GRU-uw 10 tohum; (3) TCN 10 tohum; (4) GRU-emb 3 tohum (3 CNN kodlayıcı). Dizi bileşenlerinin metrik-ağırlıklı ızgara
  dağılımları eşit ağırlıkla ortalanır, HGB dağılımıyla α = 0.6 (dizi) / 0.4 (HGB) karıştırılır; karar = sürekli ağırlıklı
  medyan. Izgara (30 sn), kalibrasyon ve yumuşatma kapalı. Dedektör özetleri yalnız HGB girdisinde (dizi girdilerinde yok).
- **GRU başlangıç kırpma artırmasıyla eğitildi:** her epoch ve her eğitim kaydı için %50 kırpma yok, %35 ilk 1–12 pencere
  (5–60 sn) atılır, %15 ilk 13 … (onset − 60 sn) pencere atılır; girdiler (geçen süre, ilk-2-dk tabanı) kırpılan
  başlangıçtan yeniden kurulur, hedefler değişmez (`scripts/gru_cv.py run_fold cfg["crop"]`). Amaç: akışın kaydın başından
  başlamadığı duruma dayanıklılık (13 Kasım FAQ). TCN ve GRU-emb artırmasız (çift CV'de artırma bu ikisinde zararlıydı).
  Artırma yalnız eğitim kayıtlarının kendi pencerelerini kullanır; etiket/kayıt düzenliliği kullanmaz.
- **Seçim (çift CV, 500 kayıt, ızgarasız karar):** 38.86 (unseen 39.93, seen 37.79); referans ızgarasız mix3 39.40'a göre −0.54,
  10 katın 8'inde iyi; α iç içe LOFO ile 10 katta da 0.6 (`scripts/b2_mix.py`, `OUT/b2_mix_cropmix.json`).
- **Eğitim (540 kayıt), sırayla** (hepsi `scripts/` içinden, `.venv/bin/python`):
  1. `train_n2det_final.py det` — dedektör, tohum 0, 13 epoch (çift CV LOFO medyanı), deterministik; eğitim satırları için
     kişi bazlı 5 iç katlı fold-dışı P (`n2det_final/p_oof.npy`).
  2. `train_n2det_final.py hgb` — 4 HGB üyesi, random_state 0, iterasyon 150 / 150 / 250 / 150 (çift CV LOFO medyanı, 25'e
     yuvarlı), eğitim satırlarında fold-dışı P'nin özetleri → `ens_c_all_n2det/`.
  3. `train_mix3_final.py --all --deterministic --crop --epochs=gru:20,tcn:15,gemb:10 --tag=crop` → `mix3crop_all/`;
     buradan YALNIZ GRU-uw alınır (10 tohum 0–9, 20 epoch = artırmalı çift CV LOFO medyanı).
  4. TCN (10 tohum × 10 epoch), CNN kodlayıcı (3 × 12 epoch), GRU-emb (3 × 10 epoch): `mix3_all/` (11 Kasım,
     `train_mix3_final.py --all`; gönderim #2 mix3_540 ile aynı ağırlıklar).
  5. `train_n2det_final.py mix --alpha=0.6 --seq=mix3_all --gru-from=mix3crop_all --out=mix3_n2det_cropg_all`
  6. `build_hgb_submission.py --model ~/neuralbench/save/mix3_n2det_cropg_all/model.joblib --name mix3_n2det_cropg_540 --smooth-a 0`
- **Tekrarlanabilirlik:** 1–3 deterministik (cuDNN deterministic, sabit tohumlar). 4. adımdaki TCN / kodlayıcı / GRU-emb
  deterministik ayar olmadan eğitildi; yeniden eğitimde bit düzeyinde aynı olmaz (ölçülen etki ≈ 0.2 s, denetim toleransı
  ±2σ içinde).
- **Ortam:** Python 3.12.3, torch 2.11.0+cu128, scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1, joblib 1.6.0; RTX 3050 Laptop
  (4 GB). Eğitim süresi: dedektör ≈ 6 dk, HGB 4 üye ≈ 16 dk (CPU), dizi modelleri ≈ 6 dk (GPU).
- **Çıkarım:** yalnız numpy / torch / joblib / scipy (find_peaks, welch); import satırları final540 ve mix3_540 ile aynı.
  ZIP SHA-256, Docker koşuları ve süreler: `docs/submissions_log.md`.
- **Beklenen mühürlü skor ≈ 37.25** (yerel 38.86; referans 39.40 ↔ mühürlü 37.79, fark −1.61).

### Aday 2 (16 Kasım): cand2_cropGc_540 — HGB(c, yakınlık + kırpma) + GRU (kırpma + yakınlık) + TCN + GRU-emb, α 0.6, ızgarasız
- **Dedektör (çok sınıflı):** aynı CNN mimarisi, 5 sınıf: onset'e ≥ 300 sn / 150–300 / 60–150 / 0–60 sn kalan akış pencereleri ve onset
  sonrası pencereler (ham dosyaların akışta kullanılmayan onset-sonrası 300 sn'si). Önceki ikili dedektörden farkı: onset'e 0–300 sn kalan
  pencereler de kalan-süre sınıflarıyla eğitimde. Çıktı = "yakınlık" = beklenen sınıf indeksi / 4; 3 tohum (0, 1, 2), çıkarımda ortalama.
  Girdi yalnız o pencerenin 640 ham örneği; özetler (EMA 30 sn / 2 dk, koşan maksimum, 30 sn artış) yalnız geçmiş pencerelerden.
  Kayıt uzunluğu, kayıt sonu, gelecek pencere, dosya adı veya denek kimliği kullanılmaz. Sınıf etiketleri yalnız eğitimde kullanılır.
- **Başlangıç kırpma artırması:** HGB eğitim tablosuna her eğitim kaydının bir kırpılmış kopyası eklenir (%70 ilk 1–12 pencere, %30 ilk
  13 … (uzunluk − 12) pencere atılır; tohum 0); GRU her epoch ve kayıtta yeniden kırpılır (%50 yok, %35 1–12, %15 13 … onset − 60 sn).
  Kırpılan kopyada bütün nedensel öznitelikler ve dedektör özetleri kırpılan başlangıçtan yeniden hesaplanır; hedefler değişmez. Amaç:
  akışın kaydın başından başlamadığı duruma dayanıklılık (13 Kasım FAQ).
- **Sızıntı kuralı:** CV'de dedektör her katta yalnız o katın eğitim kayıtlarıyla eğitildi; HGB / GRU eğitim satırlarının dedektör çıktısı
  iç katlardan fold-dışı. Finalde eğitim satırlarının yakınlığı kişi bazlı 5 iç katlı fold-dışı.
- **Deterministik:** bütün sinir ağı bileşenleri `repro.set_deterministic` ile (cuDNN deterministic, sabit tohumlar); HGB random_state 0.
- Seçim, komutlar, kontroller, SHA-256: `docs/submissions_log.md` "cand2_cropGc_540"; seçim tabloları `B_ASAMASI_RAPORU.md` 16 Kasım.
- **Beklenen mühürlü skor ≈ 36.7–36.9** (yerel 38.49).

### İş E adayları (19 Kasım): ctxE_c3_540 ve ctxE_c2h_540 — 16 Kasım mühürlü kurgusuna göre kırpma (ızgarasız)
- **Mühürlü kurgu (16 Kasım FAQ):** kayıtların açılış dakikaları modele verilmiyor; geçen süre ilk verilen pencereden. Kayma dağılımı,
  gönderilmiş 8 modelin mühürlü skor FARKLARININ 40 açık test kaydında yeniden üretilmesiyle seçildi (İş D, `scripts/shift_fit_d.py`):
  D = tekdüze U[2,5] dk. Bu yalnız eğitim artırmasının dağılımını belirler; çıkarımda kayma bilgisi, kayıt uzunluğu veya kayıt sonu kullanılmaz.
- **Bağlamlı N2 dedektörü (İş C):** pencere başına CNN gömmesi (Aday 1–2'deki kodlayıcı), son 6 pencerenin gömmesi üzerinde GRU(32), 5 sınıflı
  "yakınlık" çıkışı, 3 tohum ortalaması; akış başlangıcından önceki konumlar sıfır dolgu (yalnız akışın gördüğü pencereler). Eğitim
  pencereleri ve sızıntı kuralı yukarıdaki dedektörle aynı (fold-dışı çıktılar kişi bazlı iç katlardan).
- **Kırpma E (eğitim artırması):** dizi modellerinde her epoch ve kayıtta başlangıç k: %10 kırpmasız, %50 U[24,60] pencere (2–5 dk), %40
  U[0,120] pencere (0–10 dk); HGB tablosunda kayıt başına 2 kopya (U[2,5] dk ve U[0,10] dk) + özgün satırların rastgele ~%10'u (seed 0).
  Pencere kalmayan kopya atılır. Her kopyada bütün nedensel öznitelikler kırpılan başlangıçtan yeniden akıtılır (ilk-2-dk tabanı, EMA'lar,
  koşan ekstremumlar, geçen süre, dedektör özetleri); finallerde dedektör bağlamı da kırpmadan başlar (kırpma-kesin `close_oof_L`).
  CV'de dış-kat bağlamlı dedektörler saklanmadığı için kırpılmış kopyanın ilk 5 penceresinde bağlam kırpma öncesini görür (bilinen yaklaşım).
- **ctxE_c3_540** = HGB(c) 4 üye kırpma E + bağlamlı yakınlık (`ens_c_all_ctxE_crop`, n_iter 200 / 200 / 300 / 200) + GRU-uw 10 tohum kırpma E +
  bağlamlı yakınlık girdisi (`gru_ctxE_all`, 20 epoch), α 0.4. Önceden tanımlanmış iki bileşen setinden biri (ctx_c3 seti); kabul kuralını geçti
  (kaymalı çift CV, ctx_c3 tarifine göre D −0.22, 8/10; D' = U[0,8] dk −0.21).
- **ctxE_c2h_540 (HİBRİT, SONRADAN SEÇİLDİ)** = aynı HGB E + ctx_c2'nin eski dizi seti: GRU-uw kırpma (eski 50/35/15 karışımı) + bağlamlı
  yakınlık (`gru_ctx_all`), TCN 10 tohum ve GRU-emb 3 tohum (`mix3det_all`, artırmasız); α 0.4 (iç içe LOFO). **Bu satır önceden tanımlanmış
  aday değildi**: İş E karışım tablosundaki 5 bilgi satırından biriydi ve sonuçlar görüldükten sonra seçildi (D −0.22, 8/10; D' −0.31, 8/10).
  Önceden tanımlanmış ctx_c2 seti (TCN / GRU-emb da kırpma E ile) kat sayısında kaldı (D −0.19, 6/10). Bu seçim yanlılık taşır; kullanıcı
  iki dizi setinin mühürlüde taşınmasını ölçmek için iki paketi birlikte gönderdi.
- Ağaç sayısı / epoch: İş E tarifleri için ayrı k=0 seçimi yapılmadı; eşdeğer kırpmalı tariflerin kat seçimleri (CV) ve medyanları (final):
  HGB ctx6 kırpma, GRU 20, TCN 15, GRU-emb 7 (kırpma E finallerinde; c2h'de TCN / GRU-emb eski 10 / 10 epoch'lu dosyalar).
- Komutlar (`scripts/`): `shift_fit_d.py`; CV `shift_cv.py hgb --det=ctx [--crop=E]`, `shift_cv.py seq uw_n10_ctx6_cropE|tcn_n10_cropE|
  cnn3_emb_cropE`, `shift_cv.py mix ...` (`worke_*_chain.sh`); finaller `train_n2det_final.py hgbmc --det=ctx --cropE`, `train_gru_n2c.py
  --det=ctx --cropE`, `train_n2det_final.py mix --alpha=0.4 --ens=ens_c_all_ctxE_crop --seq=mix3det_all --gru-from=gru_ctxE_all|gru_ctx_all`
  (c3: seq_comps ("gru",), TCN / GRU-emb anahtarları çıkarılır), `build_hgb_submission.py --name ctxE_c3_540|ctxE_c2h_540 --smooth-a 0`.
  Bütün bileşenler deterministik (`--deterministic` / repro; HGB random_state 0).

### İş G (19 Kasım): HGB tohum ortalaması
- HGB üyeleri (random_state 0 / 1 / 2, `max_features` 0.5 nedeniyle farklı) aynı kırpma E tablosuyla eğitilir, olasılıklar ortalanır. Ön kontrol yalnız
  B üyesinde (kaymalı çift CV, D −0.18, 9/10). ctxE_c3s3_540: 4 üye × 3 tohum. ctxE_c2hs3p_540: süre sınırı nedeniyle yalnız B ve B_h2 × 3 tohum,
  üye ağırlıkları eşit (1/4) tutulur. Diğer bileşenler İş E paketleriyle aynı dosyalar.

### İş I (20 Kasım): mixI_540 — iki mühürlü paketin sabit ağırlıklı karışımı
- M = 0.5 · ctx_c3_540 + 0.5 · ctxE_c2hs3p_540, yeniden eğitim yok. İki paketin karar öncesi metrik ağırlıklı dağılımları ortalanır, sonra mevcut karar
  kuralı (sürekli ağırlıklı medyan). İki paket aynı α (0.4) ve aynı GRU ctx'i paylaştığı için eşdeğer biçim:
  0.4 · (⅔ GRU ctx + ⅙ TCN + ⅙ GRU-emb) + 0.6 · (½ norm(HGB ctx · ω) + ½ norm(HGB E-s3p · ω)); her HGB seti kendi bin sınırlarıyla.
- w = 0.5 önceden sabitlendi (kullanıcı, 19 Kasım); CV yalnız akıl sağlığı için kullanıldı (M hiçbir dağılımda iki bileşeninden de > 0.10 kötü
  değil). Gerekçe: 0.2 s altındaki yerel farklar mühürlüye taşınmadı; karışım yalnız mühürlüde kanıtlanmış iki modelden oluşur.
- Komutlar: `shift_cv.py imix` (`worki_chain.sh`), sarmalayıcıda isteğe bağlı ikinci HGB seti (`hgb2_*` anahtarları) ve `seq_weights`;
  testler `mixI_build_tests.sh`, `mixI_equiv_test.py`.

