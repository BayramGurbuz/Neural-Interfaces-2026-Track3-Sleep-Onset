# Gönderim günlüğü (tekrarlanabilirlik)

Her ZIP için: nasıl üretildi (eğitim + paketleme commit'i, komutlar), seed'ler, çift CV, açık test,
yerel Docker süresi; sealed sütunları gönderim yapıldıkça doldurulur.
Çift CV = (görülmemiş kişi 5-kat + görülmüş kişi 5-kat) / 2, train+val (500 kayıt) üzerinde;
açık test = 40 kayıt (yalnız son kontrol). Docker = lionelyneuro/neural-compet:v3, CPU,
`scripts/docker_test_submission.sh`. Ağırlıklar ve ZIP'ler git'te değil (submissions/, ~/neuralbench/save/).

## Özet

Sealed pencere: 10 Kasım'da açıldı (organizatör maili), bitiş 21 Kasım 23:59 CET = 22 Kasım 01:59 TR.
5 hak takım başına ortak, günde en fazla 1; en iyi geçerli skor sayılır (tekrarlanabilirlik denetimine bağlı).
Sealed log/tahmin/skorlama çıktıları gizli. Plan: #1 final540 (10 Kasım, yapıldı), #2 mix3_540 (11 Kasım, yapıldı),
#3 mix3_540_gridpi (12 Kasım, yapıldı; eski skorlama 37.49, ızgara 13 Kasım kuralıyla yasak), #4–#7 yalnızca kabul kuralını geçen modeller (mix_gru10_hgb
plandan çıkarıldı). **7 hakkın 3'ü kullanıldı, kalan 4** (13 Kasım teknik güncellemesi: +2 hak).
Son gönderim en geç 20 Kasım, 21 Kasım yedek gün.

| # | Gönderim tarihi | ZIP | Durum | Çift CV | Açık test | Docker süresi | Sealed skor | Sealed süre |
|---|---|---|---|---|---|---|---|---|
| – | (gönderilmedi) | hgb_v2_s0.zip | yedek | 46.42 | 54.9426 | 141 s | – | – |
| – | (gönderilmedi) | hgb_v3_np.zip | yedek | 43.30 | 52.4791 | 89 s ¹ | – | – |
| – | – | hgb_ens_c.zip | hazır aday | 42.50 | 52.8573 | 115 s ¹ | – | – |
| – | – | mix_gru_hgb.zip | yedek | 40.43 | 48.2226 | 123 s | _bekleniyor_ | _bekleniyor_ |
| – | (gönderilmeyecek) | mix_gru10_hgb.zip | plandan çıkarıldı ⁴ | 40.28 | 46.9131 | 132 s | – | – |
| 1 | 2026-11-10 | final540.zip | **gönderildi** | 40.28 ² | 31.5337 ³ | 122 s | **39.56** (23/60) | 1249.51 s |
| 2 | 2026-11-11 | mix3_540.zip | **gönderildi** ⁵ | 39.40 ² | 32.7449 ³ | 225 s / 213 s (×2, aynı skor) | **38.66** | kaydedilmedi |

| – | (onay bekliyor) | mix3_540_grid.zip | **aday** ⁶ | 38.35 ² | 31.6917 ³ | 230 s / 218 s (×2, aynı skor) | – | – |

| 3 | 2026-11-12 | mix3_540_gridpi.zip | **gönderildi** ⁷ | 38.27 ² | 31.3234 ³ | 185 s / 176 s (×2, aynı skor) | **37.49** | 2762.79 s |
| 4 | 2026-11-15 | mix3_n2det_cropg_540.zip | **gönderildi** (ID 1053877) ⁸ | 38.86 ² | 31.8833 ³ | 248 s / 220 s (×2, aynı skor; aynı oturumda mix3_540 229 s) | **37.09** (27.) | 1526.6 s |
² 500 kayıtlık çift CV (540'ta doğrulama seti yok). ³ Örneklem içi (test eğitimde) → yalnız boru hattı kontrolü.
⁴ final540 ile tek fark eğitim kayıt sayısı (submission.py birebir aynı; model.joblib'de α, GRU ayarı, seed, epoch, ızgara
  aynı; yalnız veriye bağlı binleme sınırları ve ağaç yapıları farklı) → ayrı bilgi vermez (2026-11-11).
¹ Arka planda CV koşusu çalışırken ölçüldü (yükte). sklearn yolu (hgb_v3.zip, aynı model/skor): 417 s.
⁵ mix3 = (GRU-uw 10 seed + TCN 10 seed + GRU-emb 3 seed)/3 metrik-ağırlıklı grid dağılımı, HGB(c) ile α 0.7;
  kabul kuralı: çift CV 39.40, mix_gru10_hgb'ye göre −0.88, 10/10 kat (`scripts/mix_seq.py`). Eğitim:
  `scripts/train_mix3_final.py --all` (540 kayıt; GRU 10×15 ep, TCN 10×10 ep, CNN kodlayıcı 3×12 ep, GRU-emb
  3×10 ep; HGB = ens_c_all) → `~/neuralbench/save/mix3_all/`; paket: `scripts/build_hgb_submission.py
  --model ~/neuralbench/save/mix3_all/model.joblib --name mix3_540` (12.9 MB). Akış ↔ toplu eşdeğerlik
  `scripts/check_stream_mix3.py`. Train+val sürümü (`mix3/`, mix3_tv.zip): açık test 47.0597.
  Docker v3 iki koşu: wbmae 32.74462474110901 her ikisinde de (bit düzeyinde aynı → deterministik), bMAE 81.76,
  MAE 151.95; duration 225.3 s / 213.1 s (final540: 122 s) → oran ≈ 1.75–1.85 → sunucu tahmini ≈ 2200–2300 s
  (sınır 3600 s, ~%37 pay). Pencere başına ≈ 36 ms (final540 ≈ 21 ms).

## Sealed sonuçlar
| # | Tarih | ZIP | W-bMAE macro (sıralama) | W-bMAE | bMAE | MAE | Duration | Sıra |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-10 | final540.zip | **39.56** | 39.91 | 101.36 | 198.26 | 1249.51 s | 23 / 60 |
| 2 | 2026-11-11 | mix3_540.zip | **38.66** | kaydedilmedi | kaydedilmedi | kaydedilmedi | kaydedilmedi | kaydedilmedi |
| 3 | 2026-11-12 | mix3_540_gridpi.zip | **37.49** | 37.78 | 105.48 | 213.98 | 2762.79 s | kaydedilmedi |
| 4 | 2026-11-15 | mix3_n2det_cropg_540.zip | **37.09** (güncel skorlama) | 37.33 | 92.96 | 181.64 | 1526.6 s | 27. |

**Güncel mühürlü skorlar (My Submissions, 15 Kasım; 13 Kasım yeniden skorlamasından sonra):**

| # | ID | ZIP | Yerel çift CV | Mühürlü (eski skorlama) | **Mühürlü (güncel)** | Yerel → güncel mühürlü |
|---|---|---|---|---|---|---|
| 1 | 1046190 | final540.zip | 40.28 | 39.56 | **38.74** | −1.54 |
| 2 | 1047946 | mix3_540.zip | 39.40 | 38.66 | **37.79** | −1.61 |
| 3 | 1050107 | mix3_540_gridpi.zip | 38.27 | 37.49 | **39.20** | +0.93 |
| 4 | 1053877 | mix3_n2det_cropg_540.zip | 38.86 | – | **37.09** | −1.77 |

Hak: 7'nin 4'ü kullanıldı, **3 kaldı** (15 Kasım'ın hakkı kullanıldı). Sıralama 15 Kasım: **27.** (37.09); ilk 3 = 34.63 / 35.05 / 35.37
(3.lüğe fark 1.72 s). mix3_540 13 Kasım'da 37.79 ile 17./123 idi → leaderboard iki günde belirgin ilerledi.

Okuma:
- **Aday 1 = 37.09**, beklenen ≈ 37.25 → kullanıcının ölçütüyle "≤ 37.3: dedektör + kırpma taşınıyor, aday 2 aynı yolda". Yerel–mühürlü
  farkı −1.77 (öncekiler −1.54 / −1.61); fazladan −0.16 s gürültü sınırında, ama yön olumlu.
- **gridpi 37.49 → 39.20:** yeniden skorlamada ızgarasız modeller ~0.85 s iyileşti (final540 −0.82, mix3_540 −0.87), ızgaralı model 1.71 s
  kötüleşti ve artık mix3_540'tan 1.41 s kötü. Çift CV'de ızgara dışı kayıtlarda gridpi'nin zararı ≈ +0.58 s idi; mühürlüdeki +1.41 daha
  büyük → mühürlü akışlarda 30 sn hizası artık YOK (onset'ler ızgara dışı ve/veya akış başlangıcı kaydın başına hizalı değil). Izgara
  yasağı ampirik olarak da doğrulandı.
- Büyük akış-başlangıcı kayması işareti yok: artırmasız mix3_540'ın yerel–mühürlü farkı final540 ile aynı (−1.61 ↔ −1.54); rastgele
  başlangıç olsaydı artırmasız modeller +30 s mertebesinde bozulurdu. Kırpma artırması bu yüzden şimdilik sigorta; küçük kaymalar
  (birkaç pencere) gridpi'deki fazladan zararla tutarlı ama ayırt edilemiyor.

**13 Kasım yeniden skorlama (yukarıdaki skorlar eski skorlamaya ait, GEÇERSİZ):** mühürlü değerlendirme teknik olarak
güncellendi, bütün Finished gönderimler aynı ID ile yeniden koşuldu. mix3_540: **37.79** (17. / 123; 3.lük 36.52), süre 1722 s.
final540 ve gridpi'nin yeni skorları bu dosyaya henüz işlenmedi. 30 sn ızgarası yeni FAQ'ya göre yasak; gridpi tarifi
terk edildi. Hak 5 → 7 (3 kullanıldı, 4 kaldı).

mix3_540_gridpi: yerel çift CV 38.27 → sealed 37.49, fark −0.78 (final540 −0.72, mix3_540 −0.74) → 30 s ızgara
kararı mühürlü veride tutuyor; mix3_540'a göre sealed −1.17. Süre: gönderim öncesi tahmin ≤ 2500 s, gerçek 2762.79 s
(sınır 3600 s). Kural (14 Kasım): yeni adaylarda pencere başına çıkarım maliyeti gridpi'ye göre en fazla +%10.

mix3_540: final540'a göre sealed −0.90 (çift CV'de beklenen −0.88); çift CV ↔ sealed farkı −0.74 (final540'ta −0.72).
İlk 3 eşiği ≈ 33.06 → fark 5.60 s. Kalan hak: 3. Bilinen kusur (gönderilen ZIP'te): NaN içeren bir pencere kaydın
kalan tahminlerini 0'a kilitliyor (`DURUM_RAPORU.md` §10.1); düzeltme `scripts/build_hgb_submission.py`'de.

Leaderboard bağlamı (2026-11-11): ilk 3 = 33.06 / 34.62 / 35.26 (bMAE 80.03 / 87.72 / 96.64); 10. sıra 37.22.
**Çift CV tahmini sealed ile tutarlı:** mix_gru10_hgb OOF (aynı tarif, 500 kayıt) makro W-bMAE 40.28,
bMAE 100.06, MAE 188.97 (`scripts/range_breakdown.py`, pencere havuzlu tanım) ↔ sealed 39.56 / 101.36 / 198.26.
Duration: yerel Docker CPU 122 s (40 kayıt) ↔ sealed 1249.51 s (sealed set daha büyük ve/veya donanım/kuyruk farkı);
60 dk sınırının ~%35'i.

## Ayrıntılar

### hgb_v2_s0.zip
- Model: HistGradientBoostingRegressor (absolute_error, metrik pencere ağırlıkları), config c5
  (lr 0.02, max_leaf_nodes 15, min_samples_leaf 1000, l2 10, max_features 0.5), 250 ağaç;
  321 nedensel öznitelik (taban + EMA + ilk-2-dk farkı + elapsed); ön-binleme; gömülü izotonik
  kalibrasyon (c5 OOF, 20 bin); yumuşatma a = 0.
- Eğitim verisi: train+val (500 kayıt). Seed: random_state = 0.
- Eğitim commit'i: 5e485d6 (`scripts/train_v2.py`), paketleme aynı commit'teki builder.
- Komutlar: `python scripts/extract_features.py` → `python scripts/train_v2.py` →
  `python scripts/build_hgb_submission.py --model ~/neuralbench/save/hgb_v2/model.joblib --name hgb_v2_s0 --smooth-a 0`
- Çift CV (c5 + kalibrasyon): unseen 47.27 ± 2.50, seen 45.57 ± 1.94 → 46.42.
- Açık test 54.9426; Docker 54.9426, 141 s.

### hgb_v3_np.zip
- Model: dağılım modeli B — HistGradientBoostingClassifier, 11 sınıf
  [0,10,20,30,40,60,90,150,225,300,450,600], lr 0.05, max_leaf_nodes 15, min_samples_leaf 500,
  l2 5, max_features 0.5, 300 iterasyon; karar = P·ω ile sürekli ağırlıklı medyan; gömülü izotonik
  kalibrasyon (B+dyn unseen OOF, 20 bin). 517 öznitelik (321 + "dyn" grubu). Inference: ağaçlar
  numpy dizilerine çevrilmiş (`scripts/tree_numpy.py`), sklearn ile maks. |ΔP| = 2.6e-15.
- Eğitim verisi: train+val (500 kayıt). Seed: random_state = 0.
- Eğitim commit'i: 1a4269c (`scripts/train_v3.py`); paketleme: c94ce11 (numpy ağaç yolu).
- Komutlar: `python scripts/extract_features.py` → `python scripts/ablation_c.py B dyn` (kalibrasyon
  için OOF) → `python scripts/train_v3.py` →
  `python scripts/build_hgb_submission.py --model ~/neuralbench/save/hgb_v3/model.joblib --name hgb_v3_np --smooth-a 0`
- Çift CV (B+dyn + kalibrasyon): unseen 44.80 ± 2.02, seen 41.79 ± 1.31 → 43.30;
  c5 + kalibrasyondan 10/10 katta daha iyi.
- Açık test 52.4791; Docker 52.4791, 89 s (pencere başına ~15 ms, 5869 pencere).

### hgb_ens_c.zip (hazır aday)
- Model: 4 dağılım sınıflandırıcısının ortalaması, ortak ızgarada (B sınırları ∪ B2 sınırları + 600'de
  nokta sınıfı): B (lr 0.05, 15 yaprak, msl 500, l2 5; 225 iter), B_h1 (lr 0.03, 31 yaprak, msl 300,
  l2 2; 200 iter), B_h2 (lr 0.05, 7 yaprak, msl 1000, l2 10; 550 iter), B2 (30 s dilimler, B
  parametreleri; 200 iter). İterasyonlar = her üyenin 10 çift-CV LOFO seçiminin medyanı.
  817 öznitelik (taban + dyn + maxmin + spec + eye). Karar: ızgara ω ile sürekli ağırlıklı medyan;
  gömülü izotonik kalibrasyon ((c) unseen OOF, 20 bin). Inference numpy ağaçları.
- Eğitim verisi: train+val (500 kayıt). Seed: random_state = 0 (tüm üyeler).
- Eğitim commit'i: 36adb04 (`scripts/train_ens_c.py`, ens_grid builder modu); CV: `scripts/ensemble_dist.py
  dyn_maxmin_spec_eye` (87c9478).
- Komutlar: `python scripts/extract_features.py` → `python scripts/ensemble_dist.py dyn_maxmin_spec_eye` →
  `python scripts/train_ens_c.py` →
  `python scripts/build_hgb_submission.py --model ~/neuralbench/save/ens_c/model.joblib --name hgb_ens_c --smooth-a 0`
- Çift CV (+ kalibrasyon): unseen 44.15 ± 2.32, seen 40.86 ± 1.50 → 42.50; tek B'den 10/10 katta iyi.
- Açık test 52.8573 (kalibrasyonsuz 52.18); Docker 52.8573, 115 s (GRU eğitimi arka planda çalışırken).
  meta ch_names = ['TP9', 'AF7', 'AF8', 'TP10'].

### mix_gru_hgb.zip (yedek)
- Model: nedensel GRU (config `uw`: Linear(125→64)+GELU+LayerNorm → GRU 64×2, dropout 0.2;
  ağırlıksız CE, AdamW lr 2e-3, wd 1e-2, cosine T_max 150, **10 epoch'ta durduruldu**; 3 seed 0/1/2
  ortalaması; karar tarafında q·ω_B) + HGB ensemble (c) (hgb_ens_c ile aynı 4 üye, kalibrasyonsuz),
  ortak ızgarada metrik-ağırlıklı dağılımlar **α = 0.5** ile karıştırılıyor; kalibrasyon yok.
  GRU girdileri: 62 pencere özniteliği + ilk-2-dk farkları + log1p(elapsed, ≤ 1500 s), train+val
  ortalama/std ile standardize (gömülü). Epoch = çift-CV ortalama eğrisinin argmin'i (10; 15 ile eşit),
  α = 10 LOFO seçiminin medyanı.
- Eğitim verisi: train+val (500 kayıt). Seed'ler: GRU 0, 1, 2 (torch + numpy); HGB random_state 0.
  Not: GRU GPU'da (cuDNN) eğitildi; yeniden eğitimde ağırlıklar bit düzeyinde aynı olmayabilir.
- Eğitim commit'i: 034eb24 (`scripts/gru_cv.py`, `scripts/mix_gru_hgb.py`,
  `scripts/train_mix_final.py`, builder `mix` modu).
- Komutlar: `python scripts/gru_cv.py w uw` → `python scripts/mix_gru_hgb.py uw` →
  `python scripts/train_ens_c.py` → `python scripts/train_mix_final.py uw 10 0.5` →
  `python scripts/build_hgb_submission.py --model ~/neuralbench/save/mix_uw/model.joblib --name mix_gru_hgb --smooth-a 0`
- Çift CV (kalibrasyonsuz): unseen 41.81 ± 1.54, seen 39.06 ± 1.13 → 40.43; HGB(c)'den 10/10,
  yalnız GRU'dan 9/10 katta iyi.
- Açık test 48.2224 (yerel) / 48.2226 (Docker); Docker 123 s (yüksüz), pencere başına ~21 ms.

### mix_gru10_hgb.zip (gönderilmeyecek; final540 ile aynı tarif, 500 kayıt)
- mix_gru_hgb ile aynı, farklar: GRU **10 seed (0–9)**, **15 epoch** (10 LOFO seçiminin medyanı,
  `scripts/seeds_cv.py`), **α = 0.6** (`mix_gru_hgb.py uw_n10`, LOFO medyanı).
- Komut: `python scripts/train_mix_final.py uw 15 0.6 0,1,2,3,4,5,6,7,8,9 _n10` →
  `python scripts/build_hgb_submission.py --model ~/neuralbench/save/mix_uw_n10/model.joblib --name mix_gru10_hgb --smooth-a 0`
- Çift CV (kalibrasyonsuz): unseen 41.80 ± 1.91, seen 38.76 ± 1.41 → 40.28; 3 seed'li karışımdan 7/10 katta iyi.
- Açık test 46.9128 (yerel) / 46.9131 (Docker); Docker 132 s.

### final540.zip (gönderim #1, 2026-11-10, sealed 39.56)
- Model: mix_gru10_hgb ile birebir aynı tarif (GRU uw 10 seed 0–9, 15 epoch, α 0.6; HGB(c) 4 üye
  225/200/550/200 iterasyon, random_state 0; 817 öznitelik; kalibrasyon yok), eğitim verisi **540 kayıt**
  (train+val+test, 80 975 pencere). Ön-binleme sınırları, ızgara ω, ω_B, GRU ortalama/std 540'tan.
- Eğitim/paketleme commit'i: bu girişle aynı commit (`--all` bayrakları, `scripts/compare_final.py`).
- Komutlar: `python scripts/train_ens_c.py --all` →
  `python scripts/train_mix_final.py uw 15 0.6 0,1,2,3,4,5,6,7,8,9 _n10 --all` →
  `python scripts/build_hgb_submission.py --model ~/neuralbench/save/mix_uw_n10_all/model.joblib --name final540 --smooth-a 0`
- Süreler: HGB 4 üye 1822 s (tekrar 1579 s), GRU 10 seed 87 s (tekrar 39 s; GPU).
- Kontroller (`python scripts/compare_final.py mix_uw_n10 mix_uw_n10_all mix_uw_n10_all_rep`):
  - Tekrarlanabilirlik: aynı seed'lerle ikinci 540 koşusu (HGB + GRU) **bit düzeyinde aynı** (GRU
    ağırlıkları, binleme, μ/σ, HGB ve GRU olasılıkları, tahminler: maks. fark 0.0).
  - 500 ↔ 540 (train+val pencereleri, ikisi de örneklem içi): ort. |Δ| 15.0 s, p95 56.9 s, r 0.980;
    TV(GRU) 0.062, TV(HGB) 0.067. Referans: aynı veri, yalnız GRU seed üçlüsü farklı (mix_uw ↔ mix_uw_s345)
    ort. |Δ| 10.4 s, r 0.993, TV(GRU) 0.080 → 540'ın GRU değişimi seed gürültüsü mertebesinde.
    Örneklem içi train+val W-bMAE: 500-model 27.99, 540-model 29.27 (aynı iterasyon, daha çok veri).
  - Test (500 için örneklem dışı, 540 için içi): 46.91 → 31.53 (GRU 39.86, HGB 26.00); ort. |Δ| 37.2 s.
- Docker 31.5337 (yerel 31.5338), 122 s (yüksüz).

## Tekrarlanabilirlik ölçümü (2026-11-10, mix_gru_hgb ayarları, açık test)
Aynı seed'lerle (0,1,2) yeniden eğitim: bit düzeyinde aynı (48.2224). Farklı seed üçlüleri:
0,1,2 → 48.22; 3,4,5 → 48.70; 6,7,8 → 47.90; 9,10,11 → 47.58 ⇒ karışım 48.10 ± 0.48
(yalnız GRU 48.54 ± 0.81).

### mix3_540_grid.zip (aday, 2026-11-12; gönderilmedi) ⁶
- Model: mix3_540 ile **aynı model.joblib** (SHA-256 265bb6b2…55d32, yeniden eğitim yok). Fark yalnız submission.py:
  (1) 30 sn ızgara kararı, π = 0.86: her bileşenin dağılımı 1 s hücrelere inceltilir, kütlenin π kadarı "onset kayıt
  başından itibaren 30 s'nin katı" olan en yakın kalan süreye taşınır, aynı sınıf ağırlıkları (ω) ve α 0.7 ile sürekli
  medyan; (2) girdi temizliği `_clean` (NaN/inf; kanalın > %50'si geçersizse o kanalın son temiz penceresi, değilse son
  geçerli örnek). Pencere bitişi 5 s'nin katı değilse eski karara düşer.
- Seçim: `scripts/decision_layer.py` (iç içe LOFO, resmi binned_mae); çift CV unseen 39.74, seen 36.96 → 38.35;
  mix3'e (39.40) göre −1.05, 10/10 kat. π tüm katlar üzerinden 0.86. Açık test (500-kayıt modeli) 47.06 → 46.61.
- Komut: `python scripts/build_hgb_submission.py --model ~/neuralbench/save/mix3_all/model.joblib --name mix3_540_grid --grid-pi 0.86`
  (zip 13.23 MB, SHA-256 f4e60d85…7dab0).
- Kontroller: `scripts/check_stream_grid.py submissions/mix3_540_grid mix3_all` → akış ↔ toplu maks |Δ| 0.0009 s,
  31.6917 ↔ 31.6917; Docker v3 ×2: wbmae 31.691708741918738 (her ikisinde), bMAE 81.35, MAE 151.66, duration
  230.4 / 218.1 s (aynı oturumda mix3_540: 210.7 s); pencere başına 31.5 ms. Temiz veride NaN düzeltmesinin etkisi 0.0.
  Log: `~/neuralbench/save/mix3_540_grid_build.log`. v4-sealed testi yapılamadı (imaj yok).
- Risk: kazanç sealed onset'lerinin 30 s ızgarasında olmasına bağlı; ızgara dışı kayıtlarda çift CV'de +1.11 s.
  Ayrıntı: `KARAR_KATMANI_RAPORU.md`.

### mix3_540_gridpi.zip (gönderim #3, 2026-11-12; sealed 37.49) ⁷
- Model: mix3_540 ile aynı model.joblib (SHA-256 265bb6b2…55d32). submission.py: ızgara kararı π(D) (kesimler 900 / 1100 /
  1300 s, değerler 540 kayıttan 1.0 / 0.592 / 0.644 / 0.444; `scripts/b1_grid_pi.py`), girdi temizliği NaN/inf + düz kanal
  (std < 0.1 µV veya ardışık örneklerin > %90'ı değişmiyor).
- Çift CV (500 kayıt, B1a bins4): unseen 39.66, seen 36.89 → 38.27; mix3_540_grid tarifine göre −0.08, 10/10 kat;
  ızgara dışı kayıtlarda zarar +1.11 → +0.58 s. Açık test (500 modeli) 46.61 → 46.39.
- Komut: `python scripts/build_hgb_submission.py --model ~/neuralbench/save/mix3_all/model.joblib --name mix3_540_gridpi
  --grid-pi-cuts 900,1100,1300 --grid-pi-values 1.0,0.592,0.644,0.444` (zip 13.23 MB, SHA-256 4ef85c1f…b4b700063).
- Kontroller: akış ↔ toplu maks |Δ| 0.0002 s, 31.3234 ↔ 31.3234; Docker v3 ×2: 31.323388735064913 (her ikisinde),
  duration 184.9 / 176.1 s (aynı oturumda mix3_540_grid 168.5 s); temiz veride temizlik kuralının etkisi 0.0.
  Log `~/neuralbench/save/mix3_540_gridpi_build.log`. v4-sealed testi yapılamadı. Ayrıntı: `B_ASAMASI_RAPORU.md`.

### mix3_n2det_cropg_540.zip (gönderim #4, 2026-11-15 19:46, ID 1053877; mühürlü 37.09, 27.) ⁸
- **ZIP SHA-256: aed6a9a61ca0003282eb123570f74e4ca718a575abdaee60dde63371899802f9** (12.01 MB; submission.py 42 446 B,
  model.joblib 18 461 341 B, SHA-256 a85f9691…b9863340).
- Model: HGB(c) 4 üye + 5 N2-dedektör özeti (822 öznitelik; `ens_c_all_n2det`, iterasyon 150 / 150 / 250 / 150) + GRU-uw 10 tohum
  **başlangıç kırpma artırmalı, deterministik, 20 epoch** (`mix3crop_all`'dan yalnız GRU) + TCN 10 tohum ve GRU-emb 3 tohum
  (`mix3_all`, mix3_540 ile aynı ağırlıklar); **α 0.6**, ızgara / kalibrasyon / yumuşatma kapalı; girdi temizliği NaN/inf + düz kanal
  (gridpi ile aynı kural; dedektör temizlenmiş pencereyi görür). Dedektör özetleri yalnız HGB girdisinde (`n2det_into = ()`).
- Çift CV (500 kayıt, ızgarasız, `b2_mix_cropmix.json` "HGB(n) + seq(cropG)"): unseen 39.93, seen 37.79 → **38.86**; referans
  ızgarasız mix3 (39.40) göre **−0.54, 8/10 kat** → kabul kuralı (Δ ≤ −0.5 ve ≥ 8/10) geçildi. α iç içe LOFO 10/10 katta 0.6.
- **Paket tarifi ↔ 38.86 veren çift CV tarifi:** bileşenler, α (0.6), sınıf ağırlıkları ω (ızgara ω ve GRU ω_B; 540'tan yeniden
  hesaplanan değerler iki kaynak pakette de aynı, assert), 5 dedektör özeti (tek kaynak `n2det_stream.py`), dedektörün yalnız HGB'ye
  girmesi, GRU 20 epoch / TCN 10 / kodlayıcı 12 / GRU-emb 10, tohumlar: **aynı**. Farklar yalnız 540'a geçişin olağan sonuçları:
  (1) eğitim 500 → 540 kayıt; (2) kat başına LOFO seçimleri yerine medyanları (GRU epoch 20; dedektör epoch 13; HGB iterasyon
  150/150/250/150, 25'e yuvarlı); (3) HGB eğitim satırlarının fold-dışı dedektör P'si için iç katlar kişi bazlı (CV'de görülmüş kişi
  protokolünde kayıt bazlıydı). TCN / kodlayıcı / GRU-emb: CV'deki "eski" OOF'lar da deterministik olmayan özgün koşulardı, paket de öyle.
- Komutlar: `docs/method.md` "Aday 1" bölümü (det → hgb → `train_mix3_final.py --all --deterministic --crop
  --epochs=gru:20,tcn:15,gemb:10 --tag=crop` → `train_n2det_final.py mix --alpha=0.6 --seq=mix3_all --gru-from=mix3crop_all
  --out=mix3_n2det_cropg_all` → `build_hgb_submission.py --model …/mix3_n2det_cropg_all/model.joblib --name mix3_n2det_cropg_540`).
- Kontroller (15 Kasım, log `~/neuralbench/save/mix3_n2det_cropg_540_build.log`):
  - Bileşen kimliği (`scripts/check_stream_n2mix.py`, toplu yol, açık test, örneklem içi): GRU 37.4522, TCN 40.3806, GRU-emb 49.5618,
    HGB+n2det 26.7506 — dördü de ilgili eğitim logundaki değerle aynı.
  - Akış ↔ toplu: maks |Δ| 0.0357 s, ortalama 0.003 s, > 0.5 s olan pencere yok; W-bMAE 31.8833 ↔ 31.8835.
  - Girdi temizliği (`scripts/n2det_clean_test.py`, 3 en uzun test kaydı; NaN kanal, %30 NaN, 11 pencere düz kanal, +inf): tahminler
    ve dedektör özetleri sonlu; temiz sinyalde özetler toplu N2Summary ile aynı (1e-5); P / EMA30 / artış arızadan 5 dk sonra temiz
    koşuya dönüyor → PASS. Ek 9 senaryo (ilk 5 pencere NaN, ilk pencere +inf, baştan düz kanal, tüm kayıt NaN, tüm kayıt düz, bir kanal
    hep NaN, her 2. örnek NaN, 1e30, 10 pencere −inf): bütün tahminler sonlu ve [0, 600] içinde; GRU / GRU-emb / TCN durumları, tabanlar,
    dedektör EMA'ları ve öznitelik akışı durumunda NaN/inf yok.
    Not: arızadan 5 dk sonra tahmin farkı sıfıra inmiyor (ort. 0.9–1.6 s, maks 7–12 s); dedektörsüz mix3_540_gridpi'de de aynı
    (ort. 0.4–1.5 s) → uzun bellekli özniteliklerin (ilk-2-dk tabanı, koşan ekstremumlar) doğal sonucu, dedektöre özgü değil.
  - Docker v3 ×2: wbmae 31.883341811607796 her ikisinde (bit düzeyinde aynı), bMAE 76.98, MAE 143.14; duration **248.2 / 220.2 s**;
    aynı oturumda mix3_540: 229.3 s (32.74462474110901, 11 Kasım'la aynı). Oran 0.96 (ikinci koşu) – 1.08 (ilk koşu, başlangıçta yerel
    testlerin yükü vardı; üst sınır) → kural (≤ +%10) sağlanıyor. Mühürlü süre tahmini ≈ 1722 × 0.96–1.08 ≈ **1650–1860 s** (sınır 3600).
    Dedektör pencere başına maliyet ekliyor ama HGB ağaç sayısı azaldı (1175 → 700 iterasyon).
  - Import satırları final540 / mix3_540 ile birebir aynı (joblib, numpy, torch, benchmark_utils, scipy.signal find_peaks + welch).
  - v4-sealed testi yapılamadı (imaj herkese açık değil; yerelde yalnız v3).
- **Beklenen mühürlü skor ≈ 37.25** (yerel 38.86; ref 39.40 ↔ 37.79). Okuma: ≈ 37.25 → dedektör kazancı taşınıyor, mühürlü akışlarda
  kayma yok; belirgin daha iyi → kayma var ya da dedektör beklenenden iyi (sonraki adım aynı: tam artırmalı aday 2); ≈ 37.8 veya kötü →
  dedektör yeni mühürlü sete taşınmıyor, aday 2 yeniden düşünülür.
- Mühürlü: **W-bMAE makro 37.09**, W-bMAE 37.33, bMAE 92.96, MAE 181.64, süre **1526.6 s** (sınır 3600; mix3_540 1722 s → oran 0.89,
  yerel Docker oranı 0.96–1.08 ile tutarlı). Yerel 38.86 → −1.77 (mix3_540 −1.61); beklenen ≈ 37.25. Eski gönderimlerin güncel skorları
  yukarıda "Güncel mühürlü skorlar".
