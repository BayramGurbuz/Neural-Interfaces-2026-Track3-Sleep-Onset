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
| 5 | 2026-11-16 | cand2_cropGc_540.zip | **34.41** | 34.23 | 94.93 | 175.06 | 1292 s | 12. / 160 |
| 6 | ≤ 2026-11-17 | cand3_gru_540.zip | **33.72** | – | – | – | – | – |
| 7 | 2026-11-18 08:08 | ctx_c2_540.zip | **33.12** | – | – | – | – | – |
| 8 | 2026-11-18 08:08 | ctx_c3_540.zip | **33.09** | – | – | – | – | 16. / 183 |
| 9 | 2026-11-19 | ctxE_c3s3_540.zip | **33.29** | – | – | – | – | – |
| 10 | 2026-11-19 | ctxE_c2hs3p_540.zip | **33.13** | – | – | – | – | – |
| 9 | 2026-11-19 | ctxE_c3s3_540.zip | **33.29** | – | – | – | – | – |
| 10 | 2026-11-19 | ctxE_c2hs3p_540.zip | **33.13** | – | – | – | – | – |

**19 Kasım sonuçları (kullanıcı):** ctxE_c3s3_540 33.29, ctxE_c2hs3p_540 33.13 → en iyi hâlâ ctx_c3_540 (33.09). **12 hakkın 10'u
kullanıldı, 2 kaldı.** Okuma: HGB E mühürlüde nötr (ctx_c2 33.12 ↔ c2hs3p 33.13), GRU E zararlı (c3s3 33.29 ↔ ctx_c3 33.09); kaymalı
çift CV'nin öngördüğü −0.22 (İş E) taşınmadı. Bundan sonra 0.2 s altındaki yerel farklara güvenilmiyor; kararlar mühürlüde kanıtlanmış
modellere dayanır. İş H (GRU E içeriyordu) iptal. Sonraki: İş I (ctx_c3 + ctxE_c2hs3p karışımı, yeniden eğitim yok), yedek İş J.

**16 Kasım mühürlü güncellemesi sonrası skorlar (18 Kasım 14:15, kullanıcı; bütün gönderimler yeniden koşuldu):**
mühürlü kayıtların açılış dakikaları artık modele verilmiyor (FAQ "Sealed evaluation update · November 16"); geçen süre ilk verilen
pencereden. Hak 12 toplam / günde 3; 8 kullanıldı, **4 kaldı**.

| # | ZIP | Mühürlü (13 Kasım skorlaması) | **Mühürlü (16 Kasım, güncel)** |
|---|---|---|---|
| 1 | final540 | 38.74 | **87.99** |
| 2 | mix3_540 | 37.79 | **62.02** |
| 3 | mix3_540_gridpi | 39.20 | **64.96** |
| 4 | mix3_n2det_cropg_540 (aday 1) | 37.09 | **47.42** |
| 5 | cand2_cropGc_540 | – | **34.41** |
| 6 | cand3_gru_540 | – | **33.72** |
| 7 | ctx_c2_540 | – | **33.12** |
| 8 | ctx_c3_540 | – | **33.09** (en iyi, 16. / 183) |

İlk 3 = 31.96 / 32.10 / 32.29; 10. = 32.77. Artırmasız modellerin büyük düşüşü (final540 +49, mix3 +24, aday 1 +10) → mühürlü
akışlar kaydın ortasından başlıyor; aday 2 zaten bu kurguda skorlanmıştı (34.41 değişmedi). Kalibrasyon: İş D.

**Güncel mühürlü skorlar (My Submissions, 15 Kasım; 13 Kasım yeniden skorlamasından sonra):**

| # | ID | ZIP | Yerel çift CV | Mühürlü (eski skorlama) | **Mühürlü (güncel)** | Yerel → güncel mühürlü |
|---|---|---|---|---|---|---|
| 1 | 1046190 | final540.zip | 40.28 | 39.56 | **38.74** | −1.54 |
| 2 | 1047946 | mix3_540.zip | 39.40 | 38.66 | **37.79** | −1.61 |
| 3 | 1050107 | mix3_540_gridpi.zip | 38.27 | 37.49 | **39.20** | +0.93 |
| 4 | 1053877 | mix3_n2det_cropg_540.zip | 38.86 | – | **37.09** | −1.77 |
| 5 | – | cand2_cropGc_540.zip | 38.49 | – | **34.41** | −4.08 |

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

### cand2_cropGc_540.zip (gönderim #5, 2026-11-16; mühürlü 34.41, 12./160) ⁹
- **ZIP SHA-256: 0ad65422284736e630fa2f401c973efd84116bf08c547d62e629e9e7acd63f94** (13.10 MB; submission.py 43 265 B, model.joblib
  21 120 733 B, SHA-256 67fca7fe…c6a1b3d). Yedek: cand2_cropG_540.zip (SHA-256 a23ea00c…48c3ee217).
- Model: HGB(c) 4 üye, 817 öznitelik + çok sınıflı dedektörün yakınlık özetleri (5), kırpma artırmalı eğitim tablosu (`ens_c_all_n2c_crop`,
  iterasyon 200 / 175 / 375 / 200) + GRU-uw 10 tohum, kırpma artırmalı, girdisinde yakınlık özetleri (`gru_n2c_all`, 20 epoch) + TCN 10 tohum
  ve GRU-emb 3 tohum (`mix3det_all`, artırmasız); **α 0.6**; ızgara / kalibrasyon / yumuşatma kapalı. Dedektör: çok sınıflı (5 sınıf),
  3 tohum (epoch 18 / 15 / 12), çıkış = tohum ortalaması yakınlık. **Bütün bileşenler deterministik** (dedektör, GRU, TCN, CNN kodlayıcı,
  GRU-emb: `--deterministic` / repro; HGB random_state 0).
- Çift CV (500 kayıt, ızgarasız, `b2_mix_cand2.json` "HGB(n2c) + seq(cropGc)"): unseen 39.49, seen 37.49 → **38.49**; 39.40'a göre
  −0.91 (8/10), **aday 1'e (38.86) göre −0.37, 7/10** → aday 2 kabulü (Δ ≤ −0.15 ve ≥ 7/10) geçildi. α iç içe LOFO 9/10 katta 0.6.
- Dayanıklılık (train+val modeller, 40 test kaydı): k = 6 karışım zararı **+0.56** (< +1.0 ✓); k = 1 / 3 / 12 / 24 / 60 / rastgele
  −0.84 / −0.68 / +0.77 / −0.42 / +1.53 / +2.81. Aday 1'e en yakın ölçümde k = 6 +3.53, rastgele +14.34.
- **Paket tarifi ↔ 38.49 veren çift CV tarifi:** bileşenler, α 0.6, ω (ızgara ve GRU ω_B; assert), 5 yakınlık özeti (tek kaynak
  `n2det_stream.py`; HGB'de ve GRU girdisinde, kırpmada kırpılan başlangıçtan yeniden akıtılır), kırpma dağılımları (HGB 70/30, GRU 50/35/15),
  tohumlar: aynı. Farklar: (1) 500 → 540 kayıt; (2) kat başına LOFO seçimleri yerine medyanları (dedektör epoch'ları, GRU 20 epoch, HGB
  iterasyonları, 25'e yuvarlı); (3) eğitim satırlarının fold-dışı yakınlığı kişi bazlı iç katlardan; (4) **TCN / CNN kodlayıcı / GRU-emb
  deterministik yeniden eğitildi** — CV'deki "eski" OOF'lar deterministik olmayan özgün koşulardı. Test kayıtlarında TCN bit düzeyinde aynı
  (40.3806), GRU-emb 49.56 → 49.07 (yalnız kodlayıcı değişti).
- Komutlar (`scripts/`): `train_n2det_final.py detmc` → `train_n2det_final.py hgbmc` → `train_gru_n2c.py` →
  `train_mix3_final.py --all --deterministic --tag=det` → `train_n2det_final.py mix --alpha=0.6 --ens=ens_c_all_n2c_crop --seq=mix3det_all
  --gru-from=gru_n2c_all --out=cand2_cropGc_all` → `build_hgb_submission.py --model ~/neuralbench/save/cand2_cropGc_all/model.joblib
  --name cand2_cropGc_540 --smooth-a 0`. CV: `n2det_v2_inner.py 0|1|2`, `ens_crop_cv.py c`, `b2_seq_cv.py n2c gru --crop`, `b2_mix.py`.
- Kontroller (16 Kasım, log `~/neuralbench/save/cand2_build.log`, `cand2_docker.log`):
  - Bileşen kimliği (toplu yol, açık test, örneklem içi): TCN 40.3806, GRU-emb 49.0675, HGB 26.5861 — eğitim loglarıyla aynı; GRU+yakınlık 34.16.
  - Akış ↔ toplu: maks |Δ| 0.041 s, > 0.5 s pencere yok; 31.1905 ↔ 31.1904.
  - Girdi temizliği (`n2det_clean_test.py`, çok sınıflı dedektör dahil) PASS; ek 9 senaryo PASS (tahminler sonlu ve [0, 600] içinde; GRU /
    GRU-emb / TCN durumları, tabanlar, dedektör özetleri ve GRU'nun dedektör tabanında NaN/inf yok).
  - Paketleyici regresyonu: aday 1 yeni paketleyiciyle yeniden üretildi → akış tahminleri gönderilen paketle bit düzeyinde aynı.
  - Docker v3 ×2: wbmae **31.1904923768249** her ikisinde (bit düzeyinde aynı), bMAE 79.79, MAE 150.29; duration 256.4 / 244.3 s; aynı
    oturumda aday 1: 221.8 s (31.883341811607796, 15 Kasım'la aynı). **Oran 1.10–1.16** (kural ≤ 1.50 ✓) → mühürlü süre tahmini
    1527 × 1.10–1.16 ≈ **1680–1770 s** (sınır 3600). Import satırları aday 1 ile aynı.
  - Yedek cand2_cropG_540: Docker 30.390960407877003, 249.5 s.
- **Beklenen mühürlü skor ≈ 36.7–36.9** (yerel 38.49; aday 1 38.86 ↔ 37.09; yerel–mühürlü farkı −1.61 … −1.77).

**Gönderim #5 sonucu (16 Kasım):** W-bMAE makro **34.41**, W-bMAE 34.23, bMAE 94.93, MAE 175.06, süre 1292 s; sıra **12 / 160**; ilk 3 =
32.03 / 32.63 / 32.75, 10. = 33.81. Hak: 7'nin 5'i kullanıldı, **2 kaldı**. Beklenen 36.7–36.9 idi: aday 1 → aday 2 yerelde −0.37,
mühürlüde **−2.68**. Kullanıcı hipotezi: mühürlü akışlar kaydın başından başlamıyor; aday 2'deki HGB kırpma artırması bu yüzden mühürlüde
büyük kazandırdı (gridpi'nin 37.49 → 39.20 düşüşüyle tutarlı). Bundan sonra seçim kaymalı değerlendirmeyle (İş A).

## Aday 3 hazırlığı (17 Kasım): cand3_gru_540 — HAZIR, GÖNDERİLMEDİ (karar kullanıcıda)
- Tarif: İş B satır 5 = aday 2'nin HGB(c) n2c kırpma (4 üye, 540) + yalnız GRU-uw kırpma+yakınlık (10 tohum, 540) + 3 tohumlu çok sınıflı
  dedektör; TCN ve GRU-emb YOK; **α 0.5** (k=0 ve kaymalı çift CV'nin iç içe LOFO seçimi); ızgara kapalı. Bileşenler aday 2'nin 540
  bileşenleriyle aynı dosyalar (hepsi deterministik); yalnız α ve dizi bileşen listesi değişti.
- Seçim ölçütleri (ayrıntı B raporu 17 Kasım): k=0 çift CV 38.67 (aday 2'ye göre +0.18 ≤ +0.3 ✓, 2/10); kaymalı çift CV 9 dağılımda
  +0.07 … +0.11 (3–4/10, aday 2 hepsinde önde); 40 test kaydı (train+val modelleri) İş A dağılımlarında −0.64 … −1.16 (≈ 1 SE, 18–20/40 kayıt).
- Paketleyici: `build_hgb_submission.py` mix3'e isteğe bağlı `seq_comps` (varsayılan 3 bileşen); kullanılmayan bileşen ağırlıkları
  pakete girmiyor. Regresyon: aday 2 yeni paketleyiciyle → akış tahminleri gönderilen paketle **bit düzeyinde aynı**.
- Komutlar: bundle = `cand2_cropGc_all` + mix_alpha 0.5 + seq_comps ("gru",) − TCN/GRU-emb anahtarları → `SAVE/cand3_gru_all`;
  `build_hgb_submission.py --model ~/neuralbench/save/cand3_gru_all/model.joblib --name cand3_gru_540 --smooth-a 0`.
- Kontroller (17 Kasım): bileşen kimliği HGB 26.5861 (eğitim logu ✓), GRU+yakınlık 34.1594 (aday 2 ile aynı); akış ↔ toplu maks |Δ| 0.045 s,
  > 0.5 s pencere yok, 28.9375 ↔ 28.9376 PASS; `n2det_clean_test.py` PASS; `extra_faults.py` 9 senaryo PASS (tahminler sonlu, [0, 600];
  durumlarda NaN/inf yok). Import satırları aday 2 ile aynı.
- Docker v3 ×2 (log `~/neuralbench/save/cand3_docker.log`): wbmae **28.93754041402854** ikisinde (bit düzeyinde aynı), bMAE 77.64, MAE 149.50;
  duration 200.8 / 187.6 s; aynı oturumda aday 1 225.5 s (31.883341811607796, öncekiyle aynı) → **oran 0.83–0.89** (kural ≤ 1.50 ✓).
  Not: GPU'da İş C eğitimi sürüyordu (süreler üst sınır). Yerel akış 27 ms/pencere (aday 2 40 ms).
- ZIP `submissions/cand3_gru_540.zip`, 8.89 MB, SHA-256 `6059fe7099f3cb92e00ecc4f81031e110ad33a7ba9eb22a1a67ce9e0b7de6fc0`.
- Beklenen mühürlü: kaymalı çift CV'ye göre aday 2 ≈ +0.1 (≈ 34.5), 40 teste göre −0.6 … −1.2 (≈ 33.2–33.8); final sıralama takımın en iyi
  mühürlü skoru olduğu için kötü sonuç 34.41'i bozmaz, yalnız bir hak harcar.

## İş C adayları (18 Kasım): ctx_c2_540 ve ctx_c3_540 — HAZIR, GÖNDERİLMEDİ (bileşen seti kullanıcı kararı)
- Ortak: HGB(c) 4 üye kırpma + **bağlamlı dedektör** yakınlık özetleri (ens_c_all_ctx_crop; n_iter 125/175/300/200 = CV medyanları),
  bağlamlı dedektör (son 6 pencere, 3 tohum, epoch 18/12/16; n2det_ctx_final), GRU kırpma + bağlamlı yakınlık (gru_ctx_all, 10 tohum,
  20 epoch). Kırpılmış eğitim akışlarında dedektör yalnız kırpma sonrası pencereleri görüyor (close_oof_L, kırpma-kesin). Hepsi deterministik.
  - **ctx_c2_540** = + eski TCN + eski GRU-emb (mix3det_all; aday 2'nin seti), α 0.5.
  - **ctx_c3_540** = yalnız GRU (cand3'ün seti), α 0.4 (seq_comps ("gru",)).
- Seçim ölçütleri (B raporu 18 Kasım): k=0 çift CV 38.20 / 38.36 (aday 2 38.49); kaymalı çift CV Δ −0.30 … −0.44 / −0.23 … −0.36;
  kaymalı 40 test Δ −0.7 … −2.4 / −1.1 … −2.8. İkisi de kuralı geçiyor.
- Komutlar (`scripts/`): `n2det_ctx_inner.py 6 0|1|2` (CV) → `train_n2det_final.py detctx` → `train_gru_n2c.py --det=ctx` →
  `train_n2det_final.py hgbmc --det=ctx` → `train_n2det_final.py mix --alpha=0.5 --ens=ens_c_all_ctx_crop --seq=mix3det_all
  --gru-from=gru_ctx_all --out=ctx_c2_all` (c3: aynı + α 0.4, seq_comps ("gru",), TCN/GRU-emb anahtarları çıkarıldı → ctx_c3_all) →
  `build_hgb_submission.py --model ~/neuralbench/save/ctx_<c>_all/model.joblib --name ctx_<c>_540 --smooth-a 0`.
- Kontroller (`scripts/ctx_build_tests.sh`, loglar `~/neuralbench/save/ctx_build[2].log`, `ctx_docker.log`):
  - Bileşen kimliği: HGB 25.0725 (CPU dedektörle 25.0722), TCN 40.3806, GRU-emb 49.0675 = eğitim logları; GRU+bağlamlı yakınlık 32.11.
  - Akış ↔ toplu: GPU'da hesaplanmış dedektör referansıyla 4 pencerede > 0.5 s (maks 2.0 / 2.3 s; HGB bölme eşiğinde ~3e-5 GPU/CPU farkı);
    referans dedektör CPU'da eğitim ileri yoluyla hesaplanınca (`--cpu-det`) maks |Δ| **0.027 / 0.030 s**, > 0.5 s yok → PASS (sarmalayıcı CPU'da).
  - `n2det_clean_test.py` PASS, `extra_faults.py` 9 senaryo PASS (bağlam tamponu n2_buf dahil durumlarda NaN/inf yok),
    `ctx_wrapper_test.py` (a) reset PASS, (b) bağlam < 6 akış başlangıçları maks |Δ| 1.2e-7 (CPU) / 2.8e-5 (GPU) PASS. Import satırları aday 2 ile aynı.
  - Docker v3 ×2: ctx_c2 **28.20569984992941** ikisinde, 265.8 / 235.5 s; ctx_c3 **26.705727309192987** ikisinde, 180.9 / 179.6 s;
    aynı oturumda aday 1 204.9 s (31.883341811607796, öncekiyle aynı) → oran ctx_c2 **1.15–1.30**, ctx_c3 **0.88** (kural ≤ 1.50 ✓).
- ZIP: `submissions/ctx_c2_540.zip` 13.25 MB, SHA-256 `e5bc61dfb0140af4d69afd8fe568815fece2f703d24c6b6c129026cff0015ce0`;
  `submissions/ctx_c3_540.zip` 9.05 MB, SHA-256 `416a9dfd35e44748ee40777a8f7060483c83677fb14cce4bcdc41076f17c90ad`.

## İş E adayları (19 Kasım 00:20): ctxE_c3_540 ve ctxE_c2h_540 — HAZIR, GÖNDERİLMEDİ (kullanıcı ikisini 19 Kasım'da birlikte gönderecek)
- **ctxE_c3_540** (ana aday, önceden tanımlı ctx_c3 seti): HGB(c) kırpma E + bağlamlı yakınlık (`ens_c_all_ctxE_crop`, 12 068 özgün + 107 573
  kırpılmış satır, 988 kopya, n_iter 200/200/300/200) + GRU kırpma E + bağlamlı yakınlık (`gru_ctxE_all`), α 0.4.
  Kaymalı çift CV (ctx_c3 tarifine göre): D = U[2,5] dk −0.22 (8/10), D' = U[0,8] dk −0.21 (7/10), U[1,6] −0.23 (7/10) → kural GEÇTİ.
- **ctxE_c2h_540 (HİBRİT — SONRADAN SEÇİLDİ, önceden tanımlı aday DEĞİL):** aynı HGB E + ctx_c2'nin eski dizi seti (GRU ctx `gru_ctx_all`,
  TCN + GRU-emb `mix3det_all`), α 0.4. İş E tablosundaki bilgi satırlarından biri olarak sonuç görüldükten sonra seçildi: D −0.22 (8/10),
  D' −0.31 (8/10), U[1,6] −0.28 (7/10). Önceden tanımlı ctx_c2 seti (hepsi kırpma E) D'de 6/10 ile kuralda kaldı. Amaç: iki dizi setinin
  mühürlüde taşınmasını ölçmek (kullanıcı kararı).
- Paketleme: `train_n2det_final.py mix` (yöntem ctx_c2_all'ın yeniden üretimiyle doğrulandı: bütün anahtarlar aynı) → c3 için seq_comps ("gru",).
  Paketleyici regresyonu: ctx_c3_540 yeni paketleyiciyle (isteğe bağlı seq_weights) → akış tahminleri gönderilen paketle **bit düzeyinde aynı**.
- Kontroller (`scripts/ctxE_build_tests.sh`, loglar `~/neuralbench/save/ctxE_build.log`, `ctxE_docker.log`):
  - Bileşen kimliği (toplu, açık test, örneklem içi, CPU dedektör referansı): HGB E 29.5816 (eğitim logu ✓); c2h TCN 40.3806, GRU-emb 49.0675 ✓;
    GRU E 37.23, GRU ctx 32.11.
  - Akış ↔ toplu: c3 maks |Δ| 0.0139 s, c2h 0.0118 s; > 0.5 s pencere yok; 31.6951 ↔ 31.6951, 30.4092 ↔ 30.4091 → PASS.
  - `n2det_clean_test.py` PASS, `extra_faults.py` 9 senaryo PASS, `ctx_wrapper_test.py` (a) reset (b) bağlam < 6 başlangıçları PASS (ikisinde de).
  - Docker v3 ×2: ctxE_c3 **31.695066336152696** ikisinde, 201.0 / 188.6 s; ctxE_c2h **30.409171256671726** ikisinde, 244.2 / 246.4 s; aynı
    oturumda aday 1 215.3 s (31.883341811607796, öncekiyle aynı) → **oran c3 0.88–0.93, c2h 1.13–1.14** (sınır 1.50 ✓). Import satırları değişmedi.
- ZIP: `submissions/ctxE_c3_540.zip` 8.72 MB, SHA-256 `f853d9fcdc79b9367bd867ee135f351c3c3f1aeec2b8392fee0cc43ceaa5c5b5`;
  `submissions/ctxE_c2h_540.zip` 12.93 MB, SHA-256 `5816a944c841f3318482c7485607bec2255f8cafbd8cd5e6e1c8b6b9449f2bff`.
- Train+val HGB E (`ens_c_tv_ctxE_crop`, açık test örneklem dışı 42.65) ve GRU E / mix3cropE train+val dosyaları kaymalı 40 test kontrolü için hazır.

## İş G (19 Kasım): HGB tohum ortalamalı sürümler — HAZIR, GÖNDERİLMEDİ
- Ön kontrol (`shift_cv.py hgb --det=ctx --crop=E --members=B --seeds=0,1,2`, `seedeval`; yalnız B üyesi, yalnız HGB kararı, kaymalı çift CV):
  3 tohum ortalaması tohum 0'a göre D −0.18 (9/10), D' −0.18 (9/10) → kural (D altında Δ ≤ −0.10, ≥ 7/10) GEÇTİ. Paket düzeyinde etki ÖLÇÜLMEDİ
  (tam CV yok); HGB zaten 4 üye ortalaması ve karışımda 0.6 ağırlıklı → kazanç daha küçük beklenir.
- 540 HGB E, 4 üye × 3 tohum (`train_n2det_final.py hgbmc --det=ctx --cropE --seeds=0,1,2` → `ens_c_all_ctxE_s3_crop`, 4765 s); tohum 0 modelleri ve bin
  sınırları tek tohumlu finalle birebir aynı.
- **ctxE_c3s3_540** = ctxE_c3 + 12 HGB modeli (düz ortalama). HGB 29.4903 (eğitim logu ✓), akış ↔ toplu maks |Δ| 0.0135 s, 31.6827 ↔ 31.6827; temizlik,
  9 arıza, bağlam testleri PASS; Docker ×2 31.6827174743059 ikisinde. Süre: gece oturumu 286.7 / 283.2 s (aday 1 199.9) → 1.42–1.43; aralıklı
  ölçüm (10:26–10:48) 217.2 / 221.3 s (aday 1 166.9 / 151.8 / 151.0) → **1.36–1.46** (sınır 1.50, kenarda). 16.42 MB,
  SHA-256 `1722ad753e24efe291d9411ac1a3e81024d416021d85dde1f4f8d44c9f2ccc64`.
- ctxE_c2hs3_540 (hibrit + 12 HGB modeli): testler PASS ama süre 1.67–1.69 → sınırı aşıyor, KULLANILMAYACAK (SHA-256 97139a9c…bdd49).
- **ctxE_c2hs3p_540** (kullanıcı seçimi): hibrit + yalnız B ve B_h2 × 3 tohum (üye maliyeti B 1.27, B_h1 2.27, B_h2 0.87, B2 2.54 ms/pencere);
  ağırlıklar 1/12 ×3, 1/4, 1/12 ×3, 1/4 (her üye 1/4; sarmalayıcı isteğe bağlı `member_weights`, yoksa eski yol: ctxE_c2h_540 yeniden kurulup akış
  tahminleri bit düzeyinde aynı). HGB 29.5489 (bağımsız eğitim-yolu hesabı ✓), TCN 40.3806, GRU-emb 49.0675 ✓; akış ↔ toplu maks |Δ| 0.0122 s,
  30.4004 ↔ 30.4003; temizlik, 9 arıza, bağlam testleri PASS; Docker 30.400400883297824 (4 koşuda aynı). Süre aralıklı ölçümde 207.2 / 206.9 s →
  **1.30–1.37**. 15.14 MB, SHA-256 `821ae6bd6eab1bcaec9c9a23ac6fe6853e83f0dcdb22ce264526e9dfdeddf1ee`.
- Yerel Docker süreleri oturumlar arasında dalgalı (aday 1: 148–216 s); oranlar aynı oturumdaki komşu aday 1 koşularına göre. Mühürlü süre tahmini
  (aday 1 1526.6 s): c3s3 ≈ 2080–2230 s, c2hs3p ≈ 1990–2090 s (sınır 3600 s).

## Gönderim #9–#10 sonucu (19 Kasım): İş E / İş G taşınmadı
- **ctxE_c3s3_540 33.29, ctxE_c2hs3p_540 33.13**; en iyi hâlâ **ctx_c3_540 33.09**. Hak: 12'nin **10'u kullanıldı, 2 kaldı**. Tek tohumlu ctxE_c3_540 ve
  ctxE_c2h_540 GÖNDERİLMEDİ (kullanıcı tohum ortalamalı sürümleri gönderdi).
- Okuma (kullanıcı): HGB E mühürlüde nötr (ctx_c2 33.12 ↔ ctxE_c2hs3p 33.13; aynı dizi seti, HGB ctx → HGB E + kısmi 3 tohum); GRU E zararlı
  (ctx_c3 33.09 → ctxE_c3s3 33.29: HGB E + GRU E + 3 tohum). Kaymalı CV'nin öngördüğü −0.22 taşınmadı (yerelde c2h −0.22, mühürlüde +0.01 / +0.04).
  **Bundan sonra 0.2 s altındaki yerel farklara güvenilmez; kararlar mühürlüde kanıtlanmış modellere dayanır.** İş H (GRU E içeriyordu) iptal.
- Sonraki: İş I = 0.5 · ctx_c3_540 + 0.5 · ctxE_c2hs3p_540 (yeniden eğitim yok, w sabit); İş J yedek = (ctx_c3 + ctx_c2 + ctxE_c2hs3p) / 3.

## İş I paketi (20 Kasım): mixI_540 — 0.5 · ctx_c3_540 + 0.5 · ctxE_c2hs3p_540 (hazır, gönderilmedi)
- Yeniden eğitim yok; w = 0.5 SABİT (CV'den seçilmedi). Paket = ctxE_c2hs3p_all paketi + ctx_c3_all'ın 4 HGB ctx modeli ikinci HGB seti olarak
  (kendi bin sınırları, kendi ω normalleştirmesi, `hgb2_weight` 0.5) + `seq_weights` [⅔, ⅙, ⅙] (GRU ctx, TCN, GRU-emb); α 0.4. Ortak bileşenler
  (bağlamlı dedektör, öznitelikler, GRU ctx, ızgara, ω, α) iki pakette birebir aynı olduğu doğrulandı ve bir kez çalışır.
- Akıl sağlığı (`shift_cv.py imix`, 3 çekiliş; R1 = HGB E B + B_h2 × 3 tohum, R2 = B_h1 + B2 tohum 0, aynı CV katları):

  | dağılım | ctx_c3 | ctxE_c2hs3p | M | M − c3 | M − c2hs3p | J | M' |
  |---|---|---|---|---|---|---|---|
  | k0 | 38.335 | 38.495 | 38.289 | −0.046 (6/10) | −0.206 (9/10) | 38.256 | 38.292 |
  | D = W24-60 | 35.892 | 35.635 | 35.604 | −0.288 (10/10) | −0.032 (6/10) | 35.650 | 35.610 |
  | D' = W0-96 | 35.660 | 35.357 | 35.373 | −0.287 (9/10) | +0.016 (4/10) | 35.409 | 35.377 |
  | U[1,6] = W12-72 | 35.948 | 35.678 | 35.648 | −0.300 (10/10) | −0.029 (5/10) | 35.704 | 35.654 |

  Kural (M hiçbir dağılımda ikisinden de > 0.10 kötü değil): en kötü Δ **+0.016 → PASS**. M' +0.007 PASS, İş J +0.052 PASS (ctx_c2'ye göre en
  kötü −0.030). Not: M − c3 farkı büyük ölçüde HGB E kaynaklı; HGB E kaymalı CV'de −0.26 öngörülüp mühürlüde +0.04 çıkmıştı → taşınması beklenmez.
- Testler (`scripts/mixI_build_tests.sh`, `scripts/mixI_equiv_test.py`): iki kaynak paketin klasörü == gönderilen ZIP; yeni paketleyiciyle yeniden
  kurulan ctx_c3 / ctxE_c2hs3p akış tahminleri bit düzeyinde aynı; bileşen kimliği GRU 32.1071, TCN 40.3806, GRU-emb 49.0675, HGB E 29.5489,
  HGB ctx 25.0722 (beklenen 25.0725); akış ↔ toplu (CPU dedektör referansı) maks |Δ| 0.0158 s, 28.3004 ↔ 28.3003; eşdeğerlik: karar öncesi
  dağılım ile 0.5 · W(ctx_c3_540) + 0.5 · W(ctxE_c2hs3p_540) arasındaki fark 2.2e-16, tahminler float32'ye yuvarlanmış referansla bütün
  pencerelerde aynı (ilk sürümün FAIL'i yalnız sarmalayıcının float32 çıktısından, ölçüt düzeltildi); temizlik, 9 arıza, bağlam testleri PASS;
  Docker ×2 28.300361165769242 ikisinde.
- Süre (20 Kasım 10:07–10:31, boş makine, aralıklı): aday 1 241 / 211 / 205 s, mixI 413 / 345 s → komşu ortalamaya göre oran 1.83 / 1.66
  (tek komşulara göre 1.64–1.96) → mühürlü tahmin ≈ 1526.6 × 1.66–1.83 ≈ **2530–2790 s** (en kötü ≈ 2990 s). Kullanıcı kuralı ≤ 3000 s ✓;
  sınır 3600 s. Geçmiş: bir ≤ 2500 s tahmini gerçekte 2762.79 s çıkmıştı (~+%10).
- **mixI_540.zip** 20.24 MB, SHA-256 `5ece144bfffd28c67e1b554b2ce7c48d27958525ac817732914c9b16cb2c1a5a`.

