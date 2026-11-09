# Gönderim günlüğü (tekrarlanabilirlik)

Her ZIP için: nasıl üretildi (eğitim + paketleme commit'i, komutlar), seed'ler, çift CV, açık test,
yerel Docker süresi; sealed sütunları gönderim yapıldıkça doldurulur.
Çift CV = (görülmemiş kişi 5-kat + görülmüş kişi 5-kat) / 2, train+val (500 kayıt) üzerinde;
açık test = 40 kayıt (yalnız son kontrol). Docker = lionelyneuro/neural-compet:v3, CPU,
`scripts/docker_test_submission.sh`. Ağırlıklar ve ZIP'ler git'te değil (submissions/, ~/neuralbench/save/).

## Özet

| # | Gönderim tarihi | ZIP | Durum | Çift CV | Açık test | Docker süresi | Sealed skor | Sealed süre |
|---|---|---|---|---|---|---|---|---|
| – | (gönderilmedi) | hgb_v2_s0.zip | yedek | 46.42 | 54.9426 | 141 s | – | – |
| – | (gönderilmedi) | hgb_v3_np.zip | yedek | 43.30 | 52.4791 | 89 s ¹ | – | – |
| – | – | hgb_ens_c.zip | hazır aday | 42.50 | 52.8573 | 115 s ¹ | – | – |
| 1 | 2026-11-10 (planlanan) | mix_gru_hgb.zip | gönderilecek | 40.43 | 48.2226 | 123 s | _bekleniyor_ | _bekleniyor_ |

¹ Arka planda CV koşusu çalışırken ölçüldü (yükte). sklearn yolu (hgb_v3.zip, aynı model/skor): 417 s.

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

### mix_gru_hgb.zip (gönderim #1, 2026-11-10)
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
