# Final plan: seçilen modeli 540 kaydın tamamıyla yeniden eğitmek

## Seçilen model (çift CV ile)
GRU (config `uw`, 10 seed) + HGB ensemble (c) (4 üye), ortak ızgarada metrik-ağırlıklı karışım.
Çift CV (kalibrasyonsuz): 3 seed 40.43 → 10 seed 40.28 (3 seed'e göre 7/10 katta iyi; asıl kazanç
seed varyansının düşmesi: 3 seed'li karışımda açık test σ ≈ 0.48).

## CV'den sabitlenen her şey (540 eğitiminde DEĞİŞMEZ)
| Parça | Değer | Kaynak |
|---|---|---|
| GRU mimarisi / optimizasyon | Linear(125→64)+GELU+LN → GRU 64×2, dropout 0.2; ağırlıksız CE; AdamW lr 2e-3, wd 1e-2; cosine T_max 150; augmentasyon aynı | `gru_cv.CONFIGS["uw"]` |
| GRU epoch | 15 (10 LOFO seçiminin medyanı) | `seeds_cv.py`, uw_n10 |
| GRU seed'leri | 0–9 | — |
| Karışım α | 0.6 (10 LOFO seçiminin medyanı) | `mix_gru_hgb.py uw_n10` |
| HGB üyeleri | B 225, B_h1 200, B_h2 550, B2 200 iterasyon; parametreler `ensemble_dist.MEMBERS` | `train_ens_c.py` (LOFO medyanları) |
| Öznitelikler | 817 (taban + dyn + maxmin + spec + eye) | `ablation_c` |
| Kalibrasyon | yok (karışımda zarar veriyor) | çift CV |

Veriden yeniden hesaplanan (öğrenilen parametre değil, eğitim verisinin deterministik fonksiyonu):
HGB ön-binleme sınırları, ızgara ω, sınıf ω_B, GRU standardizasyon ortalaması/std'si.

## Adımlar
1. `train_ens_c.py` ve `train_mix_final.py`'ye `--all` bayrağı: eğitim verisi = 540 kayıt (train+val+test).
   Test skorlaması bu modda "örneklem içi" olur → yalnızca boru hattı kontrolü olarak raporlanır.
2. HGB(c) 4 üye 540 kayıtta (~30–35 dk), sonra GRU 10 seed × 15 epoch (~1 dk).
3. Kontroller (doğrulama seti kalmadığı için):
   - 540-model ile 500-model tahminlerinin karşılaştırması (aynı pencerelerde ortalama mutlak fark,
     korelasyon) → büyük sapma = hata belirtisi.
   - Örneklem içi skor 500-model'in skorundan düşük olmalı (sağlık kontrolü, iyimser).
   - Aynı seed'lerle ikinci koşu → bit düzeyinde aynı mı (tekrarlanabilirlik).
4. Paketleme (`build_hgb_submission.py`), Docker testi (skor = örneklem içi; süre ölçümü önemli).
5. `docs/submissions_log.md`'ye giriş: commit hash, komutlar, seed'ler, çift CV (500 kayıtlık CV skoru),
   Docker süresi.

## Neden epoch/iterasyonları 540'a göre ölçeklemiyoruz
Veri %8 artıyor; CV eğrileri seçilen noktada düz (GRU 10 ≈ 15 epoch; HGB eğrileri ±25 ağaçta düz).
Ölçekleme doğrulanamayan bir tahmin olur; sabit tutmak denetimde savunulabilir ve tekrarlanabilir.

## Gönderim sırası (günde 1, toplam 5; sayılan = en iyi sealed skor)
Sealed pencere: 17 Kasım 02:01 – 22 Kasım 01:59 TR (5 gün → 5 hak).
- 17 Kasım: mix_gru_hgb.zip (3 seed, 500 kayıt).
- 18 Kasım: final540 (10 seed, 540 kayıt) — Docker testinden geçerse.
- 19–21 Kasım, kalan 3 hak: sealed süre/hata sorunları ve olası son iyileştirmeler için yedek.
Kötü bir gönderim en iyi skoru silmediği için final540'ı göndermenin sıralama riski yok.

## Süre bütçesi
Docker (açık test, 40 kayıt): 3 seed 123 s, 10 seed 132 s. 60 dk sınırı için asıl ölçü v3_np /
mix_gru_hgb'nin sealed süresi; oran ~1.1 ile çarpılarak 10 seed tahmin edilir.

## Riskler
- Doğrulama seti yok → hatalar ancak kontrollerle (adım 3) yakalanır.
- Test kayıtlarının kişileri zaten train'de → seen grubuna yeni geceler eklenir (yararlı); unseen
  performansı değişmez.
- Train'deki en uzun kayıt 300 pencere; GRU'nun daha uzun kayıtlardaki davranışı denenmedi
  (elapsed 1500 s'de kırpılıyor). İstenirse sentetik uzatma testi eklenebilir.
