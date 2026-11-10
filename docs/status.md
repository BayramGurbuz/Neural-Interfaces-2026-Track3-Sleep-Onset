# Proje durumu (son güncelleme: 2026-11-10)

Yeni oturumda buradan devam et. Ayrıntılı gönderim kayıtları: `docs/submissions_log.md`.
Final planı: `docs/final_plan.md`. GRU tasarımı: `docs/gru_design.md`. Kurallar: `CLAUDE.md`, `docs/*.pdf`.

## Model gelişimi (W-bMAE, saniye, düşük iyi)
Çift CV = (görülmemiş kişi 5-kat + görülmüş kişi 5-kat) / 2, train+val (500 kayıt). Açık test = 40 kayıt,
yalnız son kontrol (seçim için kullanılmadı).

| Model | Çift CV | Açık test | Not |
|---|---|---|---|
| Sabit 35 s / elapsed kuralı / EEGNet | – | ~60–61 | tabanlar |
| HGB v1 (regresyon, 321 öznitelik) | 47.78 | 54.83 | |
| c5 + kalibrasyon (hgb_v2_s0) | 46.42 | 54.94 | |
| B: dağılım sınıflandırıcısı (11 sınıf, ağırlıklı medyan) | 43.97 | – | en büyük sıçrama |
| B + dyn + kalibrasyon (hgb_v3_np) | 43.30 | 52.48 | |
| B + dyn + maxmin + spec + eye | 42.92 | – | 817 öznitelik |
| HGB ensemble (c): 4 dağılım modeli (hgb_ens_c) | 42.50 | 52.86 | |
| GRU-uw (3 seed) | 41.66 | 48.46 | nedensel GRU, ağırlıksız CE |
| GRU(3) + HGB(c), α 0.5 (mix_gru_hgb) | 40.43 | 48.22 | |
| **GRU(10) + HGB(c), α 0.6 (mix_gru10_hgb)** | **40.28** | **46.91** | en iyi |

En iyi model ayrıntısı: unseen 41.80 ± 1.91, seen 38.76 ± 1.41; HGB(c)'den ve yalnız GRU'dan 10/10
katta iyi. Kalibrasyon GRU/karışımda zarar veriyor → kullanılmıyor. Seed tekrarlanabilirliği: aynı seed
= bit düzeyinde aynı; farklı seed üçlüleri açık testte σ ≈ 0.48 (3 seed'li karışım).

## Warm-up metrikleri, açık test (40 kayıt, test kayıtlarını görmemiş modeller)
bMAE/MAE = warm-up tanımı (tüm pencereler havuzda, 4 aralık eşit ağırlık; = neuralbench test/bmae,
test/mae; `scripts/warmup_metrics.py`, EEGNet log'u ve final540 Docker çıktısıyla doğrulandı).

| Model | W-bMAE | bMAE | MAE |
|---|---|---|---|
| Sabit 35 s | 60.74 | 171.38 | 339.37 |
| Geçen süre kuralı | 61.25 | 155.21 | 297.39 |
| EEGNet (3 seed ort.) | 60.03 | 165.57 | 326.18 |
| hgb_v3 (np, kalibrasyonlu) | 52.48 | 119.06 | 220.62 |
| mix_gru10_hgb (500) | 46.91 | 104.95 | 189.56 |

**Warm-up leaderboard bilgi verici değil, sızıntılı girişler** (2026-11-10, `docs/warmup_update1_leaderboard.pdf`):
Warm-up (update 1) leaderboard'unda W-bMAE sütunu var, ama üst sıralar büyük ölçüde test kayıtlarıyla eğitilmiş
modeller (0.0 skorlar, "leaky" / "trained on all public data including warm-up test" notları). Dürüst kıyas
yapılamıyor; model kararları yalnızca çift CV ile verilir.

## Hazır ZIP'ler (submissions/, hepsi Docker'da test edildi)
| ZIP | Durum | Docker süresi (açık test) |
|---|---|---|
| mix_gru_hgb.zip | **17 Kasım gönderilecek (gönderim #1)** | 123 s |
| final540.zip | **18 Kasım gönderilecek (gönderim #2)**; 540 kayıt, kontrollerden geçti | 122 s |
| mix_gru10_hgb.zip | hazır aday (10 seed) | 132 s |
| hgb_ens_c.zip | hazır aday | 115 s |
| hgb_v3_np.zip / hgb_v3.zip | yedek | 89 s / 417 s |
| hgb_v2_s0.zip | yedek | 141 s |

Sealed pencere (2. erteleme): 17 Kasım 02:01 – 22 Kasım 01:59 TR. 5 hakkın tamamı duruyor.
Gönderim planı: 17 Kasım mix_gru_hgb → 18 Kasım final540 (540 kayıtla yeniden eğitilmiş 10 seed'li
model, testten geçerse) → 19–21 Kasım kalan 3 hak yedek. Final sıralama = en iyi sealed skor (otomatik, işaretleme
yok). Günde 1, toplam 5 gönderim. Göndermeden önce ZIP Docker'da test edilmeli. Claude gönderim yapmaz.

## Sıradaki adım
540 kayıtlık final eğitimi TAMAMLANDI (2026-11-10): final540.zip, kontroller ve Docker geçti
(ayrıntı `docs/submissions_log.md`). Sıradaki: 17 Kasım gönderim #1 (kullanıcı yapar), sealed skor/süre
günlüğe. Eski plan adımları (yapıldı):
1. `scripts/train_ens_c.py` ve `scripts/train_mix_final.py`'ye `--all` bayrağı (train+val+test).
2. HGB(c) 4 üye (iterasyonlar 225/200/550/200), GRU uw 10 seed × 15 epoch, α 0.6, kalibrasyon yok.
3. Kontroller: 540- ve 500-model tahmin karşılaştırması, örneklem içi skor, aynı seed'le ikinci koşu.
4. Paketle, Docker'da test et (süre), `docs/submissions_log.md`'ye yaz, commit.

## Açık sorular
- mix_gru_hgb'nin (ve v3_np yerine o gönderildiği için onun) sealed skoru ve sealed süresi — kullanıcı
  bildirecek; günlüğe yazılacak, 60 dk bütçesi buna göre netleşecek (10 seed ≈ ×1.07 süre).
- Sealed set boyutu açıklanmıyor (FAQ: "remain private").
- Train'deki en uzun kayıt 300 pencere (1500 s); GRU'nun daha uzun kayıtlardaki davranışı denenmedi
  (elapsed 1500 s'de kırpılıyor). İsteğe bağlı: sentetik uzatma testi.
- Ertelenenler: iğcik / K-kompleks detektörleri; göz kırpma eşiği (pencere başına 0.15 kırpma düşük).
- Final540 sonrası kalan haklarla ne denenecek (örn. GRU girdisine dyn öznitelikleri, ordinal kayıp).

## Önemli dosyalar
- Öznitelikler: `scripts/causal_features.py` (submission'a gömülür), `scripts/extract_features.py`.
- CV: `scripts/dual_cv.py`, `scripts/dist_hgb.py`, `scripts/ablation_c.py`, `scripts/ensemble_dist.py`,
  `scripts/gru_cv.py`, `scripts/seeds_cv.py`, `scripts/mix_gru_hgb.py`.
- Final eğitim: `scripts/train_ens_c.py`, `scripts/train_mix_final.py`.
- Paketleme: `scripts/build_hgb_submission.py` (numpy ağaçlar `scripts/tree_numpy.py`), Docker:
  `scripts/docker_test_submission.sh`.
- Git dışı: `~/neuralbench/save/` (öznitelikler, CV tahminleri `cv/dual/`, modeller `ens_c/`, `mix_uw*/`),
  `~/neuralbench/compet/` (Docker replay), `~/projects/2026-competition` (yarışma deposu).
- Ortam: `.venv/bin/python` (sistem python'unda numpy yok); script'ler `scripts/` içinden çalıştırılır.
