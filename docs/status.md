# Proje durumu (son güncelleme: 2026-11-11)

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
| mix_gru_hgb.zip | yedek (3 seed; mix_gru10_hgb onun yerine) | 123 s |
| final540.zip | **gönderildi (#1, 10 Kasım) → sealed 39.56** | 122 s (v3) |
| mix_gru10_hgb.zip | gönderilmeyecek (final540 ile aynı tarif, 500 kayıt) | 132 s |
| hgb_ens_c.zip | hazır aday | 115 s |
| hgb_v3_np.zip / hgb_v3.zip | yedek | 89 s / 417 s |
| hgb_v2_s0.zip | yedek | 141 s |

Sealed pencere AÇIK: 10 Kasım'da açıldı (organizatör maili + yarışma deposu 79b4453), bitiş 21 Kasım
23:59 CET = 22 Kasım 01:59 TR. 5 hak, takım başına ortak, günde en fazla 1.
Gönderim planı: **#1 final540 (10 Kasım, yapıldı → sealed 39.56, 23/60)** → kalan 4 hak YALNIZCA kabul
kuralını geçen yeni modeller için. **Son gönderim en geç 20 Kasım; 21 Kasım yedek gün.** mix_gru10_hgb
plandan çıkarıldı (final540 ile tek fark eğitim kayıt sayısı). Final sıralama = takımın en iyi geçerli
sealed skoru (otomatik, işaretleme yok), tekrarlanabilirlik denetimine bağlı. Sealed log/çıktılar gizli →
göndermeden önce ZIP Docker'da test edilmeli. Claude gönderim yapmaz.

## Kabul kuralı (kalan 4 hak için, 2026-11-11)
Yeni bir model ancak şu dört koşulun hepsini sağlarsa gönderilir:
1. Çift CV ortalaması mix_gru10_hgb'ye (40.28) göre **en az 0.3 iyi** (≤ 39.98);
2. **10 katın en az 8'inde** mix_gru10_hgb'den iyi (5 unseen + 5 seen kat, kat bazında eşleştirilmiş);
3. **540 kayıtla** yeniden eğitilmiş;
4. **v4-sealed Docker** testini geçmiş (imaj: `lionelyneuro/neural-compet:v4-sealed`).
Her deneyde aralık bazında döküm (`scripts/range_breakdown.py`) final540 tarifine (= mix_gru10_hgb OOF) karşı verilir.

## Sealed sonuç ve CV tutarlılığı
final540: W-bMAE macro **39.56** (W-bMAE 39.91, bMAE 101.36, MAE 198.26), Duration 1249.51 s, 23/60.
İlk 3: 33.06 / 34.62 / 35.26 (bMAE 80.03 / 87.72 / 96.64); 10. sıra 37.22.
**Çift CV tahminimiz (~40) sealed ile tutarlı:** mix_gru10_hgb OOF makro 40.28 / bMAE 100.06 / MAE 188.97.
OOF aralık katkıları (makro, W-bMAE'ye): [0,40) 7.92, [40,90) 8.01, [90,300) 14.07, [300,600] 10.28;
havuzlu aralık MAE'leri 14.95 / 30.16 / 87.73 / 267.42 → bMAE'nin ~2/3'ü [300,600]'den.

## Deney sonuçları (2026-11-11, referans = mix_uw_n10 = final540 tarifi, 500 kayıt OOF)
Δ = çift CV makro farkı (negatif iyi); aralık sütunları W-bMAE'ye katkı farkı. Kabul: Δ ≤ −0.30 ve ≥ 8/10 kat.
`scripts/mix_calib.py` (iç içe LOFO, 9 katta seç, dışarıdaki katın kayıtları seçimden maskeli), log
`~/neuralbench/save/cv/dual/mix_calib.log`. Referans yeniden üretimi birebir (maks |Δ| 0).

| Yöntem | Çift CV | Δ | [0,40) | [40,90) | [90,300) | [300,600] | bMAE | MAE | iyi kat |
|---|---|---|---|---|---|---|---|---|---|
| ref (LOFO α, τ 1) | 40.28 | – | 7.92 | 8.01 | 14.07 | 10.28 | 100.06 | 188.97 | – |
| sabit α 0.6, τ 1 | 40.25 | −0.03 | −0.02 | +0.02 | −0.02 | −0.01 | 99.95 | 188.70 | 1/10 |
| 1a τ global | 40.24 | −0.03 | +0.16 | +0.11 | −0.29 | −0.01 | 99.83 | 188.61 | 5/10 |
| 1b τ aralığa göre | 40.26 | −0.02 | −0.02 | +0.55 | −0.46 | −0.08 | 99.40 | 187.24 | 7/10 |
| 2a α aralığa göre (bölge = α0.6 tahmini) | 40.22 | −0.06 | −0.03 | +0.03 | −0.07 | +0.02 | 100.16 | 189.25 | 7/10 |
| 2b α geçen süreye göre (3 tertil) | 40.40 | +0.12 | −0.03 | +0.03 | +0.08 | +0.05 | 100.45 | 189.69 | 3/10 |
| 1+2 (α, τ) bölge başına ortak | 40.28 | +0.00 | −0.06 | +0.55 | −0.40 | −0.10 | 99.43 | 187.18 | 6/10 |
| 2a + τ global | 40.26 | −0.02 | +0.16 | +0.12 | −0.31 | +0.01 | 100.05 | 189.13 | 7/10 |
| 2b + τ global | 40.29 | +0.01 | +0.17 | +0.08 | −0.28 | +0.04 | 100.25 | 189.63 | 5/10 |

Sonuç: **hiçbiri kabul kuralını geçmiyor** (hepsi ±0.12 içinde). Seçilen değerler kararlı (τ ≈ 1.1; aralığa göre
τ ≈ 0.9 / 1.0–1.1 / 1.25 / 1.1; α ≈ 0.6 / 0.4–0.5 / 0.6 / 0.7) → karışım zaten metriğe göre iyi ayarlı. Bölgesel τ,
[90,300)'den kazandırıp [40,90)'da aynı kadar kaybettiriyor. [300,600] katkısı (~10.3) hiçbir son işlemle değişmiyor.

### Deney 3: dizi modelleri (`scripts/seq_cv.py`, `scripts/gru_cv.py` LSTMNet/TCNNet; log `seq_cv.log`)
NOT: "spec + eye'ı GRU girdisine ekle" maddesi zaten yapılmış durumda — GRU girdisi 62 pencere özniteliği
(base 40 + spec 20 + eye 2) + ilk-2-dk farkları + log1p(elapsed) = 125 (gru_cv.py başlığı, features_ext.npz
doğrulandı). Bu yüzden yalnız mimari varyantları denendi: uw tarifi, 10 seed, 30 epoch'a kadar, LOFO epoch.
TCN = nedensel dilate konvolüsyon (çekirdek 3, dilasyon 1–32, alıcı alan 127 pencere ≈ 10.6 dk), 84k parametre;
nedensellik testi geçti. Süre: LSTM 882 s, TCN 1102 s (GPU).

| Model | Çift CV | Δ (ref 40.28) | [0,40) | [40,90) | [90,300) | [300,600] | bMAE | iyi kat |
|---|---|---|---|---|---|---|---|---|
| GRU-uw 10 seed (tek) | 41.43 | +1.15 | +0.57 | +0.68 | −0.06 | −0.04 | 99.65 | 0/10 |
| LSTM 10 seed (tek) | 42.90 | +2.62 | +1.22 | +0.85 | +0.46 | +0.09 | 102.85 | 0/10 |
| TCN 10 seed (tek) | 41.51 | +1.23 | +0.43 | +0.68 | +0.37 | −0.25 | 100.92 | 1/10 |
| LSTM + HGB(c), α LOFO (0.5) | 41.03 | +0.75 | +0.18 | −0.02 | +0.51 | +0.07 | 102.02 | 1/10 |
| TCN + HGB(c), α LOFO (0.6) | 40.44 | +0.16 | −0.09 | +0.04 | +0.38 | −0.17 | 100.60 | 3/10 |
| **(GRU+TCN)/2 + HGB(c), α iç içe LOFO (0.7)** | **40.00** | **−0.28** | −0.09 | −0.02 | −0.02 | −0.14 | 99.68 | **8/10** |

LSTM her yerde kötü. TCN tek başına GRU ile aynı seviyede ama [300,600]'de biraz daha iyi; iki dizi modelinin
ortalaması + HGB **eşiğe 0.02 kala** (Δ −0.28, 8/10 kat; kazanç unseen'de 5/5, seen'de 3/5). Üç yönlü ağırlık
ızgarası (GRU/TCN/HGB ayrı ağırlık, iç içe LOFO, `mix3_gru_tcn_hgb.log`): 40.06, Δ −0.22, 8/10 — serbest
ağırlıklar kat kat oynuyor ((0.4,0.3) ile (0.5,0.2) arası), basit ortalama + tek α daha sağlam. İç içe olmayan
ızgara minimumu 39.98 ((0.4,0.3)) → gerçek kazanç ≈ 0.25–0.3, eşiğin tam kenarında. **Kabul kuralını geçen
yok.**

## Sıradaki adım
Deneyler 0–3 tamamlandı (yukarıda); hiçbiri kabul kuralını geçmedi. Seçenekler (kullanıcı kararı):
(a) GRU+TCN karışımını 540'ta eğitip göndermek için kural dışına çıkmak (Δ −0.28 ≈ eşik; sealed'da ~0.3 s
kazanç, sıralamayı değiştirmez); (b) [300,600] katkısına (10.3 → liderlerle farkın büyük kısmı) doğrudan hedefli
bir yaklaşım: uzun-ufuk sınıflarında ayrı bir model / daha geniş zaman ufuklu öznitelikler (10–20 dk), ya da
TCN'nin alıcı alanını büyütmek (TCN tek başına [300,600]'de −0.25 veriyor). Son işlemler (τ, bölgesel α) bu
aralığı hiç değiştirmiyor.

## Doküman güncellemesi (2026-11-10, depo 4a8c943 → 79b4453, 5 Kasım PDF'lerine göre)
- Değişen: kota artık **takım başına ortak** (5 toplam, günde 1); leaderboard her katılımcının en iyisini
  ayrı gösterir, final sıralama yine takımın en iyi geçerli sealed skoru. Kayıt kapandı, Track 03 açık.
  Sealed baseline skorları participant guide'da yayımlanıyor.
- DEĞİŞMEYEN: Scientific FAQ'de yalnız "Opening update" notu silindi; W-bMAE tanımı, seen/unseen
  eşit ağırlık, girdi/akış sözleşmesi aynı. Süre sınırı aynı (60 dk, 1 A100). Tekrarlanabilirlik
  denetimi aynı (ilk 3, 22–30 Kasım, ±2σ, yöntem açıklaması + harici veri + compute tahmini). Sealed
  log/çıktı gizliliği aynı.

## Açık sorular
- **v4-sealed imajı Docker Hub'da yok** (2026-11-11: yalnız v2, v3 tag'leri; `docker pull` "manifest unknown";
  depo yaml'ı hâlâ `docker_image: lionelyneuro/neural-compet:v3`). Doğru tag/kaynak kullanıcıdan alınacak;
  o zamana kadar v4 testi yapılamıyor.
- Leaderboard'da > 4500 s Duration'lar var, oysa sealed `execution_time_limit: 3600` (yaml) ve Rule 5
  "test pass 60 dk". Duration muhtemelen kuyruk/ingestion+scoring dahil; doğrulanmadı.
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
