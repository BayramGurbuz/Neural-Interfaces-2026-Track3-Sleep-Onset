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
| GRU(10) + HGB(c), α 0.6 (mix_gru10_hgb / final540) | 40.28 | 46.91 | gönderildi, sealed 39.56 |
| **(GRU10 + TCN10 + GRU-emb3)/3 + HGB(c), α 0.7 (mix3 / mix3_540)** | **39.40** | 47.06 | kabul kuralını geçti (−0.88, 10/10); gönderildi (#2, 11 Kasım), sealed 38.66 |

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
| mix3_540.zip | **gönderildi (#2, 11 Kasım) → sealed 38.66** | 225 s / 213 s (v3, ×2 aynı skor) |
| mix3_540_grid.zip | **aday #3** (mix3_540 ağırlıkları + 30 sn ızgara kararı π 0.86 + NaN düzeltmesi; çift CV 38.35, Δ −1.05, 10/10; gönderim onay bekliyor) | 230 s / 218 s (v3, ×2 aynı skor) |
| mix3_540_gridpi.zip | **gönderildi (#3, 12 Kasım), sealed 37.49** (W-bMAE 37.78, bMAE 105.48, MAE 213.98, süre 2762.79 s) (mix3_540 ağırlıkları + ızgara π(D) bins4 1.0/0.592/0.644/0.444 + NaN ve düz kanal temizliği; çift CV 38.27, ızgaralıya göre −0.08, 10/10; SHA-256 4ef85c1f…0063) | 185 s / 176 s (v3, ×2 aynı skor; aynı oturumda mix3_540_grid 168 s) |
| mix_gru10_hgb.zip | gönderilmeyecek (final540 ile aynı tarif, 500 kayıt) | 132 s |
| hgb_ens_c.zip | hazır aday | 115 s |
| hgb_v3_np.zip / hgb_v3.zip | yedek | 89 s / 417 s |
| hgb_v2_s0.zip | yedek | 141 s |

Sealed pencere AÇIK: 10 Kasım'da açıldı (organizatör maili + yarışma deposu 79b4453), bitiş 21 Kasım
23:59 CET = 22 Kasım 01:59 TR. 5 hak, takım başına ortak, günde en fazla 1.
Gönderim planı: **#1 final540 (10 Kasım, yapıldı → sealed 39.56, 23/60)**, **#2 mix3_540 (11 Kasım, yapıldı →
eski skorlama 38.66, 13 Kasım yeniden skorlama 37.79, 17./123)**, **#3 mix3_540_gridpi (12 Kasım, yapıldı → eski skorlama 37.49; ızgara 13 Kasım kuralıyla yasak)** → **7 hakkın 3'ü kullanıldı, kalan 4 hak** (13 Kasım: +2) YALNIZCA kabul kuralını geçen ızgarasız yeni modeller için. **Son gönderim en geç 20 Kasım; 21 Kasım yedek gün.** mix_gru10_hgb
plandan çıkarıldı (final540 ile tek fark eğitim kayıt sayısı). Final sıralama = takımın en iyi geçerli
sealed skoru (otomatik, işaretleme yok), tekrarlanabilirlik denetimine bağlı. Sealed log/çıktılar gizli →
göndermeden önce ZIP Docker'da test edilmeli. Claude gönderim yapmaz.

## 13 Kasım kural değişikliği (organizatörler; `~/projects/2026-competition` ca8f486, scientific_faq_sleep.md, terms.md)
- Mühürlü değerlendirme teknik güncelleme; tüm Finished gönderimler yeniden skorlandı (aynı ID, hak yemedi). mix3_540 yeni
  skoru **37.79** (17./123; 3.lük 36.52). Eski 39.56 / 38.66 / 37.49 geçersiz. Hak 5 → 7; 3 kullanıldı, **4 kaldı**.
- Yeni FAQ: tahminler EEG'den gelmeli; kayıtların etiketlenme/paketlenme düzenliliklerini kullanmak yasak, denetimde kontrol
  edilir. Geçen süre (akışın ilk penceresinden beri) serbest. "Akış kaydın başından başlar, rastgele başlangıç yok" cümlesi
  kaldırıldı → akış kaydın ortasından başlayabilir.
- Sonuçlar: 30 sn ızgarası (π, π(D), c/epoch grubu, t mod 30) bütün yeni adaylardan çıkarıldı; GRID kapalı. İş 1 (HGB a+c)
  iptal. Referans ızgarasız mix3 (çift CV 39.40 ↔ 37.79); kabul kuralı aynı eşiklerle bu referansa göre. Yeni iş: akış
  başlangıcına dayanıklılık testi (`scripts/stream_start_test.py`). Kodun düzenlilik taraması `B_ASAMASI_RAPORU.md`'de.

## B aşaması (2026-11-12/13; ayrıntı `B_ASAMASI_RAPORU.md`)
- B1: π(D) ızgara ağırlığı (38.27, −0.08, 10/10) + düz kanal temizliği → mix3_540_gridpi (gönderim #3, sealed 37.49; yerel 38.27 → −0.78, önceki iki gönderimle aynı fark → ızgara mühürlü veride tutuyor).
- B2 (B üyesi): bipolar −0.64 (9/10), olay dedektörleri +0.21, epoch-hizalı (c) −0.77 (10/10, ızgaraya bağlı, ayrı
  tutuluyor), a+c −1.08. HGB(c)+bipolar 42.17 (−0.48). **Karışımda kazanç kayboluyor:** en iyi HGB(a) + dizi(eski)
  38.19 (−0.08, 5/10); c dizi girdisinde zararlı (+0.44). Kabul eşiğini geçen tarif yok, yeni ZIP yok.
- B3: HMC evreleyici Muse'a taşınmıyor (Spearman −0.10 / AUC 0.67 ↔ en iyi öznitelik 0.42 / 0.83) → durduruldu.
- 14 Kasım: İş 1 (HGB a+c) 13 Kasım kuralıyla iptal. İş 2 sonucu: HGB(c)+n2det 40.76 (−1.89, 9/10); ızgarasız karışım en iyi
  HGB(n2det)+dizi(eski) 38.97 (−0.43, 8/10, α 0.6) → kabul kuralı GEÇİLMEDİ (−0.5&8/10 ya da −0.3&10/10). Kullanıcı kararı bekleniyor.
  Yeni aday kuralı: pencere başına çıkarım ≤ gridpi +%10 (gridpi sealed süresi 2762.79 s; tahmin ≤ 2500 idi).
- İş 2 (alan içi N2 dedektörü, 14 Kasım, ara sonuçlar; betikler `scripts/n2det_*.py`):
  - Ham dosyalarda onset sonrası tam 300 s veri var (540/540, NaN yok); ham değer × 0.1 = akış penceresi (1e-7).
    Akış boru hattı kaydı onset'te kesiyor → bu kısım daha önce hiç kullanılmamıştı.
  - Dedektör: pozitif = onset sonrası 60 pencere, negatif = kalan ≥ 300 s; kat başına yalnız eğitim kayıtları,
    eğitim satırlarına iç 5 katlı fold-dışı P. Akıl sağlığı (unseen OOF): CNN Spearman −0.48, AUC son60↔300+ 0.945,
    son30↔60–150 0.815; HGB −0.43 / 0.935 / 0.802; en iyi tek öznitelik 0.42 / 0.83 / 0.73 → GEÇTİ.
  - Maliyet: HGB dedektörü (+ bip) ≈ 3.2 ms/pencere (> +%10) → adım e yalnız CNN dedektörüyle (≈ 1.5 ms, yük altında).
    5 nedensel özet: P, EMA 30 s, EMA 2 dk, EMA30 koşan maks, EMA30 30 s artış.
  - B üyesi + n2det 40.86 (−2.34, 10/10). GRU 40.97 (−0.47, 5/10), TCN 41.04, GRU-emb 43.76.
    Karışım (ref 38.27): HGB(eski)+dizi(n2det) 38.17 (−0.10, 5/10); HGB(yalnız B n2det)+dizi(eski) 38.07 (−0.21, 10/10).
  - 14 Kasım sabahı Windows güncellemesiyle makine kapandı: HGB(c) üye koşuları (İş 1 a+c ve İş 2 n2det, B_h1/B_h2/B2)
    kayboldu; protokol başına ara kayıt eklendi, `scripts/b2_resume.sh` ile sırayla yeniden başlatıldı (önce İş 2).

## Karar katmanı deneyleri (2026-11-12, yeniden eğitim yok; ayrıntı `KARAR_KATMANI_RAPORU.md`)
`scripts/decision_layer.py`, mix3'ün kayıtlı fold-dışı olasılıkları, resmi binned_mae, iç içe LOFO. Referans 39.40.
| Varyant | unseen | seen | Çift CV | Δ | iyi kat |
|---|---|---|---|---|---|
| tam metrik ağırlığı w(t, y) | 41.10 | 38.33 | 39.71 | +0.32 | 2/10 |
| **30 sn ızgarası, π 0.86** | **39.74** | **36.96** | **38.35** | **−1.05** | **10/10** |
| ızgara, π LOFO | 39.77 | 37.00 | 38.38 | −1.02 | 10/10 |
| nokta tahminini ızgaraya çekme | 39.86 | 37.21 | 38.54 | −0.86 | 10/10 |
| yumuşatma (LOFO a = 0) | 40.73 | 38.07 | 39.40 | 0.00 | 0/10 |
| tam ağırlık + ızgara | 40.01 | 37.17 | 38.59 | −0.81 | 9/10 |

Izgara kazancı [0,40)'tan (−1.31); ızgaradaki kayıtlarda −1.39, ızgara dışı kayıtlarda +1.11 (sıfır noktası ızgara oranı
≈ 0.44; açık veride 0.857, sealed'da bilinmiyor). Açık test 47.06 → 46.61. **Aday #3 = mix3_540_grid.zip** (aynı
model.joblib, yalnız submission.py: ızgara kararı + NaN girdi temizliği); akış ↔ toplu maks |Δ| 0.0009 s, Docker v3 ×2
31.6917 (örneklem içi), süre mix3_540 ile aynı düzeyde. **Gönderilmedi; kullanıcı onayına bağlı.** v4-sealed testi yok.
NaN düzeltmesi: `build_hgb_submission.py` `_clean` (temiz veride çıktı birebir aynı; senaryolar raporda).

## Kabul kuralı (kalan haklar için, 2026-11-11; 12 Kasım gönderiminden (#3) sonra kalan 2 hak)
Yeni bir model ancak şu dört koşulun hepsini sağlarsa gönderilir:
1. Çift CV ortalaması mix_gru10_hgb'ye (40.28) göre **en az 0.3 iyi** (≤ 39.98);
2. **10 katın en az 8'inde** mix_gru10_hgb'den iyi (5 unseen + 5 seen kat, kat bazında eşleştirilmiş);
3. **540 kayıtla** yeniden eğitilmiş;
4. **v4-sealed Docker** testini geçmiş (imaj: `lionelyneuro/neural-compet:v4-sealed`).
Her deneyde aralık bazında döküm (`scripts/range_breakdown.py`) final540 tarifine (= mix_gru10_hgb OOF) karşı verilir.

## Sealed sonuç ve CV tutarlılığı
final540: W-bMAE macro **39.56** (W-bMAE 39.91, bMAE 101.36, MAE 198.26), Duration 1249.51 s, 23/60.
İlk 3: 33.06 / 34.62 / 35.26 (bMAE 80.03 / 87.72 / 96.64); 10. sıra 37.22.
mix3_540 (gönderim #2, 11 Kasım): W-bMAE macro **38.66** (final540'a göre −0.90; çift CV farkı −0.88). İlk 3 eşiği
≈ 33.06 → fark 5.60 s. Sealed sıra, bMAE/MAE ve Duration bu dosyaya kaydedilmedi (gönderim öncesi süre tahmini
≈ 2200–2300 s). Çift CV ↔ sealed farkı: final540 40.28 → 39.56 (−0.72), mix3 39.40 → 38.66 (−0.74).
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

### Yedek aday: (GRU+TCN)/2 + HGB(c), α 0.7 (çift CV 40.00, Δ −0.28, 8/10)
Kabul kuralını geçmediği için şimdi gönderilmiyor. **18 Kasım'a kadar daha iyisi çıkmazsa** 540 kayıtla
eğitilip boşta kalan bir hakla gönderilecek (gerekli: `train_mix_final.py`'ye TCN eğitimi, builder'a TCN
inference — hepsi torch, yeni kütüphane yok).

## Teşhis (2026-11-11): seen/unseen dökümü ve deneğe özgü bilgi
**Seen/unseen (mix_gru10_hgb OOF; katkılar W-bMAE'ye):**

| Protokol | W-bMAE | [0,40) | [40,90) | [90,300) | [300,600] | bMAE | MAE |
|---|---|---|---|---|---|---|---|
| unseen (5 kat) | 41.80 | 8.38 | 8.37 | 14.32 | 10.73 | 104.06 | 197.10 |
| seen (5 kat) | 38.76 | 7.46 | 7.65 | 13.82 | 9.83 | 96.07 | 180.84 |
| makro | 40.28 | 7.92 | 8.01 | 14.07 | 10.28 | 100.06 | 188.97 |

Seen–unseen farkı 3.0 s; her aralıkta seen daha iyi, en büyük fark [300,600] ve [0,40).

**Girdi sözleşmesi (yarışma deposu `benchmark_utils/streaming.py`, `nb_task.py`, FAQ):** `predict(X)` yalnız
(1, 4, 640) EEG tensörünü alır. `stream_id`, `record_id`, `onset`, `subject_id` pencere `info`'sunda taşınır ve
modele verilmez ("kept away from the model"). `meta` = sfreq, ch_names, chs_info, n_chans, n_times,
target_shape, device, submission_dir — denek/kayıt bilgisi yok. Her kayıt modelin taze kopyasında koşar →
kayıtlar arası durum da taşınamaz. **Denek kimliği çıkarımda yok; yalnız sinyalden çıkarılabilir.**

**Uykuya dalma süresi varyansı (`scripts/subject_info.py`, train+val 500 kayıt):** medyan 690 s (IQR 480–990,
%10–90: 330–1265), ortalama 751 ± 347. ≥2 geceli 113 denek (410 kayıt): denekler arası SD 214, denek içi SD
273, **ICC(1) = 0.38** (log: 0.40). Deneğin diğer gecelerinin medyanı bu geceyi 252 s MAE ile öngörüyor; genel
medyan 291 s → kimlik bilgisi süreyi yalnız %13 daha iyi tahmin ediyor. Denek içi varyans baskın.

**Denek tanıma sinyalden (LORO en yakın komşu, 62 pencere özniteliğinin ortalaması, 410 sorgu, şans %0.6):**
ilk 60 s top-1 %12.9 (top-5 %27), ilk 120 s **%17.3** (top-5 %37), tüm kayıt %39.5 (top-5 %60). Kimlik
sinyalden kısmen çıkarılabiliyor ama ilk 2 dakikada zayıf.

**Oracle (seen katlar, HGB B üyesine deneğin diğer eğitim gecelerinin medyan süresi + sayısı + bundan türeyen
kalan süre eklendi; yalnız katın eğitim kayıtlarından; `scripts/oracle_subject.py`, `oracle_subject.log`, B
üyesi 400 ağaç LOFO, ~50 dk):**

| Seen protokolü (5 kat) | W-bMAE | [0,40) | [40,90) | [90,300) | [300,600] | bMAE | MAE | iyi kat |
|---|---|---|---|---|---|---|---|---|
| B (817) | 41.47 | 8.08 | 7.91 | 15.23 | 10.26 | 100.82 | 188.54 | – |
| B + oracle (820) | 40.87 (−0.60) | 8.15 | 7.89 | 15.04 | **9.78** | 96.85 | 178.99 | 5/5 |
| B + GRU, α 0.6 | 38.82 | 7.48 | 7.70 | 13.94 | 9.69 | 95.31 | 178.69 | – |
| B_oracle + GRU, α 0.6 | 38.52 (−0.30) | 7.56 | 7.71 | 13.83 | **9.43** | 93.20 | 173.60 | 5/5 |
| yalnız oracle_rem (EEG yok) | 143.68 | 69.35 | 39.15 | 28.17 | **7.01** | 151.84 | 157.93 | – |

Yorum: kimlik bilgisi tam verilse bile seen tarafında −0.30 (makroya −0.15; unseen'e katkısı yok) ve kazanç
tamamen [300,600]'den. Çarpıcı nokta: EEG'siz "deneğin tipik süresi" [300,600]'de bizim tüm modellerden iyi
(7.0 vs 9.4–10.3) → o aralıktaki hata büyük ölçüde "bu kişi ne kadar geç uyur" bilgisi; pencere EEG'si bunu
taşımıyor. Kimlik çıkarımda verilmediği ve ilk 2 dk'dan tanıma %17 olduğu için gerçekçi üst sınır çok daha
küçük (≈ −0.05 makro). Olası dolaylı yol: kayıt özeti ile eğitim kayıtlarına kNN "örtük önsel" (komşuların
süre dağılımı öznitelik olarak; nedensel, model dosyasına 540 × 62 özet gömülür) — düşük öncelik.

## Tahmin (çalıştırılmadı): ham pencereden öğrenen küçük CNN kodlayıcı → GRU
Veri: 80 975 pencere × 4 × 640 float32 ≈ 83 MB, GPU'ya sığar. Kodlayıcı: 3 katman Conv1d (32–64 kanal,
~30k parametre), çıkış 32-boyutlu gömme; kayıp = bizim 11 sınıflı dağılım CE'si (pencere başına).
- **İki aşamalı** (önerilen): kodlayıcı pencere düzeyinde eğitilir, gömme GRU girdisine (125 → 157) eklenir,
  GRU yeniden eğitilir. RTX 3050'de pencere epoch'u ≈ 5–10 s (batch 512, 158 adım) → 30 epoch × 10 kat × 3 seed
  ≈ 1.5–2.5 saat; gömme çıkarımı + GRU çift CV (10 seed) ≈ 15 dk; final 540 eğitimi ≈ 20 dk. **Toplam ≈ 2–3 saat.**
- **Uçtan uca** (kodlayıcı + GRU kayıt dizisi üzerinde): epoch ≈ 15 s; 15 epoch × 10 kat × 3 seed ≈ 2 saat,
  10 seed ≈ 6 saat. Bellek: 32 kayıt × 300 pencere × 4 × 640 ≈ 100 MB + aktivasyonlar, 4 GB'a sığar.
- **Sunucu süresi:** pencere başına CNN çıkarımı CPU'da ≈ 0.2–0.5 ms (1–5 MFLOP), mevcut ≈ 21 ms/pencere
  (Welch + numpy ağaçlar) yanında ihmal edilebilir → Duration 1249 s → ≈ 1260–1290 s (+%1–3). Model dosyası
  +~0.1 MB. Yeni kütüphane yok (torch zaten var). cuDNN conv nedeniyle yeniden eğitim bit düzeyinde aynı
  olmayabilir (GRU notu gibi); ±2σ denetimi için sorun değil.
- Beklenti: el yapımı bant güçleri zaten var; kazanç iğcik/K-kompleks morfolojisinden gelebilir, büyüklüğü
  belirsiz. Uzun-ufuk deneylerinden sonra karar verilecek.

## CNN kodlayıcı, iki aşamalı (2026-11-11)
Kurulum: `extract_raw.py` (ham pencereler, 829 MB, sıra features_base ile doğrulandı); `cnn_encoder.py`:
1D CNN, 4 blok stride 2 + 2 dilasyonlu blok (çekirdek 7) → alıcı alan 667 örnek (tüm pencere), 35 323
parametre, global ort+maks havuz → 32-b gömme → 11 sınıf (B kenarları, ağırlıksız CE); pencere başına
ölçek x/20, ±300 µV kırpma; kazanç + gürültü artırması; **kat başına ayrı kodlayıcı** (held-out kayıtlar
kodlayıcıyı görmez), 3 seed, LOFO epoch (9–15), 1263 s. `gru_cnn.py`: GRU-uw + gömme (+ ilk-2-dk farkı),
seed s GRU ↔ seed s kodlayıcı. `mix_seq.py`: dizi modeli(leri) + HGB(c), iç içe LOFO α (yedek adayı
birebir yeniden üretiyor: 40.00, α 0.7).

| Model | unseen | seen | Çift CV | Δ (ref) | [0,40) | [40,90) | [90,300) | [300,600] | bMAE | iyi kat |
|---|---|---|---|---|---|---|---|---|---|---|
| CNN tek başına (3 seed) | 50.46 | 50.51 | 50.49 | +8.82 (uw_n3) | +1.74 | +0.13 | +3.35 | +3.60 | 128.2 | 0/10 |
| GRU + emb (in-sample, 3 seed) | 44.94 | 43.42 | 44.18 | +2.51 (uw_n3) | +1.85 | +0.71 | −0.59 | +0.55 | 103.1 | 1/10 |
| GRU+emb + HGB(c), α 0.5 | 41.21 | 38.95 | 40.08 | −0.20 (mix_uw_n10) | +0.10 | −0.14 | −0.49 | +0.34 | 101.0 | 7/10 |
| aynı, ref mix_uw_n3 (40.43) | | | | −0.35 | +0.07 | −0.10 | −0.55 | +0.23 | | 8/10 |
| CNN split-half tek başına | 53.04 | 51.54 | 52.29 | +10.63 (uw_n3) | +1.35 | −0.47 | +4.96 | +4.79 | 136.8 | 0/10 |
| GRU + logit (split-half, 3 seed) | 42.91 | 42.23 | 42.57 | +0.90 (uw_n3) | +0.65 | +0.21 | −0.15 | +0.19 | 101.8 | 4/10 |
| GRU+logit + HGB(c), α 0.5 | 41.51 | 39.31 | 40.41 | +0.13 (mix_uw_n10) | +0.11 | −0.11 | −0.06 | +0.19 | 101.2 | 4/10 |
| aynı, ref mix_uw_n3 (40.43) | | | | −0.02 | +0.08 | −0.07 | −0.11 | +0.08 | | 6/10 |

- CNN tek başına: [40,90) katkısı GRU ile aynı (tek pencereden "1–1,5 dk kala" morfolojisi görülüyor), kayıp
  bağlam gerektiren [90,300)/[300,600]'de; unseen ≈ seen → kodlayıcı kişiye özgü bir şey öğrenmiyor.
- CE in/out (seçilen epoch): oran ort. 1.07; unseen katlarında 1.03–1.115 (2/10 kat > %10), seen 1.04–1.07.
- GRU+emb: eğriler epoch 5'ten sonra hızla bozuluyor = stacking kayması (GRU in-sample gömmelere aşırı güveniyor).
  Tek başına kötü ama HGB ile karışımı mix_uw'dan iyi (α 0.5; kazanç [90,300)'den, [300,600] kötü):
  kodlayıcılı GRU HGB'yi daha iyi tamamlıyor. Kabul kuralını geçmiyor (−0.20, 7/10).
- Split-half (`cnn_encoder.py cnn3h 3 half`, 1330 s; eğitim kayıtları ikiye bölünür, her yarının kodlayıcısı
  diğer yarıyı kodlar, held-out = iki kodlayıcının logit ortalaması; GRU'ya 11 logit): kodlayıcı yarı veriyle
  daha zayıf (52.29), CE out/in oranı 1.08–1.19. GRU+logit 42.57: stacking kayması azalmış (emb 44.18 → 42.57,
  eğriler daha yavaş bozuluyor) ama hâlâ düz GRU'dan kötü (+0.90, 4/10); HGB karışımı 40.41 = mix_uw_n3 ile aynı.
- **Sonuç: iki varyant da kullanıcı eşiğini (Δ ≤ −0.15 veya ≥ 7/10 kat, uw_n3'e göre) geçmedi → 10 seed ve
  2D CNN varyantı çalıştırılmadı; kabul kuralını geçen model yok.** Ham pencereden öğrenilen morfoloji, el yapımı
  62 öznitelikten GRU'ya ek bilgi vermiyor; tek işe yarar iz, emb-GRU'nun HGB'yi çeşitlilikle tamamlaması
  (40.08, −0.20, 7/10) — kural eşiğinin altında, seed gürültüsü (σ≈0.3) düzeyinde.
- Süre: çıkarım 20 s, kodlayıcı 1263 + 1330 s, GRU 473 + 397 s, karışımlar ~2 dk → toplam ≈ 1 saat GPU.

**Üçlü karışım (kullanıcı isteği, yalnız karışım):** `mix_seq.py mix3_gru_tcn_cnnemb uw_n10 tcn_n10 cnn3_emb` —
dizi modelleri eşit ağırlık (her biri metrik-ağırlıklı normalize grid dağılımı, ortalama), yalnız α iç içe LOFO
(10 katta da 0.7):

| Model | unseen | seen | Çift CV | Δ (ref) | [0,40) | [40,90) | [90,300) | [300,600] | bMAE | iyi kat |
|---|---|---|---|---|---|---|---|---|---|---|
| **(GRU10+TCN10+GRU-emb3)/3 + HGB(c), α 0.7** | **40.73** | **38.07** | **39.40** | **−0.88** (mix_uw_n10) | −0.22 | −0.27 | −0.47 | +0.09 | 99.69 | **10/10** |
| aynı, ref yedek aday (GRU+TCN)/2 + HGB (40.00) | | | | −0.60 | −0.13 | −0.24 | −0.46 | +0.23 | | 10/10 |

**Kabul kuralını geçti** (Δ ≤ −0.30 ve ≥ 8/10): kazanç [0,40)/[40,90)/[90,300)'den, [300,600] hafif kötü.
Tek başına zayıf olan GRU-emb (44.18), karışımda çeşitlilikle −0.60 getiriyor (ref GRU+TCN+HGB).
**Aday #2 = bu model.** 540 eğitimi: `train_mix3_final.py --all` (GRU 10×15 ep, TCN 10×10 ep, kodlayıcı 3×12 ep,
GRU-emb 3×10 ep; HGB = ens_c_all) → `~/neuralbench/save/mix3_all/`; `build_hgb_submission.py` "mix3" türü
(vektörleştirilmiş `TCNBank`, kodlayıcı, GRU-emb; yalnız numpy/torch/joblib/scipy). Train+val sürümü
(`mix3/`): açık test mix3 47.06 (mix2 46.55, mix_gru10 46.91 — 40 kayıt, gürültülü, seçimde kullanılmaz).
Akış ↔ toplu eşdeğerlik (`check_stream_mix3.py`): maks |Δ| 0.055 s, W-bMAE 47.0599 ↔ 47.0597 ✓.
Yerel süre 33.6 ms/pencere (TCNBank öncesi 53.9; final540 ≈ 21) → sunucu tahmini ≈ 1249 × 1.6 ≈ 2000 s (< 3600).
Docker v3 ×2: skor 32.74462474110901 her ikisinde (deterministik), süre 225 / 213 s (final540 122 s) →
sunucu tahmini ≈ 2200–2300 s (< 3600, ~%37 pay). Ayrıntı submissions_log.md.

## Uzun bağlam (2026-11-11)
**TCN-long** (`gru_cv.TCNLongNet`: dilasyon 1–128, alıcı alan 511 pencere = 42.6 dk ≥ en uzun kayıt; 10 seed,
LOFO epoch = 10 her katta; 1833 s; `seq_cv_tcn_long.log`, `mix_gru_tcnlong.log`):

| Model | unseen | seen | Çift CV | Δ | [0,40) | [40,90) | [90,300) | [300,600] | bMAE | iyi kat |
|---|---|---|---|---|---|---|---|---|---|---|
| TCN-long tek | 43.03 | 40.60 | 41.81 | +1.53 | +0.41 | +0.63 | +0.31 | +0.18 | 102.74 | 0/10 |
| TCN-long + HGB(c), α 0.6 | 41.72 | 39.32 | 40.52 | +0.24 | −0.13 | −0.01 | +0.31 | +0.07 | 101.69 | 4/10 |
| (GRU+TCN-long)/2 + HGB(c), α 0.7 | 41.38 | 38.81 | 40.09 | −0.18 | −0.10 | −0.05 | −0.03 | −0.00 | 100.36 | 7/10 |

Karşılaştırma: kısa TCN (alıcı alan 127 pencere) tek 41.51, +HGB 40.44, GRU+TCN+HGB 40.00. **Alıcı alanı
büyütmek her yerde biraz kötü ve [300,600]'e hiçbir şey katmıyor.** 10 dk'dan uzun geçmiş dizi modeline
ek bilgi vermiyor (ya da 500 kayıtla öğrenilemiyor).

**"long" öznitelik grubu (HGB)**: `causal_features.py` "long" (28 çekirdek × {600 s eğim, 1200 s eğim, 600 s
ortalama − taban, 1200 s EMA} = 112 öznitelik; mevcut dyn/maxmin çıktıları HEAD ile bit düzeyinde aynı,
nedensellik testi geçti). B üyesi çift CV, aynı katlar/ağaç takvimi, referans = B on dyn_maxmin_spec_eye
(817 öznitelik); karışım = B + GRU-uw(10 seed), α LOFO (her katta 0.6 seçildi); `long_features.log`, 5101 s:

| Model | unseen | seen | Çift CV | Δ | [0,40) | [40,90) | [90,300) | [300,600] | bMAE | iyi kat |
|---|---|---|---|---|---|---|---|---|---|---|
| B (817, ref) | 44.94 | 41.47 | 43.20 | — | 8.57 | 8.27 | 15.49 | 10.88 | 106.10 | — |
| B + long (929) | 44.59 | 41.66 | 43.12 | −0.08 | −0.19 | −0.07 | +0.16 | +0.02 | 106.45 | 5/10 |
| B + GRU (ref), α 0.6 | 41.92 | 38.82 | 40.37 | — | 7.95 | 8.07 | 14.16 | 10.19 | 99.53 | — |
| B_long + GRU, α 0.6 | 41.74 | 38.86 | 40.30 | −0.07 | −0.07 | −0.04 | +0.03 | −0.00 | 99.54 | 7/10 |

Unseen'de küçük kazanç (−0.35 / −0.18), seen'de küçük kayıp (+0.19 / +0.04); hedeflenen [90,300) ve
[300,600] aralıklarında hiç iyileşme yok (+0.16 / +0.02). Katlar 5/10 ve 7/10 → gürültü düzeyinde.
**Sonuç: 10–20 dk ufuk ne dizi modelinde ne HGB'de işe yarıyor; [300,600] hatası uzun geçmişten değil
kişi düzeyindeki gecikme farkından geliyor (oracle bulgusuyla tutarlı).** GRU+TCN karışımına eklemeye
değmez (HGB tarafındaki −0.07 karışımda sıfıra iner; kabul kuralı eşiği 0.3).

## Kapatılan yönler (2026-11-11, kullanıcı kararı)
- **Deneğe özgü bilgi**: kimlik çıkarımda yok; örtük tanıma (ilk 2 dk %17) ile gerçekçi üst sınır ≈ −0.05.
  kNN örtük önsel DENENMEYECEK.
- **Uzun bağlam**: TCN-long (alıcı alan 42 dk) ve 10–20 dk "long" öznitelik grubu hedef aralıklarda sıfır
  kazanç. 10 dk'dan uzun geçmiş değersiz; [300,600] hatası kişi düzeyinde gecikme farkı.
- Bu iki yönde başka deney yapılmayacak.

## Sıradaki adım
Bu turda kabul kuralını geçen model yok. Yedek aday hâlâ (GRU+TCN)/2 + HGB(c) α 0.7 (40.00, Δ −0.28,
8/10). 18 Kasım'a kadar daha iyisi çıkmazsa 540 kayıtla eğitilip gönderilecek (TCN eğitimi
`train_mix_final.py`'ye, TCN çıkarımı `build_hgb_submission.py`'ye eklenmeli; yalnız torch).
mix3 (üçlü karışım, çift CV 39.40) 540 ile eğitildi, mix3_540.zip olarak **11 Kasım'da gönderildi (#2) →
sealed 38.66**. #3 mix3_540_gridpi 12 Kasım'da gönderildi (sealed 37.49); kalan 2 hak. Yedek aday (GRU+TCN)/2 + HGB(c) α 0.7 artık ikincil (mix3 ona göre de −0.60, 10/10).
Durum denetimi: `DURUM_RAPORU.md` (2026-11-11); karar katmanı deneyleri: `KARAR_KATMANI_RAPORU.md`.

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
