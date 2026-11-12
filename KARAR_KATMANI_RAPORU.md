# KARAR KATMANI RAPORU — mix3 üzerinde ağırlık, 30 sn ızgarası, yumuşatma

Tarih: 2026-11-12. Hiçbir model yeniden eğitilmedi; Codabench'e hiçbir şey gönderilmedi. Tüm deneyler mix3 tarifinin kayıtlı fold-dışı **sınıf olasılıkları** üzerinde yapıldı (olasılıklar kayıtlıydı; yeniden üretim gerekmedi).

**Sonuç:** 30 sn ızgarası tek başına eşiği geçiyor (makro −1.05 sn, 10/10 kat). Tam metrik ağırlıkları ve yumuşatma kazandırmıyor; ızgarayla birleşimleri ızgaranın tek başına halinden kötü. Eşik aşıldığı için `submissions/mix3_540_grid.zip` hazırlandı (aynı ağırlıklar, yalnız karar katmanı + NaN düzeltmesi); **gönderilmedi**.

Skorlar W-bMAE, saniye, düşük iyi; resmi `binned_mae` ile hesaplandı. Δ = referansa göre makro fark. "İyi kat" = kat eşleştirmeli, 10 katın kaçında referanstan iyi.

| Varyant | unseen | seen | makro | Δ | İyi kat |
|---|---|---|---|---|---|
| Referans (mix3, gönderilen karar) | 40.73 | 38.07 | 39.40 | – | – |
| **D1** tam ağırlık w(t, y), bileşen bazında | 41.10 | 38.33 | 39.71 | +0.32 | 2/10 |
| D1 tam ağırlık, olasılıklar önce karıştırılmış | 42.00 | 39.48 | 40.74 | +1.34 | 0/10 |
| D1 tam ağırlık, α iç içe LOFO | 41.14 | 38.36 | 39.75 | +0.35 | 2/10 |
| (kontrol) eski ağırlık, olasılıklar önce karıştırılmış | 41.71 | 39.25 | 40.48 | +1.08 | 1/10 |
| **D2** ızgara π = 0.5 | 39.83 | 37.18 | 38.50 | −0.90 | 10/10 |
| **D2 ızgara π = 0.86** (ZIP'teki varyant) | **39.74** | **36.96** | **38.35** | **−1.05** | **10/10** |
| D2 ızgara π = 1.0 | 39.77 | 36.96 | 38.36 | −1.04 | 10/10 |
| D2 ızgara π = kat içi eğitimdeki ızgara oranı (0.855–0.885) | 39.75 | 36.97 | 38.36 | −1.04 | 10/10 |
| D2 ızgara, π iç içe LOFO | 39.77 | 37.00 | 38.38 | −1.02 | 10/10 |
| D2 basit: nokta tahminini ızgaraya çekme | 39.86 | 37.21 | 38.54 | −0.86 | 10/10 |
| **D3** yumuşatma a = 0.2 | 40.81 | 38.13 | 39.47 | +0.07 | 1/10 |
| D3 yumuşatma a = 0.4 | 41.04 | 38.35 | 39.70 | +0.30 | 0/10 |
| D3 yumuşatma a = 0.6 | 41.71 | 38.99 | 40.35 | +0.95 | 0/10 |
| D3 yumuşatma a = 0.8 | 44.70 | 41.84 | 43.27 | +3.87 | 0/10 |
| D3 yumuşatma, a LOFO (her katta a = 0 seçildi) | 40.73 | 38.07 | 39.40 | 0.00 | 0/10 |
| **D4** 1+2, π = 0.86 (LOFO da 10/10 katta 0.86) | 40.01 | 37.17 | 38.59 | −0.81 | 9/10 |
| D4 1 + basit çekme | 40.23 | 37.48 | 38.85 | −0.54 | 8/10 |
| D4 1+3, a LOFO (a = 0 → D1 ile aynı) | 41.10 | 38.33 | 39.71 | +0.32 | 2/10 |
| D4 1+2+3, ızgara → yumuşatma, (π, a) LOFO (0.86, 0 seçildi) | 40.01 | 37.17 | 38.59 | −0.81 | 9/10 |
| D4 1+3 + çekme, yumuşatma → çekme, a LOFO (çoğunlukla 0.2) | 40.22 | 37.49 | 38.85 | −0.55 | 8/10 |
| D4 1+2+3, ızgara → yumuşatma → çekme, (π, a) LOFO | 40.01 | 37.20 | 38.61 | −0.79 | 9/10 |
| (ek) 2+3, ızgara → yumuşatma, (π, a) LOFO (a = 0 seçildi) | 39.77 | 37.00 | 38.38 | −1.02 | 10/10 |

**Kanıt işaretleri:** **[Ç]** çalıştırılıp doğrulandı · **[K]** koddan okundu · **[L]** log / sonuç dosyası.
Kaynaklar: `scripts/decision_layer.py`, log `~/neuralbench/save/cv/dual/decision_layer.log`, sayılar `…/decision_layer.json`, varyant tahminleri `…/declayer_*_preds.npz` [Ç].

---

## 1. Kurulum ve doğrulamalar

| Kontrol | Sonuç | Kanıt |
|---|---|---|
| Olasılıklar kayıtlı mı | evet: `gru_{uw_n10,tcn_n10,cnn3_emb}_{unseen,seen}_proba.npz` (11 sınıf) ve `ens_dyn_maxmin_spec_eye_{B,B_h1,B_h2,B2}.npz` (üye olasılıkları) | [Ç] |
| Referansın yeniden üretimi | karar motoru (1 sn hücreler + 0…600 sn atomlar, 1201 "slot") referans tahminlerini maks 1.5 × 10⁻⁵ sn farkla üretiyor; makro 39.40 | [Ç] log "engine check" |
| Skorlama | her kat resmi `binned_mae` + `group_scores` ile; yerel kodla eşitlik her skorda `assert` edildi | [K] `decision_layer.py` `evaluate`; [Ç] |
| Seçim düzeni | π, a, α: iç içe LOFO (`mix_calib.nested_select`; dışarıdaki katın kayıtları diğer 9 kattan maskelenir). Açık test seçimde kullanılmadı | [K] `decision_layer.py` `select` |
| Katlar | çift CV ile aynı (5 unseen + 5 seen, 500 kayıt) | [K] |

### Tam ağırlık w(t, y) ↔ resmi metrik [Ç]

`w(t, y) = W_bin(y) / (n(D) · S(D))`, `D = t + y`; pencereler 5, 10, …, ⌊D/5⌋·5'te biter [K `decision_layer.py` `range_counts`, `w_exact`]. Doğrulama: 6 gerçek kayıtta her pencerenin tahmini tek tek 1 sn bozuldu, resmi `binned_mae`'deki değişim (= o pencerenin ağırlığı) formülle karşılaştırıldı.

| Kayıt | Onset (sn) | Pencere | maks \|resmi − w\| |
|---|---|---|---|
| sub-147 / ses-001 (en kısa) | 90.00 | 18 | 0 |
| sub-057 / ses-003 (en uzun) | 1500.00 | 300 | 0 |
| sub-019 / ses-001 (ızgara dışı) | 913.80 | 182 | 0 |
| sub-122 / ses-004 (ızgara dışı) | 1492.56 | 298 | 2.7 × 10⁻²⁰ |
| sub-162 / ses-004 | 330.00 | 66 | 0 |
| sub-009 / ses-001 | 630.00 | 126 | 0 |

Her kayıtta ağırlıklar toplamı 1.000000. y = 600 nokta sınıfının ağırlığı, kat içi eğitim kayıtlarından D ≥ t + 600 olanlar üzerinden ortalama alınarak hesaplandı (yoksa D = t + 600).

---

## 2. Ayrıntılı sonuçlar

Referans değerleri: bin katkıları [0,40) 7.69, [40,90) 7.74, [90,300) 13.60, [300,600] 10.37; onset ≤ 330 sn kayıtlar 40.16 (unseen 43.62 / 59 kayıt, seen 36.69 / 68 kayıt); ızgaradaki kayıtlar 38.74 (432 / 482), ızgara dışı 43.74 (68 / 83).

### Bin katkıları, kısa kayıtlar, ızgara içi / dışı (makro; Δ referansa göre)

| Varyant | Δ [0,40) | Δ [40,90) | Δ [90,300) | Δ [300,600] | Onset ≤ 330 (Δ) | Izgarada Δ | Izgara dışı Δ |
|---|---|---|---|---|---|---|---|
| D1 tam ağırlık, bileşen bazında | −0.03 | +0.01 | +0.98 | −0.65 | 46.52 (+6.37) | +0.40 | −0.03 |
| D1 tam ağırlık, önce karışım | −0.72 | +0.05 | +1.84 | +0.18 | 43.60 (+3.45) | +1.20 | +2.23 |
| (kontrol) eski ağırlık, önce karışım | −0.72 | +0.01 | +0.88 | +0.91 | 37.56 (−2.59) | +0.89 | +2.09 |
| D2 π = 0.5 | −0.95 | −0.01 | +0.07 | −0.01 | 39.09 (−1.06) | −1.12 | +0.48 |
| **D2 π = 0.86** | −1.31 | +0.12 | +0.15 | −0.01 | 38.75 (−1.41) | −1.39 | +1.11 |
| D2 π = 1.0 | −1.41 | +0.20 | +0.18 | −0.01 | 38.53 (−1.63) | −1.43 | +1.40 |
| D2 π LOFO | −1.26 | +0.11 | +0.14 | −0.01 | 38.77 (−1.38) | −1.35 | +1.06 |
| D2 basit çekme | −0.72 | −0.12 | −0.02 | −0.01 | 39.26 (−0.89) | −1.19 | +1.15 |
| D3 a = 0.2 | −0.13 | +0.13 | +0.01 | +0.06 | 40.35 (+0.19) | +0.07 | +0.11 |
| D3 a = 0.4 | −0.28 | +0.37 | +0.05 | +0.16 | 40.80 (+0.64) | +0.28 | +0.39 |
| D3 a = 0.6 | −0.38 | +0.83 | +0.15 | +0.36 | 41.91 (+1.75) | +0.91 | +1.20 |
| D3 a = 0.8 | +0.30 | +2.11 | +0.53 | +0.93 | 45.59 (+5.43) | +3.78 | +4.48 |
| D4 1+2 π = 0.86 | −1.35 | +0.13 | +1.11 | −0.69 | 44.58 (+4.42) | −1.08 | +1.08 |
| D4 1 + basit çekme | −0.74 | −0.11 | +0.96 | −0.66 | 45.52 (+5.36) | −0.80 | +1.19 |
| D4 1+3 + çekme (yumuşatma → çekme) | −0.93 | +0.02 | +0.96 | −0.60 | 45.65 (+5.49) | −0.81 | +1.27 |
| D4 1+2+3 (ızgara → yumuşatma → çekme) | −1.37 | +0.16 | +1.09 | −0.66 | 44.70 (+4.54) | −1.11 | +1.38 |

### Seçilen varyant (D2, π = 0.86), rejim bazında [Ç]

| | Referans | D2 π = 0.86 | Fark |
|---|---|---|---|
| unseen, ızgaradaki kayıtlar (432) | 40.14 | 38.84 | −1.30 |
| seen, ızgaradaki kayıtlar (482) | 37.33 | 35.85 | −1.48 |
| unseen, ızgara dışı (68) | 45.11 | 46.24 | +1.13 |
| seen, ızgara dışı (83) | 42.37 | 43.46 | +1.09 |
| unseen, onset ≤ 330 sn (59) | 43.62 | 42.30 | −1.32 |
| seen, onset ≤ 330 sn (68) | 36.69 | 35.19 | −1.50 |

Kat bazında fark (sn): unseen −0.72, −0.82, −0.98, −1.13, −1.28; seen −0.96, −1.30, −1.37, −0.91, −1.00.

Bu iki grup ortalamasından doğrusal hesapla: ızgaradaki kayıt oranı g iken beklenen fark ≈ −1.39·g + 1.11·(1 − g) sn; sıfır noktası g ≈ 0.44. Açık veride g = 0.857 (463/540).

### Deney bazında gözlemler

**Deney 1 — tam ağırlıklar işe yaramadı.** [300,600] katkısı düşüyor (−0.65) ama [90,300) daha çok kötüleşiyor (+0.98) ve zarar kısa kayıtlarda toplanıyor (onset ≤ 330: +6.37). Yorumum (ölçülmedi): tam ağırlık, bir aralığın alt sınırına yakın adaylara (ör. kalan süre 90–120 sn; öyle bir kayıtta o aralıkta az pencere olur) çok yüksek ağırlık veriyor; modelin dağılımı ise geniş sınıflar içinde düzgün kabul edildiği için bu ince yapıyı taşımıyor. "Önce karıştır, sonra ağırlıkla" yapısı ağırlıktan bağımsız olarak kötü (+1.08, eski ağırlıkla bile).

**Deney 2 — ızgara işe yaradı.** Kazancın tamamı [0,40) aralığından (−1.31); diğer aralıklar ±0.15 içinde. π 0.75–1.0 arasında sonuç düz (38.35–38.38); LOFO katlarda 0.75 / 0.86 / 1.0 seçti, tüm katlar birlikte 0.86. Basit çekme kazancın çoğunu veriyor (−0.86). Izgara dışı kayıtlarda zarar +1.1 sn (π = 0.86), π = 0.5 ile +0.48 sn.

**Deney 3 — yumuşatma işe yaramadı.** Her a değerinde kötü; LOFO her katta a = 0 seçti. [0,40) biraz iyileşiyor, [40,90) daha çok kötüleşiyor.

**Deney 4 — birleşimler.** Tam ağırlık içeren her birleşim, ızgaranın tek başına halinden kötü (en iyisi −0.81, 9/10). Sıra: "ızgara → yumuşatma" LOFO'da a = 0'a iniyor (= 1+2); "yumuşatma → çekme" a = 0.2 seçiyor ama daha kötü (−0.55). Eski ağırlıkla 2+3 de a = 0 seçiyor (= Deney 2).

### Açık test (40 kayıt, 500-kayıt modeli, sonda bir kez) [Ç]

Parametreler 10 katın tamamı üzerinden seçilen değerler. 40 kaydın 31'i ızgarada. Referansın yeniden üretimi: kayıtlı test tahminleriyle maks fark 0.055 sn.

| Varyant | W-bMAE | Δ |
|---|---|---|
| Referans | 47.06 | – |
| D1 tam ağırlık, bileşen bazında | 50.05 | +2.99 |
| D1 tam ağırlık, önce karışım | 49.40 | +2.34 |
| **D2 ızgara π = 0.86** (LOFO seçimi de 0.86) | 46.61 | −0.45 |
| D2 basit çekme | 46.69 | −0.37 |
| D3 yumuşatma (a = 0) | 47.06 | 0.00 |
| D4 1+2 π = 0.86 | 49.85 | +2.79 |
| D4 1 + basit çekme | 49.65 | +2.59 |
| D4 1+3 + çekme (a = 0.2) | 49.51 | +2.45 |
| D4 1+2+3 ızgara → yumuşatma → çekme | 50.10 | +3.04 |

Yön çift CV ile aynı; ızgaranın açık testteki kazancı (−0.45) çift CV'dekinden (−1.05) küçük.

---

## 3. NaN hata düzeltmesi

**Kusur [Ç]:** gönderilen `mix3_540`'ta ham pencere temizlenmeden CNN kodlayıcıya giriyordu; NaN gömme GRU-emb durumuna ve gömme tabanına yerleşip kaydın kalan tahminlerini 0'a kilitliyordu.

**Düzeltme [K]:** `scripts/build_hgb_submission.py` → `HGBStreamModel._clean`, `predict`'in ilk satırında çağrılıyor; öznitelikler, CNN, tüm durumlar yalnız temizlenmiş pencereyi görüyor. Durumu (`last_win`) `reset_state` sıfırlıyor.

### Strateji karşılaştırması [Ç]

500-kayıt modeli (`mix3_tv`, açık test için örneklem dışı), ≥ 90 pencereli 10 test kaydı, temiz W-bMAE 42.48. Hücreler: bozulmuş girdide W-bMAE farkı (sn).

| Senaryo | Düzeltmesiz | NaN → 0 | Son geçerli örnek | Son temiz pencereyi tut |
|---|---|---|---|---|
| Kısmi: 50 NaN örnek, 1 kanal, 1 pencere | +36.47 | +0.01 | +0.03 | +0.03 |
| Tüm pencere NaN (1 pencere) | +36.47 | −1.34 ¹ | −1.34 ¹ | +0.04 |
| Art arda 10 NaN pencere | +36.47 | +23.72 | +23.76 | +0.04 |
| 1 kanal NaN, 60 pencere (TP10) | +36.47 | +10.07 | +9.98 | +1.26 |
| 1 kanal tümüyle NaN, tüm kayıt (TP10) | +40.50 | +16.62 | +16.62 | +16.62 |
| 1 kanal tümüyle NaN, tüm kayıt (AF7) | +40.50 | +8.31 | +8.31 | +8.31 |
| inf: 20 örnek +inf, 1 pencere | 0.00 | 0.00 | +0.01 | +0.01 |

¹ Skor tesadüfen düşmüş; tahminler temiz koşudan ortalama 41 sn, en çok 295 sn sapıyor (tutma stratejisinde 1.6 / 38 sn).

"0" ile "son geçerli örnek" pratikte aynı: pencere ortalaması çıkarıldığı için tümüyle NaN bir pencere ikisinde de düz sinyale dönüyor ve log güçler −6'ya çöküp EMA'ları, tabanı ve GRU durumunu bozuyor. **Seçilen:** bir kanalın örneklerinin yarısından fazlası geçersizse o kanalın son temiz penceresi kullanılır (henüz yoksa 0); azı geçersizse son geçerli örnek taşınır.

### Düzeltilmiş kodla senaryolar [Ç]

`mix3` (500-kayıt) ağırlıkları, ızgara kapalı, aynı 10 kayıt; "durumda NaN" = gömme tabanı, GRU / GRU-emb gizli durumları, öznitelik tabanı, TCN geçmişi, EMA'lar.

| Senaryo | W-bMAE farkı | Temiz koşudan ort. / maks sapma (sn) | Tam 0 tahmin | NaN tahmin | Durumda NaN |
|---|---|---|---|---|---|
| Kısmi: 50 NaN örnek, 1 kanal, pencere 20 | +0.03 | 0.33 / 10.6 | 0 | 0 | yok |
| Tüm pencere NaN (pencere 20) | +0.04 | 1.60 / 38.2 | 0 | 0 | yok |
| Art arda 10 NaN pencere (20–29) | +0.04 | 7.84 / 190.1 | 0 | 0 | yok |
| İlk pencere tümüyle NaN | −1.99 ² | 74.12 / 350.6 | 0 | 0 | yok |
| 1 kanal NaN, pencere 20–79 (TP10) | +1.26 | 15.79 / 180.0 | 0 | 0 | yok |
| 1 kanal tümüyle NaN, tüm kayıt (TP10) | +16.62 | 130.13 / 388.2 | 0 | 0 | yok |
| 1 kanal tümüyle NaN, tüm kayıt (AF7) | +8.31 | 67.58 / 375.4 | 0 | 0 | yok |
| 1 kanal uzun süre düz, pencere 20–79 (TP10) | +9.98 | 132.17 / 466.5 | 0 | 0 | yok |
| inf: 20 örnek +inf, pencere 20 | +0.01 | 0.18 / 13.0 | 0 | 0 | yok |

² Skor tesadüfen düşmüş; tahminler büyük ölçüde değişiyor.

**Düzeltmenin çözmediği durumlar:** (a) kayıt en baştan geçersizse tutulacak temiz pencere yok; sıfır pencere ilk-2-dk tabanına giriyor (ilk pencere NaN: ortalama sapma 74 sn; kanal baştan sona NaN: +8 / +17 sn). (b) Düz ama sonlu sinyal temizlenmiyor: NaN değil, tespit edilmiyor (+9.98 sn). Bunlar için kod yazılmadı.

**Temiz veride değişiklik yok [Ç]:** gönderilen `submissions/mix3_540` ile yeni builder'dan üretilen aynı model (ızgara kapalı), 40 test kaydı / 5 869 pencere, akış: **maks \|eski − yeni\| = 0.0 sn**.

---

## 4. Hazırlanan gönderim: `submissions/mix3_540_grid.zip` (GÖNDERİLMEDİ)

Eşik koşulu: makro kazanç ≥ 1.0 sn ve ≥ 8/10 kat → D2 π = 0.86: −1.05 sn, 10/10. π = 0.86, 10 katın tamamı üzerinden seçilen değer (540 kayıttaki ızgara oranı 0.857).

| Başlık | Değer | Kanıt |
|---|---|---|
| Komut | `python scripts/build_hgb_submission.py --model ~/neuralbench/save/mix3_all/model.joblib --name mix3_540_grid --grid-pi 0.86` | [Ç] |
| İçerik | `submission.py` (37 176 bayt), `model.joblib` (22 092 725 bayt); zip 13.23 MB, SHA-256 `f4e60d85…7dab0` | [Ç] |
| Ağırlıklar | `model.joblib`, gönderilen `mix3_540` ile bayt düzeyinde aynı (SHA-256 `265bb6b2…55d32`) | [Ç] |
| Kod farkı | yalnız karar katmanı (`GRID_PI = 0.86`, `grid_tables`, `grid_slots`, `slot_median`, `_grid_decide`) ve `_clean`; içe aktarmalar aynı (joblib, numpy, torch, scipy.signal) | [Ç] `diff` |
| Güvenlik dalı | pencere bitişi 5 sn'nin katı değilse veya olasılıklar sonlu değilse eski karar kullanılır | [K] `mix3_540_grid/submission.py` `predict` |
| Akış ↔ toplu (540 modeli) | 40 kayıt, 5 869 pencere: maks \|Δ\| 0.0009 sn; W-bMAE 31.6917 ↔ 31.6917 (örneklem içi) | [L] `~/neuralbench/save/mix3_540_grid_build.log` |
| Akış ↔ toplu (500 modeli) | maks \|Δ\| 0.0006 sn; 46.6122 ↔ 46.6122 (= deney tablosundaki açık test skoru) | [Ç] |
| Docker (`v3`, CPU, `docker_test_submission.sh`) | iki koşu, aynı skor: `wbmae_recording_mean` 31.691708741918738, bMAE 81.35, MAE 151.66 | [L] aynı log |

### Süre karşılaştırması [Ç]

| Ölçüm | mix3_540 (gönderilen) | mix3_540_grid |
|---|---|---|
| Docker `v3` toplam, açık test | 210.7 sn (bu oturum); 225.3 / 213.1 sn (11 Kasım) | 230.4 sn / 218.1 sn |
| Pencere başına, yerel akış (40 kayıt) | 32.5 ms (11 Kasım logu) | 31.5 ms |

Fark, koşular arası oynaklığın (≈ ±%4) içinde. Örneklem içi açık test skoru 32.74 → 31.69.

**Yapılamayan:** `v4-sealed` imajıyla test (imaj yerelde yok); Docker testleri `v3` ile.

---

## 5. Sınırlar ve açık noktalar

- Kazanç, mühürlü veride onset'lerin kayıt başlangıcına göre 30 sn'nin katı olmasına bağlı. Bu oran mühürlü veri için bilinmiyor; ızgara dışı kayıtlarda ölçülen zarar +1.1 sn.
- Açık verideki 77 ızgara dışı kaydın tamamı onset > 900 sn (`report_bins.py` notu: "probably cropped starts").
- Karar, pencere bitişlerinin kayıt başından itibaren 5 sn'nin katı olduğunu varsayıyor (açık veride ilk pencere bitişi her kayıtta 5.0 sn); değilse eski karara düşüyor.
- π LOFO seçimi katlar arasında 0.75–1.0 arasında oynuyor; bu aralıkta skor farkı ≤ 0.03 sn.
- Çift CV'deki −1.05 sn, 500 kayıtlık kat modellerinin olasılıklarıyla ölçüldü; 540-modeli için fold-dışı ölçüm yok.
- Deney 1'deki mekanizma açıklaması bir yorumdur; ayrı bir deneyle sınanmadı.

## 6. Değişen / eklenen dosyalar

| Dosya | Değişiklik |
|---|---|
| `scripts/decision_layer.py` | yeni: dört deney + açık test |
| `scripts/check_stream_grid.py` | yeni: ızgara kararı için akış ↔ toplu kontrolü |
| `scripts/build_hgb_submission.py` | `_clean` (NaN düzeltmesi), `--grid-pi`, `--out-root`, `--no-zip`; `--grid-pi 0` iken temiz veride çıktı eskisiyle aynı |
| `submissions/mix3_540_grid/`, `submissions/mix3_540_grid.zip` | yeni aday (gönderilmedi) |
| `docs/status.md`, `docs/submissions_log.md` | mix3_540 mühürlü 38.66 (11 Kasım, kalan hak 3); yeni aday kaydı |

Geçici test betikleri ve çıktıları proje dışında: `/tmp/claude-1000/-home-bayram-projects-muse-sleep-onset/6d989f9b-f95d-4204-a4da-760f937e76bd/scratchpad/` (`nan_strategies.py`, `nanfix_test.py`, `fix_tests.out`, `nan_strategies.out`).
