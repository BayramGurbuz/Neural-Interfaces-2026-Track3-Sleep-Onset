# B AŞAMASI RAPORU

Hedef: çift CV makroda 38.35'in (ızgaralı mix3, `mix3_540_grid` tarifi) altına inmek. Codabench'e hiçbir şey gönderilmedi.
Ölçüm düzeni her adımda aynı: unseen / seen / makro, kat eşleştirmeli fark, iyi kat sayısı, 4 bin katkısı; seçimler
(iç içe) LOFO ile; açık test (40 kayıt) seçimde kullanılmadı, sonda bir kez skorlandı; skorlar resmi `binned_mae`.
**Kanıt işaretleri:** [Ç] çalıştırılıp doğrulandı · [K] koddan okundu · [L] log / sonuç dosyası.

## Özet tablo (referans = ızgaralı mix3, 38.35)

| Adım | Varyant | unseen | seen | makro | Δ | İyi kat | Durum |
|---|---|---|---|---|---|---|---|
| ref | mix3 + ızgara π 0.86 | 39.74 | 36.96 | 38.35 | – | – | aday, gönderilmedi |
| B1a | π(D), 4 aralık (bins4) | 39.66 | 36.89 | 38.27 | −0.08 | 10/10 | builder'a seçenek olarak eklendi; tek başına ZIP yok (Δ < 0.5) |
| B1a | π(D), 2 aralık (step) | 39.65 | 36.89 | 38.27 | −0.08 | 9/10 | – |
| B1b | düz kanal kuralı | – | – | – | 0 (temiz veride çıktı aynı) | – | builder'a eklendi (varsayılan açık; eşik %50 → %90) |
| Adım 1 | **mix3_540_gridpi.zip** = B1a bins4 + B1b (%90) | 39.66 | 36.89 | 38.27 | −0.08 | 10/10 | **gönderildi (#3, 12 Kasım), sealed 37.49** (yerel −0.78); SHA-256 `4ef85c1f…0063`; 5 hakkın 3'ü kullanıldı, kalan 2 |
| B2a | B üyesi + bip (+432) | 43.85 | 41.28 | 42.56 | −0.64 (ref B 43.20) | 9/10 | **kazandırıyor**; (c) ve dizi modellerine taşıma onay bekliyor |
| B2b | B üyesi + events (+318) | 45.02 | 41.81 | 43.41 | +0.21 (ref B 43.20) | 4/10 | kazandırmıyor |
| B2c | B üyesi + epoch (+113) | 43.96 | 40.92 | 42.44 | −0.77 (ref B 43.20) | 10/10 | kazandırıyor; **ayrı tutuluyor** (ızgaraya bağlı) |
| B2a+c | B üyesi + bip + epoch (+545) | 43.29 | 40.95 | 42.12 | −1.08 (ref B 43.20) | 10/10 | c içeriyor; ayrı tutuluyor |
| B2 (c)+a | HGB(c) 4 üye, 817 + a (1249) | 43.59 | 40.75 | 42.17 | −0.48 (ref HGB(c) 42.65) | 9/10 | tek başına kazandırıyor |
| **B2 karışım** | HGB(a) + dizi(eski), π(D) | 39.51 | 36.88 | **38.19** | −0.08 | 5/10 | eşik (−0.5, 8/10) **geçilmedi** |
| B2 karışım | HGB(a) + dizi(a), π(D) | 39.51 | 37.18 | 38.34 | +0.07 | 5/10 | geçilmedi |
| B2 karışım | HGB(eski) + dizi(a), π(D) | 39.65 | 37.16 | 38.40 | +0.13 | 5/10 | geçilmedi |
| B2 karışım | HGB(a) + dizi(a+c), π(D) | 39.73 | 37.84 | 38.78 | +0.51 | 3/10 | c içeriyor; kötü |
| B3 | HMC evreleyici, Muse'a uygulandı | – | – | – | Spearman −0.10 / AUC 0.67 (en iyi öznitelik 0.42 / 0.83) | – | **durduruldu** |

---

## B1 — Karar katmanı inceltmeleri (yeniden eğitim yok)

### B1a — Onset'e bağlı ızgara ağırlığı π(D)

**Gözlem [Ç]:** açık veride ızgara oranı D'ye göre keskin değişiyor:

| D (sn) | ≤ 600 | 600–900 | 900–1000 | 1000–1100 | 1100–1200 | 1200–1300 | 1300–1400 | 1400–1490 | 1490–1500 |
|---|---|---|---|---|---|---|---|---|---|
| Kayıt | 227 | 133 | 44 | 32 | 35 | 24 | 26 | 16 | 3 |
| Izgarada | 1.000 | 1.000 | 0.591 | 0.594 | 0.714 | 0.542 | 0.423 | 0.438 | 0.667 |

Izgara dışı 77 kaydın onset'i 913.8–1492.6 sn; bu onset'ler 5 sn'nin katı bile değil (onset mod 5 değerleri 0.18–4.94).

**Yöntem [K]:** `scripts/b1_grid_pi.py`. Kalan sürenin her 1 sn hücresi için D = t + y; o hücrenin kütlesinin π(D) kadarı en yakın ızgara noktasına taşınıyor. π(D), katın **kendi eğitim kayıtlarında** o D aralığındaki ızgara oranı. Diğer her şey (ω, α 0.7, bileşen yapısı) gönderilen tarifle aynı. Referans motoru, kayıtlı ızgaralı tahminleri 1.5 × 10⁻⁵ sn farkla üretiyor [Ç].

Kat içi π değerleri [Ç]:
- step: D ≤ 900 için her katta 1.0; D > 900 için 0.55–0.65.
- bins4: (≤ 900, 900–1100, 1100–1300, > 1300) = (1.0, 0.56–0.68, 0.59–0.70, 0.40–0.58).

**Sonuçlar [Ç]** (log `~/neuralbench/save/cv/dual/b1_grid_pi.log`, sayılar `b1_grid_pi.json`):

| Varyant | unseen | seen | makro | Δ | İyi kat | Δ [0,40) | Δ [40,90) | Δ [90,300) | Δ [300,600] |
|---|---|---|---|---|---|---|---|---|---|
| sabit π 0.86 (ref) | 39.74 | 36.96 | 38.35 | – | – | 6.38 ¹ | 7.86 ¹ | 13.74 ¹ | 10.36 ¹ |
| step | 39.65 | 36.89 | 38.27 | −0.08 | 9/10 | −0.07 | −0.01 | −0.00 | −0.00 |
| bins4 | 39.66 | 36.89 | 38.27 | −0.08 | 10/10 | −0.07 | −0.01 | −0.00 | −0.00 |
| LOFO seçimi (step / bins4 karışık; tüm katlar: bins4) | 39.66 | 36.89 | 38.28 | −0.08 | 9/10 | −0.06 | −0.01 | −0.00 | −0.00 |

¹ Referans satırında mutlak katkılar; diğer satırlarda farklar.

Kat farkları (bins4): unseen −0.11, −0.06, −0.10, −0.12, −0.02; seen −0.18, −0.11, −0.00, −0.08, −0.02.

**Izgara dışı kayıtlardaki zarar [Ç]** (makro, kayıt ortalaması W-bMAE):

| | Izgarasız mix3 | Sabit π 0.86 | bins4 |
|---|---|---|---|
| Izgara dışı (68 + 83 kat-kaydı) | 43.74 | 44.85 (+1.11) | 44.32 (+0.58) |
| Izgarada | 38.74 | 37.34 (−1.39) | 37.34 (−1.40) |
| Onset ≤ 330 sn | 40.16 | 38.75 | 38.53 |

Izgara dışı kayıtlardaki zarar yarıya iniyor; ızgaradaki kayıtlardaki kazanç korunuyor.

**Açık test (sonda bir kez, π train+val'dan) [Ç]:** sabit π 46.61 → step 46.40 / bins4 46.39 (−0.22). Train+val'da bins4 π değerleri 1.0 / 0.603 / 0.643 / 0.488.

**Sonuç:** küçük ama tutarlı kazanç (−0.08, 10/10 kat); asıl değeri, mühürlü verinin ızgara oranı bilinmediği için ızgara dışı riskini yarıya indirmesi. 0.5 sn eşiğinin altında, tek başına ZIP hazırlanmadı. Builder'a seçenek olarak eklendi:
- Kullanım: `--grid-pi-cuts 900,1100,1300 --grid-pi-values …`.
- Akış ↔ toplu doğrulandı: 500-kayıt modeli, 40 kayıt / 5 869 pencere; maks fark 0.0002 sn; W-bMAE 46.3937 ↔ 46.3937 (B1a toplu skoru ile aynı) [Ç].
- 540 kayıtla bir sonraki adayda kullanılacak π değerleri, 540'ın tamamından hesaplanmalı: 1.0 / 0.592 / 0.644 / 0.444 [Ç].

### B1b — Düz (sabit) ama sonlu kanal

**Eşik seçimi [Ç]** (80 975 temiz pencere × 4 kanal):
- Kanal-pencere standart sapması en az 1.67 µV (kanal bazında: TP9 3.10, AF7 1.73, AF8 1.67, TP10 3.25).
- std < 1 µV olan pencere yok.
- Her pencerede örneklerin ≥ %99.8'i bir öncekinden farklı.

**Kural [K]:** `build_hgb_submission.py` `_clean`. Bir kanal şu iki koşuldan biri tutarsa geçersiz sayılıyor ve NaN'ın çoğunlukta olduğu kanalla aynı yoldan geçiyor (o kanalın son temiz penceresi; yoksa 0):
- pencere içi std < 0.1 µV,
- ya da ardışık örneklerin %50'sinden fazlası değişmiyor.

Eşik, gözlenen en küçük değerin ~1/17'si. `--flat-sd 0` kuralı tamamen kapatır.

**Senaryolar [Ç]:** 500-kayıt modeli `mix3_tv` (ızgarasız), ≥ 90 pencereli 10 açık-test kaydı, temiz W-bMAE 42.48. Hücreler: bozulmuş girdide W-bMAE farkı (sn) / temiz koşudan ortalama tahmin sapması (sn).

| Senaryo | Kural kapalı | Kural açık |
|---|---|---|
| 1 kanal düz, pencere 20–79 (TP10) | +9.98 / 132.2 | **+1.26 / 15.8** |
| Sıfır µV kanal, pencere 20–79 (TP10) | +10.07 / 131.6 | **+1.26 / 15.8** |
| Tüm kanallar düz, pencere 20–29 | +23.76 / 108.8 | **+0.04 / 7.8** |
| Takılı değer: pencerenin 2/3'ü sabit, 1 kanal, pencere 20–79 | +0.09 / 27.5 | +1.26 / 15.8 ² |
| 1 kanal düz, tüm kayıt (TP10) | +17.53 / 129.9 | +16.62 / 130.1 ³ |
| 1 kanal düz, tüm kayıt (AF7) | +8.38 / 67.7 | +8.31 / 67.6 ³ |
| NaN senaryoları (kısmi, tüm pencere, 10 ardışık, kanal NaN, inf) | önceki raporla aynı | aynı (değişmedi) |

² Kural kısmen takılı kanalı tamamen değiştiriyor: tahmin sapması azalıyor (27.5 → 15.8 sn) ama bu 10 kayıtta skor kötüleşiyor (+0.09 → +1.26).
³ Kayıt baştan düzse tutulacak temiz pencere yok; sonuç tümüyle NaN kanalla aynı.

Hiçbir senaryoda NaN tahmin, tam 0'a kilitlenme veya durumda NaN yok.

**Temiz veride değişiklik yok [Ç]:** gönderilen `submissions/mix3_540_grid` ile yeni builder'dan üretilen aynı tarif (π 0.86, düz kanal kuralı açık), 40 kayıt / 5 869 pencere: **maks \|eski − yeni\| = 0.0 sn**.

### Tekrarlanabilirlik (GENEL madde) [Ç]

`scripts/check_determinism.py`: TCN (seed 0, 10 epoch) ve CNN kodlayıcı (seed 0, 12 epoch) 540 kayıtla, `train_mix3_final.py --all` ile aynı şekilde ikişer kez eğitildi. Log: `~/neuralbench/save/check_determinism.log`.

| Bileşen | Ayar | Koşu 1 ↔ koşu 2, maks \|Δw\| | Gönderilen pakete göre |
|---|---|---|---|
| TCN | varsayılan | 0 | 0 (birebir) |
| TCN | deterministik | 0 | 0 (birebir) |
| CNN kodlayıcı | varsayılan | **0.363** | 0.34 / 0.38 |
| CNN kodlayıcı | deterministik | 0 | 0.39 |

- **Sonuç:** gönderilen `mix3_540` ve `mix3_540_grid` içindeki CNN kodlayıcı ağırlıkları bit düzeyinde yeniden üretilemez (deterministik olmayan cuDNN evrişimi). GRU, TCN ve HGB üretilebilir.
- **Etkisi [Ç]:** `train_mix3_final.py --tag=_det --deterministic` (500 kayıt) ile GRU, TCN ve HGB skorları eskisiyle 4 haneye kadar aynı (47.4421 / 47.1040 / 52.1836). Değişen yalnız CNN'li bileşen: GRU-emb 57.40 → 56.31; mix3 karışımı açık testte 47.06 → 46.86 (−0.20, tek örnek). Denetimde yeniden eğitimin bu mertebede oynaması beklenir.
- **Eklenen [K]:**
  - `scripts/repro.py` `set_deterministic()`: cudnn.deterministic, benchmark kapalı, `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
  - `train_mix3_final.py --deterministic`.
  - Bundan sonraki her eğitim betiği bu ayarla çalışacak.

### Değişen / eklenen dosyalar (B1)

| Dosya | Değişiklik |
|---|---|
| `scripts/b1_grid_pi.py` | yeni: π(D) deneyi |
| `scripts/build_hgb_submission.py` | düz kanal kuralı (`--flat-sd`, varsayılan 0.1 µV), π(D) seçeneği (`--grid-pi-cuts`, `--grid-pi-values`), inf uyarısının susturulması |
| `scripts/check_stream_grid.py` | π(D) desteği |
| `scripts/check_determinism.py`, `scripts/repro.py` | yeni: tekrarlanabilirlik testi ve deterministik ayarlar |
| `scripts/train_mix3_final.py` | `--deterministic` bayrağı |

---

## Adım 1 — Yeni aday: `mix3_540_gridpi.zip` (yeniden eğitim yok; kullanıcı 12 Kasım'da gönderdi, sealed 37.49)

**Tarif:**
- `mix3_540` ile aynı `model.joblib` (SHA-256 `265bb6b2…55d32`); yalnız `submission.py` farklı.
- Karar: ızgara π(D), bins4. Kesimler 900 / 1100 / 1300 sn; değerler 540 kaydın tamamından 1.0 / 0.592 / 0.644 / 0.444 [Ç].
- Girdi temizliği: NaN/inf yolu + düz kanal kuralı (std < 0.1 µV veya ardışık örneklerin **> %90**'ı değişmiyor).
- İçe aktarmalar `mix3_540` ile aynı (joblib, numpy, torch, scipy.signal) [Ç].
- Komut: `python scripts/build_hgb_submission.py --model ~/neuralbench/save/mix3_all/model.joblib --name mix3_540_gridpi --grid-pi-cuts 900,1100,1300 --grid-pi-values 1.0,0.592,0.644,0.444` (varsayılanlar: `--flat-sd 0.1 --flat-same 0.9`).
- **ZIP SHA-256: `4ef85c1feec2e0780b58342c2b8cc7d11a5f70469c8dd15fe0505d0b4b700063`** (13.23 MB).

**Beklenen etki (çift CV, B1a):** 38.35 → 38.27 (Δ −0.08, 10/10 kat); ızgara dışı kayıtlardaki zarar +1.11 → +0.58 sn. Açık test (500-kayıt modeli) 46.61 → 46.39.

**Düz kanal eşiği %90 [Ç]** (500-kayıt modeli, 10 kayıt, temiz 42.48; W-bMAE farkı / ortalama tahmin sapması):

| Senaryo | Kural kapalı | Eşik %50 (önceki) | Eşik %90 (şimdi) |
|---|---|---|---|
| Takılı değer: pencerenin 2/3'ü sabit, 1 kanal, 60 pencere | +0.09 / 27.5 | +1.26 / 15.8 | **+0.09 / 27.5** (tetiklenmiyor) |
| Takılı değer: pencerenin %95'i sabit, 1 kanal, 60 pencere | ölçülmedi | ölçülmedi | +1.26 / 15.8 (tetikleniyor) |
| 1 kanal düz, 60 pencere | +9.98 / 132.2 | +1.26 / 15.8 | +1.26 / 15.8 |
| Sıfır µV kanal, 60 pencere | +10.07 / 131.6 | +1.26 / 15.8 | +1.26 / 15.8 |
| Tüm kanallar düz, 10 pencere | +23.76 / 108.8 | +0.04 / 7.8 | +0.04 / 7.8 |

NaN / inf senaryoları değişmedi. Hiçbir senaryoda NaN tahmin, tam 0'a kilitlenme veya durumda NaN yok.

**Temiz veride sıfır fark [Ç]** (40 kayıt, 5 869 pencere, akış):
- (1) Gönderilmeyi bekleyen `mix3_540_grid` ↔ aynı tarif + yeni temizlik (%90): maks fark 0.0 sn.
- (2) `mix3_540_gridpi` ↔ aynı tarif, düz kanal kuralı kapalı: maks fark 0.0 sn.

**Akış ↔ toplu [Ç]:** maks \|Δ\| 0.0002 sn; W-bMAE (örneklem içi, 540 modeli) 31.3234 ↔ 31.3234.

**Docker `v3` [Ç]** (aynı oturumda, sırayla, makine boşken; log `~/neuralbench/save/mix3_540_gridpi_build.log`):

| ZIP | Koşu | wbmae_recording_mean | bMAE | MAE | duration |
|---|---|---|---|---|---|
| mix3_540_gridpi | 1 | 31.323388735064913 | 81.12 | 151.61 | 184.9 sn |
| mix3_540_gridpi | 2 | 31.323388735064913 | 81.12 | 151.61 | 176.1 sn |
| mix3_540_grid | 1 | 31.691708741918738 | 81.35 | 151.66 | 168.5 sn |

**Süre:**
- Aynı oturumda gridpi, grid'e göre +%4.5 ile +%9.7 arası (176–185 ↔ 168 sn).
- Pencere başına akış süresi (yerel, 40 kayıt) gridpi 25.0 ms. Grid'in aynı oturumda ayrı akış ölçümü yok; 12 Kasım'daki ölçüm 31.5 ms (makine yükü farklı).
- Oturumlar arası oynama büyük (grid 12 Kasım'da 230 / 218 sn, bugün 168 sn). Kabaca: gridpi, grid'den ≤ %10 yavaş.
- Mühürlü tahmin: mix3_540 için ≈ 2200–2300 sn × ≤ 1.1 → ≤ ~2500 sn (sınır 3600 sn). Mix3_540'ın mühürlü süresi kaydedilmediği için kaba.

---

## B2 — Yeni öznitelikler, B üyesi ablasyonu (gece zinciri, 12 Kasım 18:20–22:01)

**Kurulum [K]:**
- Kod: `scripts/features_b2.py` (gruplar), `scripts/b2_extract.py` (pencere öznitelikleri bir kez), `scripts/b2_ablation.py` (B üyesi), `scripts/b2_night.sh` (zincir).
- B üyesi tek başına; aynı sınıflar (11), parametreler, karar ve LOFO ağaç sayısı; çift CV aynı katlar.
- Referans: B'nin 817 öznitelikle kayıtlı fold-dışı tahminleri (`ens_dyn_maxmin_spec_eye_B`, çift CV 43.20; betik başında yeniden skorlanarak doğrulandı).
- Yeni pencere düzeyi gruplar akışa yalnız anlık + EMA 30 sn / 2 dk + bunların ilk-2-dk farkı olarak giriyor (×6).
- HGB CPU'da, `random_state=0` → deterministik. Log `~/neuralbench/save/cv/dual/b2_night.log`, sayılar `b2_ablation.json`.
- Kazandıran tanımı: Δ ≤ −0.15 ve ≥ 7/10 kat.

| Grup | Yeni öznitelik | unseen | seen | makro | Δ | İyi kat | Δ [0,40) | Δ [40,90) | Δ [90,300) | Δ [300,600] | Süre | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref B (817) | – | 44.94 | 41.47 | 43.20 | – | – | – | – | – | – | – | – |
| **a bip** | 432 (72 × 6) | 43.85 | 41.28 | **42.56** | **−0.64** | **9/10** | −0.12 | +0.10 | −0.10 | **−0.52** | 4635 sn | **kazandırıyor** |
| b events | 318 (53 × 6) | 45.02 | 41.81 | 43.41 | +0.21 | 4/10 | +0.12 | +0.06 | −0.05 | +0.07 | 4882 sn | kazandırmıyor |
| c epoch (ayrı tutuluyor) | 113 | 43.96 | 40.92 | **42.44** | **−0.77** | **10/10** | −0.24 | +0.27 | −0.51 | −0.28 | 3919 sn | kazandırıyor; taşıma kararı mühürlü skordan sonra |
| a + c birlikte (c içeriyor, ayrı tutuluyor) | 545 | 43.29 | 40.95 | **42.12** | **−1.08** | **10/10** | −0.32 | +0.19 | −0.53 | −0.42 | 5314 sn | kazandırıyor; etkiler kısmen toplanıyor (−0.64 + −0.77 = −1.41 yerine −1.08) |

Kat farkları [Ç]:
- a: unseen −1.78, −0.72, −1.00, −1.06, −0.88; seen +0.41, −0.02, −0.69, −0.52, −0.10.
- b: unseen +0.88, −0.44, −0.37, +0.12, +0.22; seen +0.67, −0.42, −0.24, +1.41, +0.27.
- c: unseen −1.33, −0.57, −1.16, −0.31, −1.53; seen −0.67, −0.81, −0.70, −0.31, −0.27.
- a+c: unseen −2.86, −1.22, −1.66, −1.26, −1.22; seen −0.24, −0.76, −0.95, −0.15, −0.48.

**Gözlemler:**
- a, şimdiye kadar hiçbir şeyin kıpırdatamadığı [300,600] aralığında kazanç veriyor (−0.52). Kazanç unseen'de daha büyük (−1.09 ortalama); seen'de karışık (+0.41 ile −0.69 arası).
- b kazandırmadı. Olay sıklıkları [Ç]:
  - iğcik pencerelerin %2.4–2.6'sında;
  - 75 µV üstü yavaş dalga %12–14'ünde;
  - kırpma (eşik 6 MAD) pencere başına 0.17, (eşik 4 MAD) 0.33;
  - yavaş göz hareketi 0.16.
  - Yorumum: kırpma oranının düşük olması, gözler kapalı yatarken beklenen bir durum; eşiği düşürmek (4 MAD) sayıyı yalnız iki katına çıkarıyor.
- c'nin kazancı [90,300) ve [0,40)'tan geliyor, [40,90) kötüleşiyor. Kazanç 30 sn ızgarasının mühürlü veride tutmasına bağlı olabilir (faz, kayıt başından hizalı); ayrı tutuldu.
- a ve b'den yalnız a kazandırdığı için zincir a+b birleşimini koşmadı. a+c birlikte ayrı bilgi olarak ölçülüyor (aşağıda, c yine ayrı).

**Pencere başına çıkarım maliyeti, a grubu [Ç]** (mikro ölçüm, tek iş parçacığı, arka planda ablasyon koşarken):
- bip pencere öznitelikleri 1.05 ms (yüksüz ölçüm `b2_extract.py`: 0.63 ms);
- akış güncellemesi 0.013 ms;
- binleme 817 → 1249 öznitelik: 0.28 → 0.41 ms.
- Toplam ≈ +1.2 ms/pencere ≈ **+%4–5** (bugünkü 25–32 ms'ye göre). HGB ağaç sayısı değişmiyor; dizi modellerinde yalnız giriş katmanı büyüyor (125 → 269).
- b grubu (kazandırmadı) 2.72 ms (yüksüz).

---

## B3 — Dış veri: bilgi toplama (indirme YAPILMADI)

Kaynaklar: PhysioNet sayfaları (HMC v1.1, Sleep-EDF Expanded v1.0.0), kurulu `neuralfetch` yükleyicileri ve MNE'nin
paketle gelen Sleep-EDF kayıt tablosu (`mne/datasets/sleep_physionet/age_records.csv`). [L] = sayfadan / koddan okundu.

| | HMC Sleep Staging | Sleep-EDF Expanded, Sleep Cassette (SC) |
|---|---|---|
| Sürüm | 1.1 (18 Mart 2022) | 1.0.0 (sayfadaki sürüm geçmişinde 2018'deki "Version 2" genişlemesi de anılıyor) |
| İndirme boyutu | ZIP 12.9 GB; açılmış 15.7 GB | Tüm veri kümesi 8.1 GB; SC klasörü ~7.2 GB (153 PSG × ~47 MB; sayfada toplam yok, tahmin) |
| Lisans | CC BY 4.0 | Open Data Commons Attribution v1.0 (ODC-By) |
| Kayıt / kişi | 151 tüm gece PSG (85 E / 66 K); v1.1'de 3 kayıt çıkarılmış (sayfada son sayı yok) | 153 kayıt / 78 kişi (yükleyici: SC; 2 gece/kişi, bazı eksik) |
| Kanallar | EEG F4/M1, C4/M1, O2/M1, C3/M2; EOG E1/M2, E2/M2; çene EMG; EKG | EEG Fpz-Cz, Pz-Oz; yatay EOG; çene EMG (1 Hz RMS zarfı); olay işaretçisi |
| Örnekleme | 256 Hz | EEG/EOG 100 Hz |
| Uyku evreleri | AASM (W, N1, N2, N3, R); EDF+ ve metin | R&K (W, R, 1, 2, 3, 4, M, ?); NeuralBench 3+4 → N3 birleştiriyor |
| Uyku başlangıcı bölümü | **var**: kayıtlar "lights off"–"lights on" işaretleri arasına kırpılmış (ışık kapanışından itibaren) | **var, ama ~20 saatlik kayıt**: gündüz uyanıklığı da içeriyor; ışık kapanma saati MNE tablosunda her kayıt için var (ör. SC4001: 00:38, SC4002: 21:57); yerelde doğrulandı [Ç] |
| Kapalı klinik veri mi | hayır (PhysioNet açık erişim; retrospektif klinik PSG) | hayır (açık erişim) |
| NeuralBench desteği | `neuralfetch.studies.Alvarez2022Haaglanden`; `_sleep_onset_stream/datasets/alvarez2022haaglanden.yaml` (`max_pre_n2_s 1200`, val/test'te rastgele başlangıç) | `Kemp2000Analysis` (yalnız sleep-cassette); `kemp2000analysis.yaml` (aynı ayarlar) |
| Atıf | Alvarez-Estevez & Rijsman (2022), PhysioNet, doi:10.13026/t79q-fr32; ve PLoS ONE 16(8):e0256111 (2021) | Kemp ve ark. (2000), IEEE-BME 47(9):1185–1194; PhysioNet atfı |

**Muse eşlemesi:**
- HMC'de F4-M1 ve C3-M2 türevleri, önerdiğin eşleme için uygun (F4-M1 ≈ AF8−TP9, C3-M2 ≈ AF7−TP10 yaklaşık).
- Sleep-EDF'de yalnız Fpz-Cz ve Pz-Oz var; ikisi de Muse türevlerine zayıf benziyor. Fpz-Cz, AF7−AF8 / AF7−TP9'a göre farklı geometride.
- Yorumum: eşleme açısından HMC daha değerli.

**Uyku başlangıcı odağı:**
- NeuralBench'in hazır görevleri N2'den önceki en fazla 1200 sn'yi kullanıyor (`AddSleepOnsetTargets`, `max_pre_n2_s`). Muse'da bu süre en fazla 1500 sn.
- HMC kayıtları ışık kapanışından başladığı için bu bölüm doğrudan var.
- Sleep-EDF'de ışık kapanma saatiyle kesmek gerekir; NeuralBench yükleyicisi bunu yapmıyor, kendisi N2'den 1200 sn geri gidiyor.

**Risk — disk alanı [Ç]:** WSL diski (`/`, ext4) 922 GB boş gösteriyor, ama `Ubuntu` dağıtımının sanal diski `C:\Users\bayra\AppData\Local\wsl\{856b…}` altında (Windows kayıt defteri, Lxss `BasePath`) ve **C:'de 9.4 GB boş yer var**; G:'de 8.9 GB (PowerShell `Get-PSDrive`). HMC (12.9 GB ZIP + 15.7 GB açılmış) ve Sleep-EDF SC (~7.2 GB) bu alana sığmaz; indirme C:'yi doldurup WSL'i çökertebilir. İndirmeden önce:
- ya C:'de yer açılmalı,
- ya da WSL sanal diski başka sürücüye taşınmalı.

İkisi de 30 GB sınırının altında. Alan ölçüldü [Ç]; çökme riski bir çıkarım.

### B3 — HMC, 30 kayıt kayıt kayıt indirildi (12 Kasım 20:16–23:59) [Ç]

**Yöntem:** `scripts/b3_hmc_fetch.py` (`nice 19`, tek iş parçacığı).
- 151 kayıttan `default_rng(0)` ile 30 kayıt seçildi (liste: `docs/external_data.md`).
- Her dosya PhysioNet `SHA256SUMS.txt` ile doğrulandı; diskte aynı anda en fazla bir ham kayıt bulundu, kırpmadan hemen sonra silindi.
- Saklanan: ışık kapanışından ilk N2 + 600 sn'ye kadar; F4-M1, C3-M2, C4-M1, E1-M2, E2-M2; 128 Hz, µV, float32; 30 sn evre etiketleri.
- Kayıt bazında ayrıntı: `~/neuralbench/data/hmc/fetch_log.json`.

| Ölçü | Değer |
|---|---|
| Kayıt | 30 / 30 başarılı, atlanan yok; 30'unda da "Lights off" işareti var; özgün örnekleme 256 Hz |
| İndirme süresi | toplam 13 618 sn (3.8 saat); kayıt başına medyan 200 sn, en kısa 10 sn, en uzun 1890 sn; ham toplam 3.3 GB → ortalama 0.24 MB/sn |
| Sunucu hataları | 3 kez 502 Bad Gateway, tekrar denemeyle aşıldı; ilk sürümde bir kez çökme (tekrar deneme yoktu), bir kez boş kontrol listesi (yerel kopyaya geçildi) |
| Kayıt başına saklanan | medyan 3.48 MB (1.90–10.53 MB); toplam 136 MB |
| Işık kapanışından ilk N2'ye | medyan 758 sn; %10–90: 254–2625 sn; en kısa 140, en uzun 3510 sn; 9 kayıtta > 1500 sn (Muse'da en fazla 1500 sn) |

Muse'a göre daha uzun uykuya dalma süreleri (klinik popülasyon) evreleyici için sorun değil, çünkü evreleyici kalan süreyi değil evreyi öğreniyor.
151 kayda genişletme kestirimi (aynı hızla): ~19 saat indirme, ~690 MB saklama.

---

## SABAH RAPORU — 13 Kasım (B2 taşıma, karışım, B3 akıl sağlığı testi)

### 1. HGB(c) + a, çift CV (CPU, 12 Kasım 23:29 → 13 Kasım 06:23, 25 640 sn) [Ç]

- Komut: `scripts/b2_ens.py bip`. B üyesi dünkü ablasyondan aynı ayarlarla yeniden kullanıldı; kalan üç üye `ensemble_dist.member_probas` ile.
- Önbellek: `ens_dyn_maxmin_spec_eye_bip_*`.
- Sonuç: 4 üye (c), 1249 öznitelik: unseen 43.59, seen 40.75 → **42.17**. Eski HGB(c) (817) 42.65'e göre **−0.48**, 9/10 kat.
- Katkı farkları: [0,40) −0.11, [40,90) −0.03, [90,300) −0.07, [300,600] −0.27.
- Üye süreleri: B_h1 ~11.7k sn, B_h2 5.0k, B2 8.9k.

### 2. Dizi modelleri, a ve a+c girdili (GPU, `--deterministic`, 22:14 → 01:59) [Ç]

`scripts/b2_seq_cv.py`, log `b2_seq.log`, sayılar `b2_seq.json`. Her biri tek başına, B sınıflarında, kendi ω'sıyla:

| Model | Eski (62 → 125 girdi) | a (269) | a+c (495) |
|---|---|---|---|
| GRU-uw ×10 | 41.43 | **40.99** (−0.44) | 42.25 (+0.82) |
| TCN ×10 | 41.51 | 41.41 (−0.10) | 42.01 (+0.50) |
| GRU-emb ×3 | 44.18 | 44.58 (+0.40) | 43.77 (−0.41) |

- Süreler: GRU 2017 / 4015 sn, TCN 1859 / 3605 sn, GRU-emb 772 / 1355 sn. Deterministik ayar ve paralel HGB yükü nedeniyle kat başına süre öncekinin ~2.5 katı.
- Seçilen epoch'lar ekiyle aynı aralıkta (GRU 10/15, TCN 10, GRU-emb 5–15).

### 3. Karışım değerlendirmesi (yeniden eğitim yok) [Ç]

**Yöntem:** `scripts/b2_mix.py`, log `b2_mix.log`, sayılar `b2_mix.json`.
- Karar mix3_540_gridpi ile aynı: π(D) bins4 kat içi, ω, bileşen yapısı. α iç içe LOFO ile {0.5 … 0.9}'dan seçildi.
- Her satır ızgarasız kararla da skorlandı.
- Referans (mix3 + π(D), α 0.7) kayıtlı tahminlerden 1.5 × 10⁻⁵ sn farkla yeniden üretildi.

| Satır | α (LOFO) | unseen | seen | makro | Δ | İyi kat | Δ [0,40) | Δ [40,90) | Δ [90,300) | Δ [300,600] | Izgarada | Izgara dışı |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref: HGB(eski) + dizi(eski), π(D) | 0.7 | 39.66 | 36.89 | 38.27 | – | – | – | – | – | – | 37.34 | 44.32 |
| **HGB(a) + dizi(a)**, π(D) | 0.6 | 39.51 | 37.18 | 38.34 | +0.07 | 5/10 | +0.01 | −0.08 | +0.11 | +0.02 | 37.19 | 45.55 |
| **HGB(a) + dizi(a+c)**, π(D) | 0.6 | 39.73 | 37.84 | 38.78 | +0.51 | 3/10 | −0.37 | +0.08 | +0.56 | +0.25 | 37.70 | 45.61 |
| **HGB(eski) + dizi(a)**, π(D) | 0.6–0.7 | 39.65 | 37.16 | 38.40 | +0.13 | 5/10 | +0.01 | −0.12 | +0.03 | +0.21 | 37.24 | 45.73 |
| HGB(a) + dizi(eski), π(D) (ek) | 0.7 | 39.51 | 36.88 | **38.19** | −0.08 | 5/10 | +0.00 | −0.01 | +0.05 | −0.12 | 37.26 | 44.17 |
| HGB(eski) + dizi(a+c), π(D) (ek) | 0.6 | 39.89 | 37.82 | 38.85 | +0.58 | 2/10 | −0.35 | +0.02 | +0.52 | +0.40 | 37.71 | 46.01 |

Izgarasız kararla tüm satırlar 39.29–40.00 aralığında (ref 39.40); sıralama aynı.

Kat farkları:
- HGB(a) + dizi(a): unseen −0.45, −0.70, −0.51, +0.18, +0.72; seen +0.96, +0.20, −0.18, −0.06, +0.51.
- HGB(a) + dizi(eski): unseen −0.56, −0.28, −0.37, +0.31, +0.14; seen +0.21, −0.09, −0.33, +0.10, +0.08.

**Sonuç: hiçbir satır kabul eşiğini (Δ ≤ −0.5, ≥ 8/10) geçmiyor.**
- En iyisi HGB(a) + dizi(eski): −0.08, 5/10.
- HGB'nin tek başına −0.48'lik kazancı karışımda neredeyse tamamen kayboluyor.
- dizi(a), GRU'yu tek başına iyileştirse de (−0.44) karışımı iyileştirmiyor.
- Yorumum (ayrıca sınanmadı): a'nın taşıdığı bilgi, karışımda dizi modellerinin zaten sağladığı bilgiyle örtüşüyor.
- Izgara dışı kayıtlarda a'lı satırlar daha kötü (44.32 → 45.5–45.7, dizi(a) içerenlerde).

**c'nin etkisi (dizi girdisi a+c eksi a; makro farklar, pozitif = kötü):**

| HGB | Karar | Tümü | Izgarada | Izgara dışı | c'li daha iyi kat |
|---|---|---|---|---|---|
| a | π(D) ızgaralı | +0.44 | +0.50 | +0.06 | 1/10 |
| a | ızgarasız | +0.40 | +0.43 | +0.19 | 2/10 |
| eski | π(D) ızgaralı | +0.45 | +0.47 | +0.28 | 1/10 |
| eski | ızgarasız | +0.47 | +0.50 | +0.31 | 2/10 |

- c dizi modellerine girdi olarak eklendiğinde, ızgara kararı olsun olmasın, hem ızgaradaki hem ızgara dışı kayıtlarda karışımı kötüleştiriyor.
- Yani "ızgaranın üstüne eklenme" ya da "örtüşme" sorusuna bu deney yanıt vermiyor: c dizi tarafında zaten zararlı.
- c'nin B üyesindeki −0.77'si yalnız HGB tarafında görüldü. HGB(c) için a+c kümesi, talimat gereği koşulmadı.
- Bileşen düzeyinde [Ç]: GRU ve TCN'de c zararlı (+0.82 / +0.50 tek başına); GRU-emb'de yararlı (−0.41).

### 4. B3 akıl sağlığı testi — DURDURULDU [Ç]

**Yöntem:** `scripts/b3_stager.py`, log `~/neuralbench/save/b3_stager.log`, sayılar `b3_stager.json`.
- 30 HMC kaydı (ışık kapanışından N2 + 10 dk'ya), 5 sn pencereler.
- Türevler F4-M1 (→ AF8−TP9), C3-M2 (→ AF7−TP10), E1−E2 (→ AF7−AF8).
- 36 ölçekten bağımsız spektral öznitelik × 6 nedensel akış = 216.
- Küçük HGB (7 yaprak, 150 ağaç, sınıf dengeli), sınıflar W / N1 / N2+ (N2, N3, R).

**HMC'nin kendi içinde:** kayıt bazlı 5 kat CV'de dengeli doğruluk 0.703, N2+ ↔ diğer AUC 0.967. Evreleyici HMC'de çalışıyor.

**Muse train+val'a (500 kayıt, 75 106 pencere) uygulandı; açık test kullanılmadı:**

(1) Kalan süreye göre ortalama olasılıklar:

| Kalan süre (sn) | [0,30) | [30,60) | [60,120) | [120,180) | [180,300) | [300,450) | [450,600] |
|---|---|---|---|---|---|---|---|
| Pencere | 3 000 | 3 000 | 5 994 | 5 976 | 11 532 | 12 756 | 32 848 |
| P(N2+) | 0.071 | 0.049 | 0.035 | 0.029 | 0.035 | 0.046 | 0.057 |
| P(N1) | 0.303 | 0.268 | 0.221 | 0.187 | 0.159 | 0.148 | 0.158 |
| P(W) | 0.626 | 0.683 | 0.744 | 0.784 | 0.806 | 0.806 | 0.785 |

P(N2+) onset'e yaklaşırken yükselmiyor: düz ve monoton değil, 450–600 sn önce son 30 sn'dekiyle benzer. P(N1) düzenli yükseliyor.

(2)–(3) Mevcut özniteliklerle karşılaştırma:

| Skor | Spearman (kalan süre) | AUC (son 60 sn ↔ ≥ 300 sn önce; 6 432 / 45 604 pencere) |
|---|---|---|
| Evreleyici P(N2+) | −0.103 | 0.669 |
| Evreleyici P(N1) (ek) | −0.185 | 0.719 |
| Evreleyici P(N1) + P(N2+) (ek) | −0.142 | 0.699 |
| En iyi tek mevcut öznitelik | 0.421 (`TP10_maxabs_ema120s_rel`) | 0.829 (`TP10_theta_alpha_ema30s_minus_runmin`) |
| Geçen süre (karşılaştırma için) | −0.440 | 0.794 |

**Karar:** üç ölçünün üçünde de evreleyici mevcut en iyi tek öznitelikten açıkça kötü, geçen süreden bile kötü. Kural gereği B3 durduruldu; 151 kayda genişletme yapılmadı.
- Yorumum (sınanmadı): HMC'nin ıslak elektrotlu klinik PSG'si ile Muse'un kuru elektrotlu ev kaydı arasındaki alan farkı ve türev eşlemesinin yaklaşık olması (C3-M2 ≠ AF7−TP10) evreleyiciyi taşınamaz kılıyor.
- İndirilen 30 kırpılmış kayıt (136 MB) `~/neuralbench/data/hmc/crops/` altında duruyor. Kullanıma girmediği için beyanda "denendi, modelde kullanılmadı" olarak işaretlenmeli.

### 5. Pencere başına çıkarım süresi (güncel) [Ç]

Yerel CPU, makine boşken, 13 Kasım 06:55; tek iş parçacığı mikro ölçümler + tam `predict`.

| Parça | ms/pencere |
|---|---|
| Gönderilen mix3_540_gridpi, tam `predict` (300 pencere, tek akış, 6 torch iş parçacığı) | 34.1 |
| Aynı ZIP, 40 kayıtlık akış kontrolü (12 Kasım) | 25.0 |
| a grubu ek: bip öznitelikleri | 0.95 |
| a grubu ek: akış güncellemesi | 0.01 |
| a grubu ek: binleme 817 → 1249 | +0.34 (0.33 → 0.67) |
| **a grubu toplam ek** | **≈ 1.3 (+%4–5)** |
| c grubu ek: epoch akışı | 0.02 |

Ölçümler arası oynama (25 ↔ 34 ms) koşu koşullarından; bileşen ekleri bu oynamanın altında. Kabul eşiğini geçen tarif olmadığı için yeni ZIP hazırlanmadı.

## 14 Kasım — Kural değişikliği sonrası "paketleme düzenliliği" taraması (yalnız liste; hiçbir şey kaldırılmadı)

Tarama: gönderilen `submissions/mix3_540_gridpi/submission.py` ve eğitim/öznitelik kodu (`causal_features.py`, `gru_cv.py`,
`dist_hgb.py`, `ensemble_dist.py`, `stream_utils.py`, `features_b2.py`, `n2det_*.py`, `decision_layer.py`, `b1_grid_pi.py`).

**Yasak kapsamına giren (kaldırılacak / yeni adaylarda yok):**
1. 30 sn onset ızgarası — wrapper `GRID_PI`, `GRID_PI_CUTS / VALUES`, `grid_tables / grid_slots / slot_median / _grid_decide`
   (`submission.py` 364–413, 577, 608–613, 767–772): "onset = kayıt başından 30 sn'nin katı" varsayımı + geçen sürenin 5 sn'nin
   katı olması (`abs(el - 5*round(el/5))`). Gönderilen #3 (gridpi) ve gönderilmeyen mix3_540_grid bunu kullanıyor. mix3_540 (#2)
   kullanmıyor (GRID kapalı).
2. π(D) tablosu (`b1_grid_pi.pi_table`, `decision_layer.exact_table / to_slots`): onset'in 30 sn ızgarasında olma oranı, eğitim
   onset'lerinden; yalnız CV/karar katmanında.
3. B2 "epoch" (c) grubu (`features_b2.B2Stream(epoch_group=True)`, `b2_ablation.b2_table(("epoch",))`): geçen süre mod 30 fazı
   ve 30 sn epoch hizalı özetler. Yalnız deneylerde; hiçbir gönderimde yok. İş 1 (HGB a+c) iptal edildi.

**Şüpheli görülebilecek ama bence eğitim-kümesi tanımı / serbest girdi (kaldırılmadı, karar sende):**
4. N2 dedektörü eğitim kümesi: pozitifler açık eğitim kayıtlarının onset sonrası 300 sn'si (`n2det_extract.py`: her ham dosya
   onset'ten tam 300 sn sonra bitiyor, `assert x.shape[1] - o == N_POST*WIN`), negatifler kalan ≥ 300 sn (`n2det_cv.NEG_MIN`).
   Çıkarımda yalnız o pencerenin EEG'si kullanılır; "dosya onset'ten 300 sn sonra biter" bilgisi yalnız etiketli pencere seçimi.
5. Geçen süre: HGB `elapsed_s` özniteliği (`CausalFeatureStream.n_seen * step_s`), dizi girdisi `log1p(min(t*5, 1500))`
   (`gru_cv.ELAPSED_CLIP`), wrapper `self.gru_t * 5.0` — akışın ilk penceresinden itibaren sayılıyor; FAQ'ya göre serbest.
   NOT: `gru_t * 5.0` ve `n_seen * step_s` pencere uzunluğunu 5 sn varsayıyor; `meta`/pencere boyutundan geliyor (step_s), GRU
   tarafında sabit 5.0 (eğitimle aynı).
6. İlk-2-dk tabanı (`gru_baseline_n = 24` pencere) ve 30 s / 2 dk / 5 dk EMA'lar, 10/20 dk eğimler (`DYN_EMA_S`,
   `LONG_WINDOWS_S`): akış başlangıcına göre nedensel; düzenlilik değil ama akış ortadan başlarsa dağılımı kayar →
   `stream_start_test.py` ile ölçülüyor.
7. Sınıf sınırları `VARIANTS["B"]` ve hedefin 600'de kırpılması: etiket tanımı (serbest, senin notun).
8. Dağılım kararı: sınıf içi temsilci = eğitimdeki aralık içi medyan (`class_stats`), ω = metrik ağırlıkları: etiket dağılımından,
   paketleme düzenliliği değil.
9. Giriş temizliği eşikleri (std < 0.1 µV, > %90 değişmeyen örnek): sinyal özelliği.

Kayıt uzunluğu / kayıt sonu / dosya adı / denek kimliği hiçbir yerde kullanılmıyor (wrapper yalnız `meta["sfreq"]`,
`meta["ch_names"]` ve batch'leri okuyor).

## 14 Kasım — Akış başlangıcına dayanıklılık (`scripts/stream_start_test.py`, yeniden eğitim yok)

Train+val mix3 paketi (500 kayıtla eğitilmiş GRU/TCN/GRU-emb + HGB(c)), 40 açık test kaydı (örneklem dışı; yalnız görülmüş
kişiler → unseen/seen ayrımı yok). Çift CV kat modelleri diske yazılmadığı için kat bazlı ölçüm yeniden eğitim gerektirir.
Her kaydın ilk k penceresi atılıp akış oradan başlatıldı (geçen süre, EMA'lar, taban, GRU/TCN durumu sıfırdan); hedefler aynı.
Skor: aynı pencereler üzerinde kesik akış ↔ tam geçmiş (resmi W-bMAE, kayıt ortalaması). k=0 kayıtlı 47.06'yı yeniden üretiyor;
yeniden akıtılan 817 öznitelik önbellekle bire bir.

| k | pencere | tam geçmiş | kesik | Δ (karışım) | Δ, geçen süre gerçek değerle yamalı | HGB-only Δ (yamalı) | dizi-only Δ (yamalı) | Δ katkı [0,40) / [40,90) / [90,300) / [300,600] |
|---|---|---|---|---|---|---|---|---|
| 6 (30 s) | 5629 | 47.20 | 51.77 | **+4.57** | +3.20 | +7.41 (+7.50) | +3.45 (+1.33) | −0.98 / +2.92 / +2.48 / +0.15 |
| 24 (2 dk) | 4922 | 46.87 | 52.95 | **+6.08** | +0.68 | +26.29 (+24.95) | +5.27 (+0.31) | +1.05 / +5.78 / +1.94 / −2.69 |
| 60 (5 dk) | 3569 | 40.70 | 80.82 | **+40.11** | +3.24 | +123.31 (+115.89) | +31.11 (+2.72) | +19.15 / +15.57 / +7.55 / −2.16 |
| rastgele [0, onset−60 s], tohum 0 (medyan 55, maks 216) | 3251 | 43.99 | 78.09 | **+34.09** | +8.59 | +83.80 (+73.75) | +26.24 (+7.28) | +12.89 / +16.63 / +5.95 / −1.37 |

Okuma: bozulma eşiğin (+0.5) çok üstünde. Dizi modellerinde bozulmanın kaynağı büyük ölçüde geçen-süre girdisi (gerçek
değerle yamalanınca +26 → +7). HGB'de ise yamalama işe yaramıyor (+84 → +74): 817 öznitelikteki ilk-2-dk farkları, koşan
maks/min'e göre göreli öznitelikler ve 5–20 dk eğim/EMA'lar akışın başında "kayıt başı" imzası taşıyor; model bunu zamanın
göstergesi olarak öğrenmiş. Yakın onset bin'leri ([0,40), [40,90)) en çok zarar görüyor (kesik akışta model uzak tahmin ediyor).
Bağlam: mix3_540'ın yeni skorlamada 37.79 (yerel 39.40'tan iyi) alması, mühürlü akışların hâlâ kayıt başından başladığını
düşündürüyor; rastgele başlangıç olsaydı +30 civarı bozulma görünürdü. Karar kullanıcıda (öneri: kayıt başı kırpma artırması).

## 14 Kasım — İş 2 sonucu: HGB(c) + n2det ve ızgarasız karışım (ref ızgarasız mix3 39.40 ↔ mühürlü 37.79)

HGB(c) 4 üye + 5 CNN-dedektör özeti (822 öznitelik, fold-dışı P): **40.76** (unseen 41.61 / seen 39.91), HGB(c) 817'ye
(42.65) göre **−1.89, 9/10 kat**; Δ katkı −0.31 / +0.51 / −1.60 / −0.49. Süre 4.9 saat (3 üye).

Karışım, ızgarasız karar, α iç içe LOFO ile 0.3–0.8 ızgarasından (`b2_mix.py ... nogrid`, `OUT/b2_mix_n2nogrid.json`):

| Satır | unseen | seen | makro | Δ | iyi kat | Δ katkı [0,40) / [40,90) / [90,300) / [300,600] | α (kat başına) |
|---|---|---|---|---|---|---|---|
| ref: HGB(eski)+dizi(eski), α 0.7 | 40.73 | 38.07 | 39.40 | – | – | – | 0.7 ×10 |
| HGB(n2det)+dizi(eski) | 40.18 | 37.75 | **38.97** | **−0.43** | **8/10** | −0.06 / +0.19 / −0.41 / −0.15 | 0.6 ×10 |
| HGB(n2det)+dizi(n2det) | 40.06 | 38.63 | 39.34 | −0.06 | 6/10 | +0.04 / +0.51 / −0.56 / −0.04 | 0.6 ×10 |
| HGB(n2det)+yalnız GRU n2det | 40.04 | 38.03 | 39.03 | −0.37 | 8/10 | 0.00 / +0.35 / −0.50 / −0.22 | 0.6 ×6, 0.7 ×4 |

Kat Δ (en iyi satır): unseen [−1.31, −0.21, −0.56, −0.17, −0.46], seen [+0.22, −0.70, −0.50, −0.78, +0.15].
**Kabul kuralı geçilmedi** (−0.5 & 8/10 ya da −0.3 & 10/10). α 0.7 → 0.6: HGB ağırlığı 0.3 → 0.4 (HGB artık dizi
modellerine yaklaştı → HGB'nin akış başlangıcı duyarlılığı karışımda daha ağır basar). dizi(n2det) görülmüş kişilerde yine
zararlı (+0.56). Kazanç [90,300) ve [300,600]'de; [40,90) kötüleşiyor.

**Teşhis (`scripts/stream_start_diag.py`, k=6 ve k=24; kesik akış girdilerinde bir öznitelik grubu tam geçmiş değeriyle
yamalanıyor; geri kazanım = Δ'nın ne kadarı kapanıyor):**

| HGB-only | k=6 (Δ +7.41) | k=24 (Δ +26.29) |
|---|---|---|
| elapsed | −0.09 | +1.34 |
| ham pencere | 0.00 | 0.00 |
| ilk-2-dk farkı (`_rel`) | −1.08 | +13.71 |
| EMA 30 s / 2 dk / 5 dk | +2.31 | +6.74 |
| EMA − koşan maks/min | **+5.27** | **+19.28** |
| eğimler / 5 dk var / ac1 | +2.64 | +5.35 |
| 10 dk EMA | **+5.76** | **+16.06** |

Gruplar örtüşür (toplam > Δ); kümülatif yamalama tam skora döner. Dizi-only: k=6 kesik 51.19 → elapsed yamalı 49.07, taban yamalı
49.19, ikisi 46.47; k=24 kesik 51.43 → elapsed 46.46, taban 41.57, ikisi 41.30. Sonuç: HGB'de "kayıt başı" imzası uzun bellekli
özniteliklerde (koşan ekstremumlar, 10 dk EMA, ilk-2-dk farkı); dizi tarafında ilk-2-dk tabanı + elapsed. Kırpma artırması
ikisini de hedefliyor; elapsed'i tek başına yamalamak HGB'yi kurtarmıyor.

## 14 Kasım — Dizi kırpma artırması (GPU, deterministik, eski 62 girdi; `b2_seq_cv.py old ARCH --crop`)

Kırpma dağılımı (kullanıcı): %50 yok, %35 k∈[1,12], %15 k∈[13, onset−60 s]; her epoch ve kayıt için yeniden; girdiler yeni
başlangıçtan kurulur (`gru_cv.run_fold cfg["crop"]`). Çift CV k=0, artırmasız deterministik referansa göre:
GRU 41.43 → **40.85 (−0.58, 6/10)**, TCN 41.51 → 42.32 (+0.81, 3/10), GRU-emb 44.18 → 44.52 (+0.34, 4/10); hepsinde yakın
bin'ler iyileşir, [300,600] kötüleşir. LOFO epoch'lar uzadı (GRU 15 → 20, TCN 10 → 15).

Izgarasız karışım (ref 39.40, α LOFO 0.3–0.8; `b2_mix_cropmix.json`):

| Satır | unseen | seen | makro | Δ | iyi kat | Δ katkı | α |
|---|---|---|---|---|---|---|---|
| HGB(n2det) + dizi(üçü artırmalı) | 40.24 | 38.34 | 39.29 | −0.11 | 6/10 | −0.06 / +0.15 / −0.51 / +0.31 | 0.5 |
| **HGB(n2det) + yalnız GRU artırmalı (TCN, GRU-emb eski)** | 39.93 | 37.79 | **38.86** | **−0.54** | **8/10** | −0.11 / +0.10 / −0.53 / +0.01 | 0.6 ×10 |
| HGB(eski) + dizi(üçü artırmalı) | 40.85 | 38.67 | 39.76 | +0.36 | 2/10 | | 0.7 |
| HGB(eski) + yalnız GRU artırmalı | 40.47 | 38.08 | 39.27 | −0.13 | 6/10 | | 0.7 |

Kat Δ (kalın satır): unseen [−1.37, −0.28, −0.70, −0.33, −1.33], seen [+0.42, −0.95, −0.26, −0.70, +0.08].
**KABUL KURALINI GEÇEN İLK TARİF** (Δ ≤ −0.5 ve 8/10). HGB(n2det)+dizi(eski) 38.97 (−0.43) ile fark −0.11.

**Dayanıklılık testi, artırmalı diziyle (`stream_start_test.py --bundle=mix3crop` / `--gru-from=mix3crop`; train+val
modeller, 40 test kaydı; Δ = kesik akış − aynı pencerelerde tam geçmiş, karışım):**

| k | artırmasız (mix3) | üçü artırmalı (mix3crop) | yalnız GRU artırmalı | HGB-only (artırmasız, ortak) |
|---|---|---|---|---|
| 1 (5 s) | +0.37 | +0.08 | −0.10 | +1.17 |
| 3 (15 s) | +2.25 | +0.22 | +0.63 | +3.38 |
| 6 (30 s) | +4.57 | **+2.82** | +3.53 | +7.41 |
| 12 (60 s) | +5.36 | +2.91 | +5.64 | +25.65 |
| 24 (2 dk) | +6.08 | +0.85 | +4.50 | +26.29 |
| 60 (5 dk) | +40.11 | +5.47 | +15.53 | +123.31 |
| rastgele | +34.09 | +4.21 | +14.34 | +83.80 |

k=0 test skorları (örneklem dışı, görülmüş kişiler): mix3 47.06, mix3crop 44.27, yalnız-GRU 46.16. Dizi-only kesik Δ (k=6):
3.45 → 1.65 (üçü) / 2.12 (GRU). **Kullanıcının aday-2 eşiği (k=6 zararı < +1.0) hiçbir bileşimde sağlanmadı**; kalan zararın
kaynağı artırmasız HGB (k=6'da tek başına +7.4; karışımda ağırlığı 0.3–0.4). HGB artırması B-üyesi ön kontrolü
(`scripts/hgb_crop_precheck.py`) aday-1 final HGB eğitiminden sonra CPU'da otomatik başlayacak.
