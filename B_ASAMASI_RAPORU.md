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
| Adım 1 | **mix3_540_gridpi.zip** = B1a bins4 + B1b (%90) | 39.66 | 36.89 | 38.27 | −0.08 | 10/10 | **gönderildi (#3, 12 Kasım), sonuç bekleniyor**; SHA-256 `4ef85c1f…0063`; 5 hakkın 3'ü kullanıldı, kalan 2 |

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

## Adım 1 — Yeni aday: `mix3_540_gridpi.zip` (yeniden eğitim yok; kullanıcı 12 Kasım'da gönderdi, sonuç bekleniyor)

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
