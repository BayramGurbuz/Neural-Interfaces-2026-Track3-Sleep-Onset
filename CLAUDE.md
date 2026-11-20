> Yeni oturumda önce docs/status.md ve docs/submissions_log.md dosyalarını oku.

# Muse Sleep-Onset — EEG/EMG Foundation Challenge 2026, Track 03

Bu proje, NeurIPS 2026 Brain & Body Workshop kapsamındaki EEG/EMG Foundation Challenge 2026'nın
Track 03 (Sleep Onset) yarışması için model geliştirme deposudur.

## Benimle çalışma şekli
- Benimle Türkçe konuş. Kod, değişken adları ve commit mesajları İngilizce olabilir.
- Her adımı yapmadan önce kısaca ne yapacağını ve neden yapacağını açıkla; ben öğrenmek istiyorum.
- Büyük değişikliklerden önce plan çıkar ve onayımı bekle.
- Her çalışan adımdan sonra git commit öner.
- Emin olmadığın competition kuralını tahmin etme; docs/ altındaki dosyalara bak veya bana sor.

## Ortam
- Windows 11 + WSL2 (Ubuntu), proje WSL dosya sisteminde (~/projects/...), /mnt/c altında DEĞİL.
- GPU: NVIDIA RTX 3050 (sınırlı VRAM). Küçük/orta modeller tercih et; REVE gibi ~69M parametreli
  modellerin tam fine-tuning'i bu kartta gerçekçi değil.
- Python >= 3.12, neuralbench==0.4.0 (eski sürümlerde _sleep_onset_stream yok).
- NeuralBench ayarları: ~/.neuralbench/config.json (DATA_DIR, CACHE_DIR, SAVE_DIR). Yerel çalışma, SLURM yok.
- Veri, cache ve checkpoint'ler git'e girmez (.gitignore).

## Görev
- Girdi: Muse kafa bandı, 4 kanal (TP9, AF7, AF8, TP10), 128 Hz, mikrovolt, ek filtre/ölçekleme yok.
- Kayıt 5 saniyelik pencereler halinde, kronolojik sırayla, birer birer gelir.
- Çıktı: her pencere için ilk N2 epoch'una kalan saniye, 600 s ile sınırlı.
  Hedef = clip(n2_onset - window_stop, 0, 600).
- Model yalnızca mevcut ve geçmiş pencereleri kullanabilir, gelecek pencereleri asla.
  Model durumu her kayıt başında sıfırlanır (reset_state).

## Puanlama (sealed final)
- W-bMAE (saniye, düşük iyi). Gerçek kalan süreye göre aralıklar:
  [0,40) x10, [40,90) x5, [90,300) x3, [300,600] x1.
- Her kayıt içinde hesaplanır, kayıtlar üzerinden ortalanır; görülmüş kişiler (yeni geceler) ve
  hiç görülmemiş kişiler ayrı skorlanır, final skor ikisinin eşit ağırlıklı ortalaması.
- Yerel metrik: test/wbmae_stream_mean (seen/unseen ayrımı YOK). Açık test setinde görülmemiş kişi
  yok, bu yüzden yerel skor final skoru öngörmez. Kendi doğrulamamızı KİŞİ BAZINDA ayırarak yap.

## Veri
- Açık veri: Interaxon2026Muse (NEMAR nm000287 v1.0.0, CC-BY-NC-SA-4.0), ~1.1 GB,
  540 kayıt / 203 kişi; sağlanan split 500 train / 40 test.
- NeuralBench görevi: `neuralbench eeg _sleep_onset_stream` (yarışma için bunu kullan,
  `sleep_onset` değil).
- KRİTİK TUZAK: açık veride her N2 başlangıcı kayıt bitişinden tam 300 s önce. Kayıt uzunluğu veya
  kaydın sonuna dair hiçbir bilgi özellik olarak kullanılmamalı (leakage).
- Kayıt genelinde hesaplanan istatistikler (tüm kayıt üzerinden fit edilen scaler vb.) kullanılamaz;
  ön işleme pencere başına veya geçmişe dayalı (causal) olmalı.

## Gönderim (Codabench, competition 17983)
- ZIP: `submission.py` (içinde `class Solver(CompetSolver)`) + ağırlık dosyaları.
- Değerlendirme yalnızca inference; worker hiçbir paket kurmaz. submission.py yalnızca worker
  image'ında olan paketleri import edebilir; kendi ek Python modüllerimiz desteklenmez.
  Mimari ve ön işleme tek dosyada olmalı.
- Docker image: sunucu sealed'da lionelyneuro/neural-compet:v4-sealed kullanıyor (Codabench sayfa başlığı);
  bu imaj herkese açık değil (Docker Hub'da yalnız v2/v3). Yerel testler v3 ile sürer. KURAL: final540 bu
  imajda sorunsuz çalıştı; yeni modellere final540'ın kullanmadığı HİÇBİR kütüphane/import eklenmez
  (numpy, torch, joblib, scipy'nin mevcut kullanılan parçaları dışında bir şey yok).
- Wrapper yalnızca `meta` ve batch'lere dayanmalı; dataset adı, dosya yolu, subject id veya sabit
  kanal sayısı okumamalı.
- Checkpoint'i korumak için deneyi Python'dan çalıştır: Experiment.delete_checkpoints_on_exit = False.
  Ağırlıkları "state_dict" içinden "model." önekini kaldırarak çıkar.
- Codabench Submission Guide kesin otoritedir; docs/ altına kaydedilmiş halini oku.
  NeuralBench dokümanı onunla çelişirse Codabench geçerlidir.

## Takvim
- Track 03 sealed final AÇIK: 10 Kasım 2026'da açıldı (organizatör maili; yarışma deposu commit
  79b4453 "Track 03 is now open!"). Bitiş 21 Kasım 23:59 CET = 22 Kasım 01:59 Türkiye saati (GMT+3).
  Kesin başlangıç saati için Codabench Phases sekmesi geçerli. Kayıtlar kapandı. Diğer track'ler 5–21 Kasım.
- 16 KASIM MÜHÜRLÜ GÜNCELLEMESİ (FAQ "Sealed evaluation update · November 16"): mühürlü kayıtların açılış dakikaları (başlık
  otururken) artık modele VERİLMİYOR; geçen süre ilk verilen pencereden ölçülüyor. Bütün gönderimler yeniden skorlandı; mühürlü veride
  başka değişiklik planlanmıyor. Hak: toplam 12, günde en fazla 3 (takım ortak). Bitiş 21 Kasım 23:59 CET (yaml 22:59 UTC).
- Güncel mühürlü skorlar (20 Kasım, 16 Kasım yeniden skorlaması sonrası): #1 final540 87.99, #2 mix3_540 62.02, #3 gridpi 64.96,
  #4 aday1 47.42, #5 cand2 34.41, #6 cand3_gru 33.72, #7 ctx_c2 33.12, #8 ctx_c3 33.09, #9 ctxE_c3s3 33.29, #10 ctxE_c2hs3p 33.13,
  #11 mixI_540 **32.92** (EN İYİ, 20./198, süre 2178.67 s; ilk 3: 31.59 / 31.70 / 31.84; 10.: 32.16). 12 hakkın 11'i kullanıldı, KALAN 1.
  20 KASIM OKUMASI: iki mühürlü modelin yarı yarıya karışımı ikisinden de iyi → mühürlüde güvenilir tek şey farklı modellerin ortalaması.
  SÜRE KALİBRASYONU: mühürlü ≈ 1526.6 × yerel oran (aday 1'e göre) × 0.82; son paket ≤ 2900 s bu dönüşümle VE ≤ 3500 s faktör 1.0 ile.
  Kırpmaya duyarlı (artırmasız)
  modellerin çöküşü → mühürlü akışlar kaydın ortasından başlıyor; seçim KAYMALI değerlendirmeyle (İş D kalibrasyonu, İş E/F).
- 19 KASIM OKUMASI (kullanıcı): HGB E mühürlüde nötr (ctx_c2 33.12 ↔ c2hs3p 33.13), GRU E zararlı (c3s3 33.29); kaymalı CV'nin öngördüğü
  −0.22 taşınmadı. 0.2 s altındaki yerel farklara GÜVENİLMEZ; kararlar mühürlüde kanıtlanmış modellere dayanır. İş H iptal (GRU E içeriyor).
- 16 KASIM BULGUSU: aday 1 → aday 2 yerelde −0.37, mühürlüde −2.68 → mühürlü akışlar büyük olasılıkla kaydın başından
  başlamıyor. SEÇİM ARTIK k=0 ÇİFT CV'YE DEĞİL KAYMALI DEĞERLENDİRMEYE dayanır (İş A kayma kalibrasyonu; aday 3 kuralı:
  kaymalı test + kaymalı çift CV birincil, k=0 çift CV'de aday 2'ye göre en fazla +0.3 kötüleşme).
- 13 KASIM KURAL DEĞİŞİKLİĞİ (2026-competition: codabench/pages/scientific_faq_sleep.md, terms.md): mühürlü değerlendirme
  güncellendi, bütün gönderimler yeniden skorlandı (eski skorlar 39.56 / 38.66 / 37.49 GEÇERSİZ). Tahminler EEG'den gelmeli;
  kayıtların etiketlenme/paketlenme düzenliliklerini kullanmak YASAK ve denetimde kontrol ediliyor. Geçen süre (akışın
  ilk penceresinden beri) serbest. "Akış kaydın başından başlar" ifadesi FAQ'dan kaldırıldı → akış kaydın ortasından
  başlayabilir; akış başlangıcına dayanıklılık gerekir. SONUÇ: 30 sn onset ızgarası (π, π(D), c/epoch grubu, t mod 30)
  her yerden çıkarıldı; yeni adaylarda GRID KAPALI; referans ızgarasız mix3 (çift CV 39.40 ↔ mühürlü 37.79).
- Final sıralama bu aşamadan alınır: takımın en iyi geçerli (eligible) sealed skoru, otomatik;
  ilk 3 takım 22–30 Kasım'da tekrarlanabilirlik denetiminden geçer (eğitim yeniden çalıştırılır,
  skor ±2σ içinde kalmalı).
- Gönderim limiti: takım başına (Codabench organizasyonu, tüm üyeler ortak) toplam 12, günde en fazla 3 (16 Kasım).
  Failed gönderim hakkı yemez; kuyruğa giren/çalışan hak ayırır.
- Sealed'da çalışma log'ları, tahmin dosyaları ve skorlama çıktısı GİZLİ; yalnız durum ve skor görünür.
  Hata ayıklama ancak yerelde yapılabilir.
- KURAL: Günde 1 gönderim olduğu için her gönderimden önce ZIP yerelde Docker image
  (lionelyneuro/neural-compet:v4-sealed) ile test edilmeli.
- Değişiklikler Codabench Phases sekmesinden ve Discord'dan takip edilecek.

## Değerlendirme kuralı ve strateji
- KURAL: Model kararları ÇİFT CV ortalamasıyla verilir: (görülmemiş kişi CV + görülmüş kişi CV) / 2.
  Görülmemiş kişi CV = train+val üzerinde kişi bazlı 5 katlı GroupKFold. Görülmüş kişi CV = birden
  fazla gecesi olan kişilerin birer gecesi dışarıda, kalan geceleri eğitimde. Açık test setine
  (40 kayıt) sadece son kontrol için bakılır; test'e bakarak seçim/ayar yapılmaz.
- Gönderim planı (20 Kasım, GÜNCEL): #11 mixI_540 (32.92) en iyi; kalan 1 hak. İş K = mixI + ek tohumlar (yeni bileşen yok; öncelik GRU ctx
  tohum 10–19 → HGB ctx B/B_h2 tohum 1–2 → bütçe kalırsa HGB E B_h1/B2 tohum 1–2; kontroller: tohum 0 bit-aynı, ek tohum ağırlığı 0 → mixI ile
  bit-aynı akış, tam test zinciri, Docker ×2, karışık sıralı süre). İş J yedek = (ctx_c3 + ctx_c2 + ctxE_c2hs3p) / 3. ZIP'ler en geç 21 Kasım
  10:00; kullanıcı 21 Kasım 15:00'e kadar K'yı gönderir, Failed olursa J. 22 Kasım: docs/method.md son hali + tek son commit.
  DURUM (20 Kasım 17:50): mixK_540.zip (öncelik 1+2; SHA-256 81ed4466…16c9, tahmini mühürlü ≈ 2570–2610 s) ve yedek mixJ_540.zip (76a2f730…6d77,
  ≈ 2180 s) HAZIR, bütün testler + Docker ×2 PASS. Öncelik 3 (mixK3_540) süre kuralını geçmedi, kullanılmaz.
  KARAR (20 Kasım ~19:45): 0.72 arayışı (geri sayım süzgeci, uçtan uca CNN→GRU = İş L) kazanç vermedi → son hak mixK_540.zip; kullanıcı
  21 Kasım en geç ~18:00 TR gönderir; Failed olursa (hak yemez) mixJ_540.zip.
  GÖNDERİLDİ: #12 mixK_540, 20 Kasım ~19:50 (kullanıcı); 12/12 hak kullanıldı; sonuç bekleniyor.
- Eski gönderim planı (19 Kasım): #8 ctx_c3_540 (33.09) o zamanki en iyi; 12 hakkın 10'u kullanıldı, kalan 2. İş I: M = 0.5 · ctx_c3_540 +
  0.5 · ctxE_c2hs3p_540 (yeniden eğitim yok; karar öncesi metrik ağırlıklı dağılımlar ortalanır, sonra sürekli ağırlıklı medyan; ortak
  bileşenler bir kez; w = 0.5 SABİT, CV'den seçilmez). Akıl sağlığı: k=0 ve kaymalı çift CV (D, D', U[1,6]) hiçbir dağılımda M ikisinden
  de > 0.1 kötü olmasın. Süre: tahmini mühürlü ≤ 3000 s; aşarsa M' = 0.5 · ctx_c3 + 0.5 · ctxE_c2h (bildir). ZIP 19 akşamı / 20 sabahı.
  İş J (yedek, İş I'dan sonra, süre kalırsa): (ctx_c3 + ctx_c2 + ctxE_c2hs3p) / 3, aynı kural ve testler; 21 Kasım son hakkı için,
  gönderim kararı İş I'nın mühürlü sonucuna göre kullanıcıda. Son gönderim 21 Kasım akşamı (sınır 22 Kasım 01:59 TR). Kullanıcı gönderir.
  Ayrıntı: docs/status.md, docs/submissions_log.md.
- 18 KASIM KABUL KURALLARI (kullanıcı): BİRİNCİL ölçüt İş D'nin en iyi kayma dağılımı D ile kaymalı CV (dışarıdaki kayıtlar D'den
  örneklenen kaymayla, sabit tohum, 3 tekrar); k=0 yalnız bilgi. İş E (D ile yeniden eğitim): ctx_c3 tarifine göre Δ ≤ −0.15 ve ≥ 7/10 kat.
  İş F (ctx_c2 + ctx_c3 karar öncesi dağılım karışımı): ctx_c3'e göre Δ ≤ −0.10 ve ≥ 7/10. Süre: aday 1 × 1.5. Testler: akış ↔ toplu
  (CPU dedektör referansı), NaN / düz kanal, bağlam testleri, Docker ×2, süre.
- ADAY 2 KABULÜ (15 Kasım): referans aday 1 (çift CV 38.86, HGB(n2det) + yalnız GRU artırmalı, α 0.6; Δ'lar hem 39.40'a hem
  38.86'ya göre verilir); 38.86'ya göre Δ ≤ −0.15 ve ≥ 7/10 kat VE k=6 karışım zararı < +1.0. Tüm bileşenler --deterministic.
- KABUL KURALI (14 Kasım): karışım çift CV makro Δ ≤ −0.5 (ref IZGARASIZ mix3 tarifi 39.40 ↔ mühürlü 37.79) VE 10 katın
  ≥ 8'inde iyileşme, YA DA Δ ≤ −0.3 VE 10/10 kat; bütün karışım ölçümleri ızgarasız kararla. Süre kuralı (15 Kasım,
  kullanıcı): yerel pencere başına süre aday 1'e (mix3_n2det_cropg_540, mühürlü 1526.6 s, sınır 3600) göre en fazla +%50; 3 tohumlu
  dedektör korunur, tek tohuma inilmez. (Eski: mix3/gridpi'ye göre +%10.) Eski kural: mix_gru10_hgb'ye (40.28) göre ≥ 0.3 iyileşme VE 10 katın ≥ 8'inde iyileşme;
  540 kayıtla eğitilmiş; v4-sealed Docker testini geçmiş.
- Plan (sırayla, her adım çift CV ile raporlanır):
  A. Çift CV protokolü; v1 ve c5 yeniden ölçülür.
  B. Dağılım tahmini: kalan süre 10–14 aralığa bölünür, HistGradientBoostingClassifier; tahmin =
     aralık olasılıkları × metriğin etkin pencere ağırlıklarıyla ağırlıklı medyan (aralık temsilcisi:
     train'deki aralık içi medyan). c5 + kalibrasyon ile karşılaştırılır, bin bazında hatalar verilir.
  C. Yeni öznitelik grupları, her biri ayrı ablasyonla: göz hareketi (AF7−AF8 0.2–0.6 Hz, kırpma
     sayısı); bipolar derivasyonlar (AF8−TP10, AF7−TP9, TP9−TP10); uyku iğciği sayıları (TP9/TP10
     12–15 Hz RMS patlamaları, son 30/60 s); K-kompleks adayları (0.5–2 Hz büyük tepe-tepe);
     dinamikler (30 s/1/2/5 dk nedensel eğimler, 5 dk varyans ve lag-1 otokorelasyon, 10 dk EMA);
     spektral şekil (beta/alfa tepe frekansı, kenar frekansı, aperiodik eğim, entropi); göreli
     öznitelikler "şimdiye kadarki maksimum/minimuma göre".
  D. Küçük nedensel GRU (öznitelik dizisi → aralık olasılıkları), HGB ile ensemble.
  Final: seçilen model 540 kaydın tamamıyla (test dahil) yeniden eğitilir.

## Kaynaklar
- Track 3 rehberi: https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html
- Gönderim rehberi: https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_submission_guide.html
- Codabench: https://www.codabench.org/competitions/17983/
- Yarışma sitesi: https://neural-interfaces26.github.io/
