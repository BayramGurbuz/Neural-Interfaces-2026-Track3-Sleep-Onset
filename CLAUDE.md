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
- Docker image: lionelyneuro/neural-compet:v3 (yerel test için kullanılabilir).
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
- 5 gönderim hakkı (10 Kasım itibarıyla hiçbiri kullanılmadı); 10–21 Kasım = 12 gün, günde en fazla 1.
- Final sıralama bu aşamadan alınır: takımın en iyi geçerli (eligible) sealed skoru, otomatik;
  ilk 3 takım 22–30 Kasım'da tekrarlanabilirlik denetiminden geçer (eğitim yeniden çalıştırılır,
  skor ±2σ içinde kalmalı).
- Gönderim limiti: takım başına (Codabench organizasyonu, tüm üyeler ortak) toplam 5, günde en fazla 1.
  Failed gönderim hakkı yemez; kuyruğa giren/çalışan hak ayırır.
- Sealed'da çalışma log'ları, tahmin dosyaları ve skorlama çıktısı GİZLİ; yalnız durum ve skor görünür.
  Hata ayıklama ancak yerelde yapılabilir.
- KURAL: Günde 1 gönderim olduğu için her gönderimden önce ZIP yerelde Docker image
  (lionelyneuro/neural-compet:v3) ile test edilmeli.
- Değişiklikler Codabench Phases sekmesinden ve Discord'dan takip edilecek.

## Değerlendirme kuralı ve strateji
- KURAL: Model kararları ÇİFT CV ortalamasıyla verilir: (görülmemiş kişi CV + görülmüş kişi CV) / 2.
  Görülmemiş kişi CV = train+val üzerinde kişi bazlı 5 katlı GroupKFold. Görülmüş kişi CV = birden
  fazla gecesi olan kişilerin birer gecesi dışarıda, kalan geceleri eğitimde. Açık test setine
  (40 kayıt) sadece son kontrol için bakılır; test'e bakarak seçim/ayar yapılmaz.
- Gönderim planı: 10 Kasım #1 submissions/final540.zip → 11 Kasım #2 submissions/mix_gru10_hgb.zip
  → kalan 3 hak YALNIZCA kabul kuralını geçen yeni modeller için. Ayrıntı: docs/status.md,
  docs/submissions_log.md.
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
