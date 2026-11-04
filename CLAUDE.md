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
- Track 03 sealed final: 7 Kasım 2026 02:01 – 22 Kasım 2026 01:59 Türkiye saati (GMT+3)
  = 7 Kasım 00:01 – 21 Kasım 23:59 CET (Codabench Phases sekmesi). Diğer track'ler 5–21 Kasım.
- Final sıralama bu aşamadan alınır (en iyi sealed skor); ilk 3 takım tekrarlanabilirlik
  denetiminden geçer (eğitim yeniden çalıştırılır, skor ±2σ içinde kalmalı).
- Gönderim limiti: günde 1 (Terms'e göre toplam en fazla 5). Failed gönderim hakkı yemez.
- KURAL: Günde 1 gönderim olduğu için her gönderimden önce ZIP yerelde Docker image
  (lionelyneuro/neural-compet:v3) ile test edilmeli.
- Değişiklikler Codabench Phases sekmesinden ve Discord'dan takip edilecek.

## Kaynaklar
- Track 3 rehberi: https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html
- Gönderim rehberi: https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_submission_guide.html
- Codabench: https://www.codabench.org/competitions/17983/
- Yarışma sitesi: https://neural-interfaces26.github.io/
