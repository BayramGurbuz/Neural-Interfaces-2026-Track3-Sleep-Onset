# Yöntem notu (denetim paketi; 14 Kasım 2026 taslağı — final adayla tamamlanacak)

## Veri
- Yalnız yarışma verisi Interaxon2026Muse (540 kayıt / 203 kişi). Dış veri gönderilen hiçbir modelde kullanılmadı
  (`docs/external_data.md`: HMC yalnız denendi ve bırakıldı).

## N2 dedektörü (Work 2; 14 Kasım'dan itibaren adaylarda)
- Tek pencerelik CNN (`scripts/cnn_encoder.Encoder`, 2 çıkış, ~35k parametre), girdi 4 × 640 ham µV örnek (×1/20, ±15 kırpma).
- **Eğitim verisi yalnız açık eğitim kayıtlarının EEG'si:** pozitif = ilk N2 epoch'undan sonraki 5 sn'lik pencereler
  (ham BrainVision dosyalarının akış boru hattında kullanılmayan onset-sonrası bölümü; `scripts/n2det_extract.py`),
  negatif = onset'e ≥ 300 sn kalan pencereler. Aradaki pencereler eğitimde kullanılmaz.
- **Çıkarımda yalnız o ana kadarki EEG kullanılır:** her pencere için P(N2) o pencerenin ham örneklerinden hesaplanır;
  nedensel özetler (EMA 30 sn / 2 dk, koşan maksimum, 30 sn artış; `scripts/n2det_stream.py`) yalnız geçmiş pencerelerin
  P'lerinden. Gelecek pencere, kayıt uzunluğu, kayıt sonu, dosya adı veya denek kimliği kullanılmaz.
- Sızıntı kuralı: HGB üyeleri eğitilirken eğitim satırlarının P'si fold-dışıdır (kişi bazlı 5 iç kat); CV'de dedektör her katta
  yalnız o katın eğitim kayıtlarıyla eğitildi.
- Deterministik eğitim: `scripts/repro.set_deterministic` (cuDNN deterministic, `use_deterministic_algorithms`), tohum 0.

## Karar katmanı
- 13 Kasım FAQ güncellemesine göre 30 sn onset ızgarası (π / π(D)) ve epoch-hizalı öznitelikler yeni adaylardan çıkarıldı.
  Karar = sınıf dağılımı × metrik ağırlıkları ile sürekli ağırlıklı medyan (ızgarasız).

## Tamamlanacak (final aday ile)
- Eğitim komutları, tohumlar, epoch sayıları, ortam (paket sürümleri), ZIP SHA-256, Docker koşuları, pencere başına süre.
- Not: mix3_540 (#2) CNN kodlayıcısı deterministik ayar olmadan eğitildi; bit düzeyinde tekrar üretilemez (ölçülen etki
  ≈ 0.2 s). Yeni adaylar `--deterministic` ile.
