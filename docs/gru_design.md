# Adım D — küçük nedensel GRU: tasarım taslağı (henüz eğitilmedi)

## Girdi
- Pencere başına öznitelikler (durumsuz): taban 40 (4 kanal × 10) + spec 20 + eye 2 = 62.
  EMA / dyn / maxmin gibi geçmiş özetleri VERİLMEZ; dinamikleri GRU kendisi öğrenir.
- Kayda göre göreli girdi (nedensel): aynı 62 özniteliğin "ilk 2 dk ortalamasından farkı"
  (CausalFeatureStream'deki *_rel ile aynı tanım; 2 dk dolana kadar kümülatif ortalama).
- Geçen süre: log1p(elapsed_s), train'deki en uzun kayıt (300 pencere = 1500 s) ile kırpılmış.
- Toplam ~125 girdi. Standardizasyon: ortalama/std yalnızca eğitim fold'unun pencerelerinden
  (dataset düzeyi sabit; kayıt geneli istatistik değil), submission'a sabit olarak gömülür.

## Mimari (~60k parametre)
Linear(125→64) + GELU + LayerNorm → GRU(64, 2 katman, tek yönlü, dropout 0.2) → Linear(64→11)
Çıktı: B'nin 11 sınıfı [0,10,20,30,40,60,90,150,225,300,450,600] üzerinde softmax.

## Kayıp
- Ağırlıklı çapraz entropi, ağırlık = window_weights (metriğin pencere ağırlıkları; kayıt başına
  toplam 1/N_rec, yani her kayıt eşit), padding maskeli.
- ÖNEMLİ: ağırlıklı CE ile model P(c|x) değil, P(c|x)·E[w | c, x] ile orantılı "metrik-ağırlıklı"
  bir dağılım öğrenir. Bu yüzden kararda ω İKİNCİ KEZ uygulanmaz: tahmin = q'nun sürekli medyanı
  (ω = 1). Kontrol ablasyonu: ağırlıksız CE + B'deki gibi ω ile karar.
- Olası ek: sınıflar sıralı olduğu için kümülatif (ordinal) kayıp — ilk sürümde yok.

## Eğitim
- Tam dizi BPTT (kayıtlar ≤ 300 adım), sağdan padding + maske. Tek yönlü GRU'da sağ padding
  önceki çıktıları etkilemez → kayıt uzunluğu/bitişi modele sızmaz.
- Batch 32 kayıt, AdamW lr 2e-3, weight decay 1e-2, ≤ 150 epoch, gradient clip 1.0.
- Augmentasyon: kayıt başına log-güç özniteliklerine rastgele sabit kayma (kişiler arası genlik
  farkı), küçük Gauss gürültüsü, öznitelik dropout. Başlangıç kırpma YOK (sealed kaydın başından
  başlar, elapsed anlamlı).
- Epoch sayısı: her fold'un eğri(epoch)'u kaydedilir, fold k için diğer fold'ların ortalama
  eğrisinin argmin'i (HGB'deki LOFO ile aynı mantık).
- 3 seed ortalaması (varyans ↓; tekrarlanabilirlik denetiminde ±2σ şartı için de önemli).
- Değerlendirme: aynı çift CV (5 unseen + 5 seen fold), düz + sızıntısız kalibrasyon, bin katkıları.

## HGB ile ensemble
- HGB → metrik-ağırlıklı dağılım: W = P·ω / Σ(P·ω). GRU → q. Karışım: α·q + (1−α)·W,
  karar ω = 1 ile sürekli medyan. α ∈ {0, 0.25, 0.5, 0.75, 1}, LOFO ile seçilir.
- Sonra mevcut kalibrasyon (izotonik harita) aynen.

## Inference (submission.py)
- Model durumu: gizli durum h; reset_state() → h = None ve baseline sıfırlanır.
- Her pencere: öznitelikler (numpy) → standardize → GRU tek adım (torch, CPU yeterli) → softmax.
- Ağırlıklar state_dict olarak ZIP'te; mimari submission.py içinde.

## Süre tahminleri
- Eğitim (RTX 3050): tüm veri GPU'da (~75k × 125 float ≈ 40 MB). Epoch ≈ 13 batch × 300 adım,
  cuDNN GRU-64 → < 1 s/epoch. Fit başına ~150 epoch ≈ 1–2 dk. 10 fold × 3 seed = 30 fit
  ≈ 45–60 dk; ablasyon (ağırlıksız CE) ile ~1.5–2 saat. Final (540 kayıt, 3 seed) ~5 dk.
- Docker inference: GRU adımı ~0.2–0.5 ms + öznitelikler ~1–2 ms → yalnız GRU açık testte
  ~15–30 s; HGB (numpy, 817 öznitelik) ile ensemble ~100–120 s (60 dk sınırının çok altında).

## Riskler
- Train'deki en uzun kayıt 300 pencere; sealed kayıtları daha uzunsa GRU'nun 300 adım sonrası
  davranışı test edilmemiş olur (elapsed kırpması kısmen korur). Test: sentetik olarak uzatılmış
  diziler üzerinde çıktının patlamadığı kontrol edilir.
- Az veri (500 kayıt) → aşırı öğrenme; küçük model + güçlü regularizasyon + seed ortalaması.
