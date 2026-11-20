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
- **Eğitim pencereleri, açıkça:** her açık eğitim kaydı için (a) POZİTİF = ilk N2 epoch'unun başlangıcından sonraki
  60 pencere (onset+0 … onset+300 sn; ham dosyalar bu bölümü içeriyor, akış boru hattı kullanmıyor), (b) NEGATİF = akıştaki,
  onset'e ≥ 300 sn kalan pencereler, (c) onset'e 0–300 sn kalan pencereler eğitimde KULLANILMAZ (etiket belirsizliği).
  Dedektör ikili sınıflandırıcıdır; onset zamanı, kayıt uzunluğu veya pencere indeksi girdi değildir.
- **Çıkarım:** değerlendirici her pencereyi sırayla verir; P(N2) yalnız o pencerenin 640 örneğinden hesaplanır, özetler yalnız
  önceki pencerelerin P değerlerinden; kaydın sonu/uzunluğu/gelecek pencereler hiçbir hesaba girmez.
- Sızıntı kuralı: HGB üyeleri eğitilirken eğitim satırlarının P'si fold-dışıdır (kişi bazlı 5 iç kat); CV'de dedektör her katta
  yalnız o katın eğitim kayıtlarıyla eğitildi.
- Deterministik eğitim: `scripts/repro.set_deterministic` (cuDNN deterministic, `use_deterministic_algorithms`), tohum 0.

## Karar katmanı
- 13 Kasım FAQ güncellemesine göre 30 sn onset ızgarası (π / π(D)) ve epoch-hizalı öznitelikler yeni adaylardan çıkarıldı.
  Karar = sınıf dağılımı × metrik ağırlıkları ile sürekli ağırlıklı medyan (ızgarasız).

## Aday 1 (15 Kasım): mix3_n2det_cropg_540 — HGB(n2det) + GRU (kırpma artırmalı) + TCN + GRU-emb, α 0.6, ızgarasız
- **Bileşenler:** (1) HGB(c): 4 dağılım sınıflandırıcısı (B, B_h1, B_h2, B2), 817 nedensel öznitelik + 5 dedektör özeti = 822;
  (2) GRU-uw 10 tohum; (3) TCN 10 tohum; (4) GRU-emb 3 tohum (3 CNN kodlayıcı). Dizi bileşenlerinin metrik-ağırlıklı ızgara
  dağılımları eşit ağırlıkla ortalanır, HGB dağılımıyla α = 0.6 (dizi) / 0.4 (HGB) karıştırılır; karar = sürekli ağırlıklı
  medyan. Izgara (30 sn), kalibrasyon ve yumuşatma kapalı. Dedektör özetleri yalnız HGB girdisinde (dizi girdilerinde yok).
- **GRU başlangıç kırpma artırmasıyla eğitildi:** her epoch ve her eğitim kaydı için %50 kırpma yok, %35 ilk 1–12 pencere
  (5–60 sn) atılır, %15 ilk 13 … (onset − 60 sn) pencere atılır; girdiler (geçen süre, ilk-2-dk tabanı) kırpılan
  başlangıçtan yeniden kurulur, hedefler değişmez (`scripts/gru_cv.py run_fold cfg["crop"]`). Amaç: akışın kaydın başından
  başlamadığı duruma dayanıklılık (13 Kasım FAQ). TCN ve GRU-emb artırmasız (çift CV'de artırma bu ikisinde zararlıydı).
  Artırma yalnız eğitim kayıtlarının kendi pencerelerini kullanır; etiket/kayıt düzenliliği kullanmaz.
- **Seçim (çift CV, 500 kayıt, ızgarasız karar):** 38.86 (unseen 39.93, seen 37.79); referans ızgarasız mix3 39.40'a göre −0.54,
  10 katın 8'inde iyi; α iç içe LOFO ile 10 katta da 0.6 (`scripts/b2_mix.py`, `OUT/b2_mix_cropmix.json`).
- **Eğitim (540 kayıt), sırayla** (hepsi `scripts/` içinden, `.venv/bin/python`):
  1. `train_n2det_final.py det` — dedektör, tohum 0, 13 epoch (çift CV LOFO medyanı), deterministik; eğitim satırları için
     kişi bazlı 5 iç katlı fold-dışı P (`n2det_final/p_oof.npy`).
  2. `train_n2det_final.py hgb` — 4 HGB üyesi, random_state 0, iterasyon 150 / 150 / 250 / 150 (çift CV LOFO medyanı, 25'e
     yuvarlı), eğitim satırlarında fold-dışı P'nin özetleri → `ens_c_all_n2det/`.
  3. `train_mix3_final.py --all --deterministic --crop --epochs=gru:20,tcn:15,gemb:10 --tag=crop` → `mix3crop_all/`;
     buradan YALNIZ GRU-uw alınır (10 tohum 0–9, 20 epoch = artırmalı çift CV LOFO medyanı).
  4. TCN (10 tohum × 10 epoch), CNN kodlayıcı (3 × 12 epoch), GRU-emb (3 × 10 epoch): `mix3_all/` (11 Kasım,
     `train_mix3_final.py --all`; gönderim #2 mix3_540 ile aynı ağırlıklar).
  5. `train_n2det_final.py mix --alpha=0.6 --seq=mix3_all --gru-from=mix3crop_all --out=mix3_n2det_cropg_all`
  6. `build_hgb_submission.py --model ~/neuralbench/save/mix3_n2det_cropg_all/model.joblib --name mix3_n2det_cropg_540 --smooth-a 0`
- **Tekrarlanabilirlik:** 1–3 deterministik (cuDNN deterministic, sabit tohumlar). 4. adımdaki TCN / kodlayıcı / GRU-emb
  deterministik ayar olmadan eğitildi; yeniden eğitimde bit düzeyinde aynı olmaz (ölçülen etki ≈ 0.2 s, denetim toleransı
  ±2σ içinde).
- **Ortam:** Python 3.12.3, torch 2.11.0+cu128, scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1, joblib 1.6.0; RTX 3050 Laptop
  (4 GB). Eğitim süresi: dedektör ≈ 6 dk, HGB 4 üye ≈ 16 dk (CPU), dizi modelleri ≈ 6 dk (GPU).
- **Çıkarım:** yalnız numpy / torch / joblib / scipy (find_peaks, welch); import satırları final540 ve mix3_540 ile aynı.
  ZIP SHA-256, Docker koşuları ve süreler: `docs/submissions_log.md`.
- **Beklenen mühürlü skor ≈ 37.25** (yerel 38.86; referans 39.40 ↔ mühürlü 37.79, fark −1.61).

## Aday 2 (16 Kasım): cand2_cropGc_540 — HGB(c, yakınlık + kırpma) + GRU (kırpma + yakınlık) + TCN + GRU-emb, α 0.6, ızgarasız
- **Dedektör (çok sınıflı):** aynı CNN mimarisi, 5 sınıf: onset'e ≥ 300 sn / 150–300 / 60–150 / 0–60 sn kalan akış pencereleri ve onset
  sonrası pencereler (ham dosyaların akışta kullanılmayan onset-sonrası 300 sn'si). Önceki ikili dedektörden farkı: onset'e 0–300 sn kalan
  pencereler de kalan-süre sınıflarıyla eğitimde. Çıktı = "yakınlık" = beklenen sınıf indeksi / 4; 3 tohum (0, 1, 2), çıkarımda ortalama.
  Girdi yalnız o pencerenin 640 ham örneği; özetler (EMA 30 sn / 2 dk, koşan maksimum, 30 sn artış) yalnız geçmiş pencerelerden.
  Kayıt uzunluğu, kayıt sonu, gelecek pencere, dosya adı veya denek kimliği kullanılmaz. Sınıf etiketleri yalnız eğitimde kullanılır.
- **Başlangıç kırpma artırması:** HGB eğitim tablosuna her eğitim kaydının bir kırpılmış kopyası eklenir (%70 ilk 1–12 pencere, %30 ilk
  13 … (uzunluk − 12) pencere atılır; tohum 0); GRU her epoch ve kayıtta yeniden kırpılır (%50 yok, %35 1–12, %15 13 … onset − 60 sn).
  Kırpılan kopyada bütün nedensel öznitelikler ve dedektör özetleri kırpılan başlangıçtan yeniden hesaplanır; hedefler değişmez. Amaç:
  akışın kaydın başından başlamadığı duruma dayanıklılık (13 Kasım FAQ).
- **Sızıntı kuralı:** CV'de dedektör her katta yalnız o katın eğitim kayıtlarıyla eğitildi; HGB / GRU eğitim satırlarının dedektör çıktısı
  iç katlardan fold-dışı. Finalde eğitim satırlarının yakınlığı kişi bazlı 5 iç katlı fold-dışı.
- **Deterministik:** bütün sinir ağı bileşenleri `repro.set_deterministic` ile (cuDNN deterministic, sabit tohumlar); HGB random_state 0.
- Seçim, komutlar, kontroller, SHA-256: `docs/submissions_log.md` "cand2_cropGc_540"; seçim tabloları `B_ASAMASI_RAPORU.md` 16 Kasım.
- **Beklenen mühürlü skor ≈ 36.7–36.9** (yerel 38.49).

## İş E adayları (19 Kasım): ctxE_c3_540 ve ctxE_c2h_540 — 16 Kasım mühürlü kurgusuna göre kırpma (ızgarasız)
- **Mühürlü kurgu (16 Kasım FAQ):** kayıtların açılış dakikaları modele verilmiyor; geçen süre ilk verilen pencereden. Kayma dağılımı,
  gönderilmiş 8 modelin mühürlü skor FARKLARININ 40 açık test kaydında yeniden üretilmesiyle seçildi (İş D, `scripts/shift_fit_d.py`):
  D = tekdüze U[2,5] dk. Bu yalnız eğitim artırmasının dağılımını belirler; çıkarımda kayma bilgisi, kayıt uzunluğu veya kayıt sonu kullanılmaz.
- **Bağlamlı N2 dedektörü (İş C):** pencere başına CNN gömmesi (Aday 1–2'deki kodlayıcı), son 6 pencerenin gömmesi üzerinde GRU(32), 5 sınıflı
  "yakınlık" çıkışı, 3 tohum ortalaması; akış başlangıcından önceki konumlar sıfır dolgu (yalnız akışın gördüğü pencereler). Eğitim
  pencereleri ve sızıntı kuralı yukarıdaki dedektörle aynı (fold-dışı çıktılar kişi bazlı iç katlardan).
- **Kırpma E (eğitim artırması):** dizi modellerinde her epoch ve kayıtta başlangıç k: %10 kırpmasız, %50 U[24,60] pencere (2–5 dk), %40
  U[0,120] pencere (0–10 dk); HGB tablosunda kayıt başına 2 kopya (U[2,5] dk ve U[0,10] dk) + özgün satırların rastgele ~%10'u (seed 0).
  Pencere kalmayan kopya atılır. Her kopyada bütün nedensel öznitelikler kırpılan başlangıçtan yeniden akıtılır (ilk-2-dk tabanı, EMA'lar,
  koşan ekstremumlar, geçen süre, dedektör özetleri); finallerde dedektör bağlamı da kırpmadan başlar (kırpma-kesin `close_oof_L`).
  CV'de dış-kat bağlamlı dedektörler saklanmadığı için kırpılmış kopyanın ilk 5 penceresinde bağlam kırpma öncesini görür (bilinen yaklaşım).
- **ctxE_c3_540** = HGB(c) 4 üye kırpma E + bağlamlı yakınlık (`ens_c_all_ctxE_crop`, n_iter 200 / 200 / 300 / 200) + GRU-uw 10 tohum kırpma E +
  bağlamlı yakınlık girdisi (`gru_ctxE_all`, 20 epoch), α 0.4. Önceden tanımlanmış iki bileşen setinden biri (ctx_c3 seti); kabul kuralını geçti
  (kaymalı çift CV, ctx_c3 tarifine göre D −0.22, 8/10; D' = U[0,8] dk −0.21).
- **ctxE_c2h_540 (HİBRİT, SONRADAN SEÇİLDİ)** = aynı HGB E + ctx_c2'nin eski dizi seti: GRU-uw kırpma (eski 50/35/15 karışımı) + bağlamlı
  yakınlık (`gru_ctx_all`), TCN 10 tohum ve GRU-emb 3 tohum (`mix3det_all`, artırmasız); α 0.4 (iç içe LOFO). **Bu satır önceden tanımlanmış
  aday değildi**: İş E karışım tablosundaki 5 bilgi satırından biriydi ve sonuçlar görüldükten sonra seçildi (D −0.22, 8/10; D' −0.31, 8/10).
  Önceden tanımlanmış ctx_c2 seti (TCN / GRU-emb da kırpma E ile) kat sayısında kaldı (D −0.19, 6/10). Bu seçim yanlılık taşır; kullanıcı
  iki dizi setinin mühürlüde taşınmasını ölçmek için iki paketi birlikte gönderdi.
- Ağaç sayısı / epoch: İş E tarifleri için ayrı k=0 seçimi yapılmadı; eşdeğer kırpmalı tariflerin kat seçimleri (CV) ve medyanları (final):
  HGB ctx6 kırpma, GRU 20, TCN 15, GRU-emb 7 (kırpma E finallerinde; c2h'de TCN / GRU-emb eski 10 / 10 epoch'lu dosyalar).
- Komutlar (`scripts/`): `shift_fit_d.py`; CV `shift_cv.py hgb --det=ctx [--crop=E]`, `shift_cv.py seq uw_n10_ctx6_cropE|tcn_n10_cropE|
  cnn3_emb_cropE`, `shift_cv.py mix ...` (`worke_*_chain.sh`); finaller `train_n2det_final.py hgbmc --det=ctx --cropE`, `train_gru_n2c.py
  --det=ctx --cropE`, `train_n2det_final.py mix --alpha=0.4 --ens=ens_c_all_ctxE_crop --seq=mix3det_all --gru-from=gru_ctxE_all|gru_ctx_all`
  (c3: seq_comps ("gru",), TCN / GRU-emb anahtarları çıkarılır), `build_hgb_submission.py --name ctxE_c3_540|ctxE_c2h_540 --smooth-a 0`.
  Bütün bileşenler deterministik (`--deterministic` / repro; HGB random_state 0).

## İş G (19 Kasım): HGB tohum ortalaması
- HGB üyeleri (random_state 0 / 1 / 2, `max_features` 0.5 nedeniyle farklı) aynı kırpma E tablosuyla eğitilir, olasılıklar ortalanır. Ön kontrol yalnız
  B üyesinde (kaymalı çift CV, D −0.18, 9/10). ctxE_c3s3_540: 4 üye × 3 tohum. ctxE_c2hs3p_540: süre sınırı nedeniyle yalnız B ve B_h2 × 3 tohum,
  üye ağırlıkları eşit (1/4) tutulur. Diğer bileşenler İş E paketleriyle aynı dosyalar.

## İş I (20 Kasım): mixI_540 — iki mühürlü paketin sabit ağırlıklı karışımı
- M = 0.5 · ctx_c3_540 + 0.5 · ctxE_c2hs3p_540, yeniden eğitim yok. İki paketin karar öncesi metrik ağırlıklı dağılımları ortalanır, sonra mevcut karar
  kuralı (sürekli ağırlıklı medyan). İki paket aynı α (0.4) ve aynı GRU ctx'i paylaştığı için eşdeğer biçim:
  0.4 · (⅔ GRU ctx + ⅙ TCN + ⅙ GRU-emb) + 0.6 · (½ norm(HGB ctx · ω) + ½ norm(HGB E-s3p · ω)); her HGB seti kendi bin sınırlarıyla.
- w = 0.5 önceden sabitlendi (kullanıcı, 19 Kasım); CV yalnız akıl sağlığı için kullanıldı (M hiçbir dağılımda iki bileşeninden de > 0.10 kötü
  değil). Gerekçe: 0.2 s altındaki yerel farklar mühürlüye taşınmadı; karışım yalnız mühürlüde kanıtlanmış iki modelden oluşur.
- Komutlar: `shift_cv.py imix` (`worki_chain.sh`), sarmalayıcıda isteğe bağlı ikinci HGB seti (`hgb2_*` anahtarları) ve `seq_weights`;
  testler `mixI_build_tests.sh`, `mixI_equiv_test.py`.

