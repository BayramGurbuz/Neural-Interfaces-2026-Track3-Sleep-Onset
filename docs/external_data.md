# Dış veri beyanı (denetim için)

Bu projede kullanılan, yarışma verisi (Interaxon2026Muse) dışındaki her veri kümesi burada kayıtlıdır.

## HMC Sleep Staging (Haaglanden Medisch Centrum sleep staging database)

| Alan | Değer |
|---|---|
| Kaynak | PhysioNet, https://physionet.org/content/hmc-sleep-staging/1.1/ |
| Sürüm | 1.1 (18 Mart 2022) |
| Lisans | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| DOI | https://doi.org/10.13026/t79q-fr32 |
| Atıf | Alvarez-Estevez, D., & Rijsman, R. (2022). Haaglanden Medisch Centrum sleep staging database (version 1.1). PhysioNet. Ayrıca: Alvarez-Estevez D., Rijsman R. (2021) PLoS ONE 16(8): e0256111. PhysioNet platform atfı. |
| Erişim | Açık erişim (kimlik doğrulama yok); kapalı klinik veri değil |
| Bütünlük | Her indirilen dosya PhysioNet'in `SHA256SUMS.txt` listesindeki SHA-256 ile doğrulandı (`scripts/b3_hmc_fetch.py`) |
| Kullanım amacı | B3: Muse türevlerine eşlenen küçük nedensel uyku evreleyicisi (uyanık / N1 / N2+); çıktısı Muse modellerine öznitelik olarak girecek |

**Saklanan:**
- Yalnız ışık kapanışından ("Lights off" işareti) ilk N2 epoch'u + 600 sn'ye kadarki bölüm.
- Kanallar EEG F4-M1, C3-M2, C4-M1, EOG E1-M2, E2-M2.
- 128 Hz'e yeniden örneklenmiş (MNE `resample`), µV, float32; o bölümün 30 sn hipnogram etiketleri.
- Konum: `~/neuralbench/data/hmc/crops/SNxxx.npz`. Ham EDF'ler kırpıldıktan hemen sonra silindi; diskte aynı anda en fazla bir ham kayıt bulundu.

**Kullanılan kayıtlar:** `SHA256SUMS.txt`'te listelenen 151 kayıttan `numpy.random.default_rng(0)` ile seçilen 30 kayıt:

SN001, SN003, SN006, SN010, SN024, SN035, SN040, SN041, SN059, SN066, SN070, SN078, SN081, SN084, SN085,
SN088, SN090, SN099, SN102, SN106, SN109, SN113, SN117, SN119, SN124, SN129, SN134, SN136, SN143, SN152

Kayıt başına indirme süresi, saklanan boyut, ışık kapanma ve N2 zamanı: `~/neuralbench/data/hmc/fetch_log.json`.
151 kayda genişletilirse bu bölüm güncellenecek.

## Kullanılmayan (yalnız incelenen)

- Sleep-EDF Expanded (PhysioNet sleep-edfx 1.0.0, ODC-By 1.0): bilgi toplandı (`B_ASAMASI_RAPORU.md` B3), **indirilmedi**.
