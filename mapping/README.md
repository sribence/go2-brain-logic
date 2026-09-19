# 🚀 NERO_GO2 — 3D LiDAR SLAM & Térkép-Horgonyzó Modul

Ez a modul felelős a Unitree Go2 robot fedélzeti Hesai 3D LiDAR adatainak feldolgozásáért, az odometriai drift (kerekek/lábak csúszása) kompenzálásáért és a sűrű 3D térképek előállításáért.

---

## 🎯 Fejlesztési Alapelvek & Tapasztalatok

1. **Szigorúan 0% Pont-Törlés (No Point Deletion):**
   * A LiDAR mérései ultra-pontos geometriai adatok. A pontok szűrése/elrejtése nem megoldás, mert elrejti a driftet és elveszíti a távoli falak információit.
   * A driftet és a dupla falakat **szigorúan geometriai alapon**, térbeli SLAM horgonyzat segítségével húzzuk a helyére.

2. **Divergens Odometria Hiba (Drift):**
   * 1-2 másodperces ablakban a nyers odometria pontos.
   * 5-10 másodperc folyamatos séta/forgás után az odometria 1.5–2.0m-t és 15–30°-ot elcsúszik.
   * Szűrés nélkül a kanyarok után dupla falak keletkeznének — ezt a problémát a Kulcsképkockás és Sík-horgonyzó SLAM oldja meg.

---

## 🌐 A 3 Megvalósított SLAM Opció & Webes Vizsgálók

### Opció 1: Kulcsképkockás Horgonytérkép SLAM (`keyframe_slam.py` / `keyframe_slam.html`)
* **Működés:** $0.4\text{m}$ haladásonként vagy $12^\circ$ forgásonként **Fix Kulcsképkockát (Keyframe)** rögzít a térbeli vázként (narancssárga gömbökként megjelenítve). Az új beérkező pontok valós időben ezekre a rögzített kulcsképkockákra lokkolnak rá.
* **Elérés:** `http://localhost:8088/keyframe_slam.html`
* **Használat:** Valós idejű élő SLAM integrációhoz a roboton.

### Opció 2: Sík-Horgonyzat SLAM (`planar_slam.py` / `planar_slam.html`)
* **Működés:** Az első mérésekből kinyeri a fő fali síkokat ($Ax+By+Cz+D=0$), amelyek láthatatlan horgonyként tartják a sűrű pontfelhőt, megszüntetve a falak duplázódását.
* **Elérés:** `http://localhost:8088/planar_slam.html`
* **Használat:** Sík dominanciájú beltéri szobákban és folyosókon.

### Opció 3: 3-Állású Oktatási & Demonstrációs Felület (`benchmark_3d.py` / `compare_3d.html`)
* **Működés:** Háromállású interaktív vizsgáló felület a diákoknak és bemutatókhoz:
  - 🔴 **Nyers Odometria:** Bemutatja az elcsúszást (drift) 10-90 mp alatt.
  - 🟡 **Kanyarodási Zaj Szűrés:** Bemutatja a forgások hatását a pontfelhőre.
  - 🟢 **Globális Fal-Horgonyzás:** Bemutatja az eredeti fali horgonyzat ráragasztását pontvesztés nélkül.
* **Elérés:** `http://localhost:8088/compare_3d.html`

### Opció 4: Inkremens 3D Konfidencia-Box SLAM (`incremental_confidence_slam.py` / `confidence_slam.html`)
* **Működés:** Ahogy telik az idő, a robot fokozatosan észleli a falakat, asztalokat és tereptárgyakat. Minden tárgy egy **Konfidencia pontszámot ($C \in [0.0, 1.0]$)** kap. Amikor $C \ge 0.50$, a tárgy megszilárdult zöld 3D Bounding Box-szá válik a térben.
* **Elérés:** `http://localhost:8088/confidence_slam.html`
* **Használat:** Folyamatos 3D objektum-szilárdítás és 2D alapsík-térképezés szemléltetésére.

---

## 💻 Parancsok & Futtatás

```bash
# 1. Helyi HTTP Webszerver elindítása (Port 8088)
python -m http.server 8088 --directory docker/mapping

# 2. SLAM opciók újra-feldolgozása
python incremental_confidence_slam.py  # Generálja a confidence_slam.html-t
python keyframe_slam.py              # Generálja a keyframe_slam.html-t
python planar_slam.py                # Generálja a planar_slam.html-t
python benchmark_3d.py                # Generálja a compare_3d.html-t
```
