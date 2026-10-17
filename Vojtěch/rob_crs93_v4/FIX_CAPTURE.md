# CRS93 – oprava pádu při ChArUco capture (OpenCV 5.x)

## Příčina

Na stanici byl vidět `cv2.error` v `cv2.aruco.drawDetectedCornersCharuco`:

```
(-215:Assertion failed) _charucoCorners.total() == _charucoIds.total()
```

Původní `capture` **uložil PNG**, následně volal `inspect`, které vykreslovalo
body funkcí citlivou na chybné či nestandardní tvary ChArUco polí.
Protože `cv2.error` nebyl ošetřen v host konzoli, ukončila se aplikace.

## Co mění patch

- `inspect` validuje počet souřadnic rohů vs. počet jejich identifikátorů;
- pro vizualizaci používá `cv2.circle`, nikoliv `drawDetectedCornersCharuco`;
- neplatné páry přeskočí a vypíše upozornění, původní snímek nezahodí;
- `calibrate` přeskočí záznamy s neplatnými páry;
- `capture` neukončí hlavní aplikaci, pokud vznikne chyba při náhledu.

## Instalace

Zálohujte případné vlastní úpravy v `extensions/charuco_calibration.py`.
Rozbalte ZIP **do** složky `rob_seme/Vojtěch/rob_crs93_v4/`.
Přepište jen `extensions/charuco_calibration.py` a případně testovací soubor.
Žádné soubory v `data/`, `app/`, `main.py`, `Prokop/` ani nastavení pohybu se nemění.

Pokud host běží:

```
reload charuco_calibration
```

Jinak:

```
python main.py
camera open
load charuco_calibration
call charuco_calibration capture
call charuco_calibration inspect
```

Je možné provádět kalibraci kamery **bez připojování či pohybu robota**.
Nesmí se předpokládat, že `robot.release()` robot opětovně povolil – to je
samostatný proces obnovení jeho řízení.

Pro výpočet intrinsics nasbírejte alespoň 12, lépe 20–30, odlišných snímků,
na nichž je deska různě posunutá a nakloněná:

```
call charuco_calibration calibrate 20 15
call charuco_calibration report
```

V `data/captures/charuco/` zůstávají originální PNG, v `overlays/` vizuální kontrola.

## Offline testy

S testovacím OpenCV 4.13.0 prošlo 42 testů včetně záměrně nesouhlasících
počtů ChArUco rohů a ID i simulované chyby náhledu. Kód dosud nebyl ověřen
na konkrétní instalaci OpenCV 5.0.0 na školním PC.

```
python -m unittest discover -s tests -v
```
