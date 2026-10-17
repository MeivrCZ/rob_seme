# CRS93 v4 – `pview start`, `load all` a ChArUco

Patch zahrnuje pouze:

- `main.py` – zkratky `pview start [PORT]`, `pview stop`, `pview status`, příkazy `load all` a `available`;
- `app/extensions.py` – bezpečné vyhledání extensions (soubor musí obsahovat `setup(ctx)`) a hromadné načtení;
- `extensions/charuco_calibration.py` – ChArUco extension pro naši desku;
- `tests/test_command_shortcuts.py` – offline testy nových příkazů;
- tento návod.

Neobsahuje `config.json`, `app/robot.py`, `app/camera.py`, `Prokop/` ani žádné `data/`.
To znamená, že **nemění nastavení reálného pohybu, limity, homing ani uložené kalibrace**.

## Instalace

1. Ukonči host aplikaci `quit` poté, co robot stojí. Fyzický stav motorů řeš podle pravidel stanice: ukončení Pythonu není vypnutí pohonů.
2. Rozbal obsah ZIP **přímo do** `rob_seme/Vojtěch/rob_crs93_v4/` (nikoli do kořene `rob_seme/`).
3. Před přepsáním doporučujeme uložit/zkontrolovat vlastní změny v `main.py` a `app/extensions.py` (`git diff`). Pokud jsi main mezitím ručně upravoval, změny slouč místo slepého přepsání.
4. Spusť ve složce aplikace:

```bash
source .venv/bin/activate
python main.py
```

## Příkazy

```text
available                         # seznam extensions s funkci setup(ctx)
load all                          # vsechny dosud nenactene extensions; neprovadi jejich akce
loaded                            # jiz nactene
pview start                       # autoload prokop_visualizer a spusteni
pview start 8082                  # alternativni port
pview status
pview stop
```

`pview start` se dá zadat vícekrát; při již běžící vizualizaci místo chyby zobrazí stav. `load all` vynechá již načtené moduly, nezkouší importovat pomocný `prokop_bridge_runner.py` (ten nemá `setup(ctx)`), a případnou chybu načtení jednoho modulu vypíše a pokračuje dalšími.

`pview start` vyžaduje existující `Prokop/view_models.py`. Pokud používáš náš **Live Sync** patch, ve složce `extensions/` musí zůstat i `prokop_bridge_runner.py`. Při již připojeném reálném nebo virtuálním robotovi se živé čtení kloubů automaticky spustí; jinak samotná vizualizace poběží bez synchronizace.

## ChArUco

Deska ze snímků: 10×14 políček, `DICT_4X4_100`, `legacyPattern=True`. Naměřené délky hran: políčko **20 mm**, značka **15 mm**.

```text
load charuco_calibration
camera open                      # pouze pokud neni kamera otevrena
call charuco_calibration capture
call charuco_calibration inspect
call charuco_calibration calibrate 20 15
call charuco_calibration report
```

Nejdřív pořiď alespoň 12, nejlépe 20–30, ostrých snímků z různých náklonů a vzdáleností. Nezaměňuj kalibraci intrinsics (`charuco_calibration`) za vnější kalibraci kamera–robot (`eye_to_hand`). Příkaz `calibrate 20 15` před tímto počtem odlišných snímků skončí chybou. Výsledek je uložen do `data/calibration/` a přežije restart aplikace.

## Offline ověření

```bash
python -m unittest discover -s tests -v
```

Na vývojovém prostředí: 39 testů OK. Na skutečné stanici je potřeba otestovat načtení přes instalované knihovny a běh Prokopova Viseru.
