# CRS93 → Prokop Viser: živá synchronizace kloubů (read-only)

## Kopírování
1. V ukončeném Python hostu rozbal obsah archivu do `rob_seme/Vojtěch/rob_crs93_v4/`.
2. Potvrď přepsání `app/robot.py` a `extensions/prokop_visualizer.py`.
3. Nekopíruj nic do `Prokop/`; jeho `view_models.py` používáme beze změny.
4. Nemaž `data/calibration/` ani osobní nastavení.

## Spuštění (Arch Linux)
```bash
cd 'Vojtěch/rob_crs93_v4'
source .venv/bin/activate
python main.py
```
Při již platné referenci a zapnuté stanici můžeš podle bezpečnostních
postupů použít `mode real nohome`; pro offline test `mode virtual`.

V host konzoli:
```text
load prokop_visualizer
call prokop_visualizer start
call prokop_visualizer status
```
Pokud už je robot připojen, `start` spustí synchronizaci na 5 Hz. Jestli
jsi se připojil až potom:
```text
call prokop_visualizer sync on 5
```
Následně otevři `http://localhost:8080`.

Další příkazy:
```text
call prokop_visualizer sync once
call prokop_visualizer sync off
call prokop_visualizer sync on 10
call prokop_visualizer status
call prokop_visualizer stop
```

VIRTUAL: pro test změny kloubů:
```text
robot virtual-set deg 0 -20 10 0 15 0
```

## Protokol
Host posílá na `127.0.0.1:8081` UDP JSON `{"q": [q0, ..., q5]}`
(šest úhlů v radiánech). Viewer se pouští jako subprocess a
provede v paměti napojení na existující `set_q(q)` pomocí AST hooku.
Diskový soubor `Prokop/view_models.py` se nemění.

## Problémy / bezpečnost
- Pokud se 3D rameno nehýbe, přečti
  `data/logs/prokop_visualizer.log` a použij `call prokop_visualizer status`.
- Pokud Prokop změnil definici `set_q`, hook může selhat. V takovém případě
  pošli `view_models.py` k úpravě — zatím nebyl k dispozici pro cílený edit.
- Při porovnání s reálným robotem nejprve zkontroluj shodu pořadí a znamének
  kloubů, nul a URDF rámců. CRS93/CRS97 mohou sdílet mechaniku, ale
  vizualizační konvence musí být prověřené.
- Synchronizace robot **pouze čte** (`get_q()`), nehýbe s ním. Není určena
  k bezpečnostnímu řízení, detekci kolizí nebo k nouzovému zastavení.
- Při `robot release` je zpětná vazba odpojena a poslední 3D poloha může
  být zastaralá.
- Testováno 23× offline (včetně lokální simulované scény), ale skutečný
  `viser` a reálná stanice vyžadují ověření.

## Testy bez hardwaru
```bash
python -m unittest discover -s tests -v
```
