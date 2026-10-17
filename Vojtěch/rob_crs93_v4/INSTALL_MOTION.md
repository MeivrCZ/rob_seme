# CRS93 – ruční ovládání kloubů (patch k v4)

## Instalace

1. Ukonči naši aplikaci příkazem `quit` **až když robot stojí**. Vypnutí aplikace neznamená zastavení motorů.
2. Rozbal soubory ze ZIPu do `Vojtěch/rob_crs93_v4/`. Při přepsání se změní `main.py` a `app/robot.py` a přidá se test.
3. Současný `config.json` zůstane zachovaný. Není třeba měnit Prokopův Viser ani kalibrační soubory.
4. `python main.py` a v konzoli `mode virtual`.

## Virtuální řízení – funguje bez další konfigurace

```text
mode virtual
load prokop_visualizer
call prokop_visualizer start
robot status
robot jog q0 2
robot jog q2 -1
robot goto deg 0 -20 10 0 15 0
```

Změna se projeví i ve vizualizaci, jestli máš nainstalovaný předešlý Live Sync patch.

- `jog qN DELTA_DEG` – klouby jsou `q0` až `q5`, úhly ve stupních.
- `goto deg Q0 ... Q5` – zadání všech šesti **absolutních** kloubových úhlů ve stupních.
- `goto rad Q0 ... Q5` – totéž v radiánech.
- Příkaz `robot virtual-set` zůstává zachovaný a mění jen virtuální stav.

## REAL ovládání – v základním nastavení ZABLOKOVÁNO

Pro reálný pohyb je potřeba do stávajícího `config.json` doplnit sekci `motion`:

```json
{
  "robot": {"tty_dev": "/dev/mars"},
  "camera": {"name": "camera-crs93"},
  "motion": {
    "real_motion_enabled": false,
    "joint_limits_deg": null,
    "max_step_deg": 3.0
  }
}
```

**Nezapínej `real_motion_enabled: true` a nenahrazuj `joint_limits_deg` odhady.** Nejprve u odborné obsluhy ověřme všech šest skutečných kloubových limitů `[[q0_min,q0_max], ..., [q5_min,q5_max]]` podle stanice CRS93 a funkčnost bezpečnostních prvků. Po ověření lze do sekce `motion` dosadit správných šest intervalů a zapnout real_motion_enabled. Nastavený limit kroku smí být 0 až 10° (výchozí 3°).

Postup až po ověření a nastavení limitů:

```text
mode real nohome
robot status
robot jog q0 1
```

`nohome` je dovolen jen při platné referenci kloubů. Program zobrazí současné/cílové úhly, požádá o `MOVE` a znovu ověří aktuální polohu a pohybový stav těsně před příkazem. Potom použije `ctu_crs.move_to_q()` a čeká na `wait_for_motion_stop()`.

Pohybové příkazy se zablokují při `robot release`, při právě probíhajícím pohybu, při rozdílu od náhledu, při překročení softwarového intervalu či maxima jednoho kroku. Pokud příkaz selže během pohybu, stav se označí jako nejistý a další pohyb je blokován.

**Nejde o bezpečnostní systém.** Nemáme kontrolu kolizí, rychlostí, trajektorie ani nastavené bezpečné zóny. Před každým reálným pohybem se ujisti, že je prostor prázdný, klec zavřená, Arm Power správně aktivní a fyzický E-STOP po ruce. Při nečekaném chování použij fyzický Emergency Stop, nikoli příkaz v konzoli.

## Testy

```bash
python -m unittest discover -s tests -v
```

Testy používají náhradu `ctu_crs` a fyzickým robotem nikdy nehýbou. Software nebyl testován na skutečné stanici; první fyzické uvedení do provozu musí být kontrolované.
