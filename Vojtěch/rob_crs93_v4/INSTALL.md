# CRS93 – ChArUco: diagnostika kalibrace a loopcapture

Tento ZIP je **patch**, ne kompletní aplikace. Vychází z poslední opravené
`charuco_calibration.py` (řešení OpenCV 5.0 `drawDetectedCornersCharuco`).

## Instalace

1. Bezpečně ukonči běžící `main.py`. Nespouštěj dvě instance současně.
2. Rozbal obsah archivu **do** `rob_seme/Vojtěch/rob_crs93_v4/`.
3. Nahraď soubory `extensions/charuco_calibration.py` a `app/camera.py`.
   Druhý soubor přidává zámek pro čtení a otevírání/zavírání Basler,
   aby paralelní `loopcapture` nekolidovalo s `camera snap` nebo přepínáním režimů.
4. Testy v `tests/` jsou volitelné. Data, jiné extensions, `config.json`,
   Prokopův model ani ovládání robota se nemění.
5. Spusť `python main.py` ve stejném virtuálním prostředí jako dosud.

## Jak sbírat novou kalibraci

```text
camera open
load charuco_calibration
call charuco_calibration archive
call charuco_calibration loopcapture start 2 30
call charuco_calibration loopcapture status
call charuco_calibration loopcapture stop
call charuco_calibration list
call charuco_calibration calibrate 20 15
```

- `archive` **přemístí staré zdrojové obrázky do podadresáře** `data/captures/charuco/archive/<datum>/`. Nic nemaže. Použij jen, chceš-li mít čistou novou dávku. Staré obrázky ve zdrojové složce by se jinak do další kalibrace znovu započítaly.
- `loopcapture` lze zapnout i zkráceně `call charuco_calibration loopcapture`; výchozí interval je 2 s a maximum 40 uložených obrázků.
- Interval musí být 0,5–60 s; maximum 1–200 **uložených** snímků. Smyčka se automaticky vypne po dosažení maxima nebo ji ukončíš `stop`. Nepokládej desku stále na stejné místo: opakované snímky nevylepší kalibraci.
- Smyčka snímky detekuje, přeskočí obrázky s méně než 16 validními rohy nebo téměř stejnou polohou rohů jako u posledního uloženého pohledu (medián změny <12 px). Posuzuje **pohyb v obraze, nikoli skutečnou 3D různorodost** – nakláněj desku kolem obou os a měň její vzdálenost.
- Na pozadí se neukládá overlay každého snímku (méně dat a výpočtů). Pro označený obrázek použij `call charuco_calibration inspect`, který zobrazí nejnovější snímek a uloží anotovanou kopii.
- Při změně režimu kamery nebo `unload charuco_calibration` se smyčka ukončuje. `stop` počká, až doběhne právě prováděný snímek.
- Při `camera replay DIR` je povolený běh, ale opakující se obrázky budou správně označeny jako duplicity.

## Diagnostika po calibrate

Příkaz `call charuco_calibration calibrate 20 15` **vždy po úspěšném výpočtu OpenCV vypíše**:
- úplnou matici `K`, `fx`, `fy`, `cx`, `cy` a `fx/fy`,
- všechny koeficienty zkreslení a globální chybu RMS,
- pro každý použitý snímek reprojekční RMS a odhad celkového náklonu roviny desky a složek X/Y,
- konkrétní důvody přijetí/odmítnutí.

Diagnostika se zapisuje do `data/calibration/camera_intrinsics_diagnostics.json` **i při zamítnutí** (pokud nebylo možné `K` vypočítat, je zde `null`). Zobrazíš ji znovu pomocí
`call charuco_calibration diagnostics`.

**Pouze přijatá** kalibrace se zapisuje do `data/calibration/camera_intrinsics.json`. Zamítnutí nikdy nepřepíše předchozí platnou kalibraci.

**Pozor:** Náklony jsou odhadnuté s vypočtenou maticí kamery. Pokud je `K` nepravděpodobná, nejsou absolutní hodnoty náklonů spolehlivé. RMS samo o sobě neprokazuje dobrou kalibraci.

## Offline ověření

V prostředí s OpenCV a ostatními závislostmi:

```bash
python -m unittest discover -s tests -v
```

Prošlo 47 testů včetně 5 nových testů (bez přístupu k fyzickému robotovi/Basler). Ve školní stanici s OpenCV 5.0 ještě nebyla nová smyčka testována.
