# CRS93 v3 – přepínání REAL / VIRTUAL za běhu

Toto je rozšíření předchozí verze. Kompletní README.md obsahuje i původní kalibrace.

## Instalace

    python -m pip install -r requirements.txt
    python main.py

## Nejdůležitější příkazy

    mode status
    mode virtual
    mode virtual data/captures/intrinsics
    mode real nohome
    mode real home
    mode virtual
    mode off

- `mode virtual [DIR]`: používá CRS93(tty_dev=None) pro kinematiku; se zadaným adresářem přehrává uložené PNG/JPG snímky. Žádná fyzická zařízení se nepřipojují. Bez snímků je kamera zavřená (virtuální kamera nesyntetizuje obraz).
- `mode real nohome`: otevře Basler, potom připojí fyzického robota bez homingu. Vyžaduje potvrzení `REFERENCE`. Pouze pokud řídicí jednotka zůstala napájená a klouby jsou referencované!
- `mode real home`: otevře Basler, potom připojí fyzického robota s homingem – robot se skutečně POHNE. Vyžaduje potvrzení `HOME`.
- `mode off`: odpojí kameru a robota, nechá ale program a extensions běžet.
- `mode status`: vypíše obě zařízení a aktuální režim. Režim `PARTIAL` znamená například jen otevřenou Basler kameru pro vnitřní kalibraci, zatímco robot je odpojený.

Načtené extensions i soubory v `data/calibration` zůstávají beze změny. Při přechodu REAL -> VIRTUAL -> REAL -> VIRTUAL se obnoví poslední virtuální konfigurace kloubů a poslední použitý adresář snímků. Přepnutí do VIRTUAL se odmítne, pokud fyzický robot hlásí pohyb. Při neúspěšném otevření Basler zůstane VIRTUAL aktivní; při selhání připojení reálného robota se program pokusí virtuální backend obnovit.

## Typická práce při vývoji

    mode virtual data/captures/intrinsics
    load diagnostics
    call diagnostics pose
    robot virtual-set deg 0 -20 10 0 15 0
    reload diagnostics
    mode status

Na stanici CRS93 pak přepnutí:

    mode real nohome
    camera snap
    mode virtual

`mode real nohome` je bezpečné jen při platné referenci. Jinak je nutné `mode real home` po zajištění pracoviště, přítomnosti obsluhy a potvrzení.

## Zásadní poznámky k perception a kalibraci

- Fotografie stolu pořízené telefonem jsou pouze orientační. Nesmí být považovány za snímky pevné kamery Basler pro její intrinsics / extrinsics.
- Odhad výšky Basler cca 1,15 m je přibližný. Není náhradou za výpočet kalibrace.
- Deska na fotografii **vypadá jako ChArUco**, ale přesná identita vzoru, slovník markerů, rozměry čtverců a markerů ještě nejsou ověřeny.
- Původní extension `camera_intrinsics.py` momentálně používá klasickou OpenCV detekci *šachovnice*, nikoliv speciální ChArUco detektor. Pro konkrétní ChArUco desku budeme po zjištění rozměrů a ID/slovníku potřebovat samostatnou detekci a odpovídající kalibrační postup.
- Offline režim **není grafický ani fyzikální simulátor** a automaticky negeneruje syntetické obrazy. Virtuální kinematika poskytuje pózu koncového členu, replay používá uložené reálné snímky.
- V reálném režimu mohou jiné aplikace, například Pylon Viewer, blokovat otevření kamery. Před `mode real ...` je zavřete.
- Extensions musí používat `ctx.robot` a `ctx.camera` až v době volání. Pokud si uloží starý interní objekt robota/kamery napřímo, přepnutí může takový odkaz zneplatnit.

## Testy

    python -m unittest discover -s tests -v

Testy používají náhrady fyzického hardwaru. Žádný robot se při jejich spuštění nepohybuje. Komunikaci s reálnou stanicí je nutné ověřit zvlášť.

---

## Starší dokumentace (v2)

# CRS93 – modulární řízení a offline testování (v2)

Malý **dlouho běžící Python proces** spravuje CRS93 a Basler kameru. Extensions lze upravovat a načítat za běhu bez zavírání hardwarových spojení. Kalibrační výsledky jsou uloženy v `data/calibration/` a snímky v `data/captures/`. **Verze 2** navíc přidává virtuální kinematický model CRS93, offline přehrávání snímků a vnější eye-to-hand kalibraci; stručné příkazy jsou ve druhé části tohoto návodu.

> **Stav:** Připravený startovací prototyp, nikoliv ověřený program na skutečné stanici. Rozhraní odpovídá dokumentaci `ctu_crs`, ale komunikaci s konkrétní stanicí CRS93 jsme zatím neotestovali. Neobsahuje automatický pohyb ani algoritmy plánování pohybu. Virtuální režim není fyzikální simulátor.

## Instalace

Na školním počítači u CRS93 (Linux):

```bash
python -m pip install -r requirements.txt
python main.py
```

`ctu_crs` je podle školní dokumentace již nainstalované. Pro offline experimentování na vlastním počítači nemusí být přístupný robot ani Basler. Příkazy `help`, `load`, `reload`, `calib list` fungují i bez nich.

V `config.json` lze změnit název kamery a cestu k sériovému portu robota (`/dev/mars`).

## První spuštění

```
crs93> help
crs93> load diagnostics
crs93> call diagnostics status
crs93> camera open
crs93> camera snap
crs93> load camera_intrinsics
crs93> call camera_intrinsics help
```

Inicializace robota se záměrně **neprovádí automaticky**:

```
crs93> robot connect home
```

Tento příkaz vykoná **fyzický homing s pohybem**. Jen po bezpečnostním školení, s volnou pracovní zónou a připraveným E-STOP. Pokud řídicí jednotka **zůstala zapnutá a robot má platnou referenci**, lze místo toho použít:

```
crs93> robot connect nohome
```

Tím se volá `initialize(home=False)`; je to tvoje odpovědnost potvrdit stav reference. Rozpojení spojení (`robot disconnect`) robotem samo nehýbe. Při ukončení programu se pouze zavřou spojení; pro bezpečné zaparkování do výchozí pozice existuje **samostatný pohybový** příkaz `robot soft-home`.

### Hot-reload

```
crs93> load diagnostics
crs93> call diagnostics status
# Změň extensions/diagnostics.py ve VS Code.
crs93> reload diagnostics
crs93> call diagnostics status
crs93> unload diagnostics
```

Robot a kamera se neodpojují; `reload` vymění jen konkrétní extension. `setup(ctx)` vrací objekt s `run(argv)` a volitelným `close()` pro uvolnění vlastních zdrojů. Dlouhotrvající procesy/threads extensions musí zastavit ve svém `close()`. Poznámka: moduly importované jinými extensions se automaticky nerekurzivně nenačtou znovu. Pro komplexnější plugin systém to lze později rozšířit.

## Vnitřní kalibrace Basler kamery

Použij **rovnou vytištěnou šachovnici** o známé velikosti čtverce. Pro kalibraci si připrav např. 15–25 odlišných záběrů s různým natočením a polohou šachovnice. Všechna nastavení kamery (rozlišení, zaostření, zoom) musí být stejná jako při běžném používání.

```
crs93> camera open
crs93> load camera_intrinsics
crs93> call camera_intrinsics capture
crs93> call camera_intrinsics capture
# ... opakuj pro různé polohy šachovnice ...
crs93> call camera_intrinsics list
crs93> call camera_intrinsics calibrate 9 6 0.025
crs93> calib show camera_intrinsics
```

**Pozor:** `9 6` znamená **počet vnitřních rohů šachovnice** (nikoliv počet čtverců), `0.025` znamená **stranu čtverce 25 mm v metrech**. Tyto hodnoty jsou pouze příklad – dosaď skutečnou šachovnici. Všechny přípustné snímky by měly mít konzistentní rozlišení, patřit ke stejné kameře a nastavení objektivu a zachycovat šachovnici z různých úhlů.

Pokud najde extension méně než 10 detekovaných pohledů nebo vyjde RMS větší než 1.0 px, výsledek se **neuloží**. Práh 1.0 px je orientační ochrana prototypu, ne univerzální garance kvality. Výsledek obsahuje matici `K`, zkreslení `dist_coeffs`, chyby reprojekce, nastavení a seznam použitých záběrů. Uloží se atomicky do `data/calibration/camera_intrinsics.json` a přežije restarty programu.

**Pozor:** toto je pouze **vnitřní kalibrace**. Neurčuje ještě polohu kamery vzhledem k robotovi.

## Další navrhované fáze

1. Ověřit přes školní počítač připojení CRS93, získání `get_q()`, `fk()` a snímku z kamery (žádný autonomní pohyb).
2. Kalibrovat intrinsics kamery, uložit `K` a zkreslení. Ověřit kvalitu na nezávislých snímcích.
3. Protože je Basler kamera **pevná** (*eye-to-hand*), připevníme šachovnici na přírubu, nasbíráme páry snímků a póz a spočteme `T_base_camera`. Implementace je v `extensions/eye_to_hand.py`; musí se ještě ověřit na stanici.
4. Pomocí značení **ArUco 4×4** na dodaných podstavcích určit `T_camera_baseplate` z obrazu. Nutná je **skutečná velikost markerů, jejich ID, přesný slovník a geometrie jejich rohů**.
5. Provázat souřadnicové rámce `robot base ↔ camera ↔ baseplate ↔ puzzle` a porovnat odhad s fyzickým měřením.
6. Až potom řešit 6D polohu objektu z CAD/geometrických příznaků; plánování pohybu a vizualizaci integrují kolegové.

Modely z poslaného `ROB1-2025-hw-main.zip` nejsou do tohoto kódu duplikované; jsou to výrobní podklady pro pozdější kalibraci a lokalizaci. Výkresové rozměry jsou v **milimetrech**, zatímco do programových transformací budeme používat **metry**.

### Souřadnicové transformace

Pro staticky umístěnou kameru a detekci markeru/objektu platí při konvenci `T_A_B` = transformace souřadnic z B do A:

```
T_base_object = T_base_camera @ T_camera_object
```

Pro polohu objektu vůči podstavci lze uvažovat `T_base_object = T_base_plate @ T_plate_object`. Bez znalosti `T_base_camera` (externí kalibrace) není monocular RGB snímek sám o sobě dostatečný pro převod do robotických souřadnic.

## Struktura

```
main.py                        # Konzolový host – běží celou relaci
config.json                    # Hardware identifikátory
app/robot.py                   # Jedna instance CRS93
app/camera.py                  # Jedna instance BaslerCamera
app/store.py                   # Trvalé ukládání kalibrací
app/extensions.py              # load / unload / reload
extensions/diagnostics.py     # Testovací extension
extensions/camera_intrinsics.py # První reálná kalibrace

# Vytváří se za běhu:
data/calibration/               # Matice kamery a další kalibrace
data/captures/intrinsics/       # Kalibrační snímky
data/captures/snapshots/        # Běžné snímky
```

Bezpečnost: program neřeší kolize ani bezpečnostní monitorování trajektorií; příkazy pohybu a homingu používej jen při dodržení pravidel stanice. Během práce v kleci postupuj přesně podle manuálu robotické stanice.

Zdroje: [návod CRS93](https://robotics-robot-stations.readthedocs.io/en/latest/stations/crs.html), [CRS API](https://robotics-robot-stations.readthedocs.io/en/latest/autoapi/ctu_crs/crs_robot/index.html), [BaslerCamera API](https://robotics-robot-stations.readthedocs.io/en/latest/autoapi/ctu_crs/basler_camera/index.html), [OpenCV kamera](https://docs.opencv.org/4.12.0/dc/dbb/tutorial_py_calibration.html).

---

# Verze 2: Virtuální CRS93 + přehrávání snímků + eye-to-hand

**Co přibylo:** `app/robot.py` má nyní dvě implementace přístupu k robotu pod stejným API: `real` a `virtual`. Virtuální režim počítá přes **skutečnou kinematiku** `CRS93(tty_dev=None)` z `ctu_crs`. Simulovaný stav kloubů je uložen jen v RAM a nijak nehýbe fyzickým robotem. Přibylo také přehrávání snímků z adresáře a `extensions/eye_to_hand.py`.

Pro offline počítač nainstaluj `python -m pip install -r requirements.txt` (`ctu_crs` je nutné i pro přesnou virtuální FK). Pokud instalace `ctu_crs` na daném OS selže, příkazy `load`, `unload`, `reload`, `calib list` a `camera replay` fungují i bez něj; virtuální robot pak bude dostupný až po instalaci knihovny.

### Práce doma bez robota

```text
crs93> robot connect virtual
crs93> robot status
crs93> robot virtual-set deg 0 -20 10 0 15 0
crs93> robot status
crs93> load diagnostics
crs93> call diagnostics pose
crs93> reload diagnostics
crs93> robot soft-home
crs93> robot disconnect
```

Pozor: `robot virtual-set` nastaví stav *okamžitě*. Není to fyzikální simulace pohybu, nehlídá limity kloubů ani kolize. Není to bezpečnostní kontrola trajektorie. **Nikdy nepřenášej naslepo virtuálně odzkoušený pohyb na skutečného robota.**

### Offline perception z nasbíraných snímků

Nejprve nasbírej skutečné obrázky na stanici (nebo přidej libovolné `.png` do adresáře):

```text
crs93> camera replay data/captures/intrinsics
crs93> camera status
crs93> camera snap
crs93> camera close
```

Při `camera replay` vrací `capture()` cyklicky další obrázek ze zvoleného adresáře, aniž by připojovala Basler. V režimu replay nelze ověřit aktuální polohu skutečného robota ani vytvářet platná kalibrační měření polohy kamery; snímky lze ale zpracovávat stejnými algoritmy jako živý obraz. Pro experiment s kamerou použij libovolný adresář s `.png` / `.jpg`.

### Vnější kalibrace eye-to-hand (pevná Basler + šachovnice na přírubě)

1. Proveď vnitřní kalibraci, aby existovalo `data/calibration/camera_intrinsics.json`; na kameře už neměň zoom, fokus ani rozlišení.
2. Připevni **pevně a kolmo určenou šachovnici** k přírubě robota. Po celou dobu kalibrace se nesmí vůči přírubě posunout. Změř skutečnou velikost čtverců v metrech. Robotem pohybuj pouze podle pravidel stanice (bezpečnostní prostor, proškolení, E-STOP). Aplikace žádné kalibrační pohyby automaticky neprovádí.
3. Na školní stanici:

```text
crs93> robot connect nohome
crs93> camera open
crs93> load eye_to_hand
crs93> call eye_to_hand capture 9 6 0.025
# Ručně bezpečně přemísti rameno a počkej na úplné zastavení.
crs93> call eye_to_hand capture 9 6 0.025
# ... opakuj ~15 až 25x s výrazně různými polohami a orientacemi
crs93> call eye_to_hand list
crs93> call eye_to_hand solve
crs93> call eye_to_hand show
```

`robot connect nohome` použij **jen tehdy**, pokud řídicí jednotka nebyla vypnuta a robot má nadále platnou referenci; jinak je nutný homing s pohybem (`robot connect home`) za bezpečných podmínek.

`9 6` = **vnitřní rohy** šachovnice; `0.025` = 25 mm; uprav pro skutečný kalibrační terč. Jednotky FK, PnP i transformací jsou **metry**. 3D modely v dodaném ZIPu jsou v **milimetrech**, pro perception je proto nutná konverze mm→m.

Každý záznam do `data/captures/eye_to_hand` obsahuje snímek, radiany kloubů, `T_base_flange`, `T_camera_target` a chybu reprojekce PnP; dvojice dat se zachovají i po `unload`/`reload` a ukončení aplikace. `call eye_to_hand solve` z uložených dvojic vypočítá `T_base_camera` a `T_flange_target`; **nevyžaduje otevřeného robota ani kameru**, jen uložené intrinsics a data. Výsledek se uloží do `data/calibration/eye_to_hand.json`, pokud jsou vnitřní konzistence pod orientačními prahy 10 mm a 2°. Tyto prahy *nezaručují* přesnost v reálném světě: validuj na nezávislém známém bodě/markeru.

**Důležité:** pokud se šachovnice otočí vůči uchycení nebo detektor nekonzistentně očísluje symetrické rohy, bude měření chybné. Nezačínej bez kontroly výsledné reziduální chyby. Asymetrický ChArUco terč může být později spolehlivější.

### Co to zatím NENÍ

- Žádná 3D grafika (tu podle domluvy vyvíjí kolega), žádná fyzikální simulace ani detekce kolizí.
- STEP modely v dodaném `ROB1-2025-hw-main.zip` jsou hlavolamy a adaptéry, **ne model všech článků CRS93**. Pro 3D vykreslení se bude hodit samostatný geometrický model/URDF robota od vyučujících nebo z ověřeného zdroje.
- Eye-to-hand solver i virtální režim byly otestované **offline** na syntetických datech a mockovaných výstupech. Skutečné propojení se školním CRS93 a jeho Basler kamerou nebylo otestováno.

Testy (bez hardwaru a `ctu_crs`):

```bash
python -m unittest discover -s tests -v
```


## Prokopova vizualizace – extension

Do `extensions/prokop_visualizer.py` je přidaný adaptér, který z našeho hostu spouští a zastavuje externí `Prokop/view_models.py` v samostatném procesu. Prokopovy originální soubory se nemění. `Prokop/` musí být vedle této složky nebo v některé nadřazené složce repozitáře. V rootu sdíleného repozitáře nainstalujte závislosti (`uv sync`) a spouštějte host uv prostředím, např. `uv run python <nase_slozka>/main.py`.

```text
crs93> mode virtual
crs93> load prokop_visualizer
crs93> call prokop_visualizer start
crs93> call prokop_visualizer status
crs93> call prokop_visualizer stop
crs93> unload prokop_visualizer
```

Prokop aktuálně používá model CRS97, zatímco host CRS93. Tato první integrace **nesynchronizuje klouby**; k tomu budeme potřebovat sjednotit model robota a upravit API vizualizace.
