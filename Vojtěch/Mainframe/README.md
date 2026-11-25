# Mainframe pro simulaci CRS93

Spusť Mainframe z kořene repozitáře.

```powershell
uv run "Vojtěch/Mainframe/main.py"
```

Příkazy zadávej ve stejném terminálu a v prohlížeči otevři scénu Viseru na [localhost:8080](http://localhost:8080).

Pokud už na portu 8080 běží jiná simulace, přidej `--port 8082` a otevři odpovídající adresu. Simulátor, kamera, detektor a přepočet souřadnic jsou připravené po startu.

| Příkaz | Výsledek |
| --- | --- |
| `robot status` | Zobrazí polohy kloubů, XYZ příruby a stav pohybu. |
| `robot jog q0 5` | Posune první kloub o 5°. Klouby jsou `q0` až `q5`. |
| `robot goto deg Q0 Q1 Q2 Q3 Q4 Q5` | Zadá absolutní polohu. Pro radiány použij `rad`. |
| `robot wait` | Počká na dokončení pohybu, třeba před focením. |
| `robot soft-home` | Vrátí robot do výchozí polohy. |
| `camera snap` | Uloží snímek ze simulované kamery a aktualizuje panel **Camera** ve Viseru. |
| `aruco detect` | Pořídí snímek, uloží originál i vyznačenou kopii a vypíše polohu základny. Když značky zmizí, použije naměřený model točny a poslední platnou detekci. |
| `homo run` | Projede kalibrační polohy, vyfotí obruč a spočítá projekční matici 3×4. |
| `homo status` | Vypíše aktivní matici a zdroj kalibrace. |
| `homo orbit [N [INTERVAL_S]]` | Vyfotí rotující držák a odhadne jeho dráhu. Výchozí je 10 snímků po 1 s. |
| `homo predict [SEKUND]` | Vypíše odhad středu a natočení v zadaném čase vzhledem k okamžiku příkazu. |
| `puzzle place X Y YAW_DEG` | Posadí vystředěnou základnu na zvolené XY v metrech a natočí ji o zadaný úhel ve stupních. |
| `puzzle static [RYCHLOST nebo demo]` | Změří základnu na stojící točně a projede D k základně a zpět ven. |
| `puzzle spin [RYCHLOST nebo demo]` | Změří rotaci a naplánuje cestu k základně i zpět ven při stále zapnutém otáčení. |
| `help` / `quit` | Zobrazí příkazy / ukončí konzoli a Viser. |

Snímky v plném rozlišení se ukládají do `Vojtěch/Mainframe/captures/`. Detekce používá `ArucoFinder` pro značky základny s ID 1 a 2 a `Homografy` pro převod z obrazu do souřadnic robota.

Mainframe běží v simulaci se zapnutým monitorem kolizí a s limity kloubů.

## Průjezd varianty D

V panelu **Turntable** umísti díl do zorného pole kamery. Pro `puzzle static` ponech **Spin** vypnuté. Pro `puzzle spin` zapni **Spin** a nastav periodu 28 s. Oba příkazy přijímají relativní rychlost robota z rozsahu `(0, 1]`. Výchozí rychlost je `0.4` při stání a `0.8` při rotaci.

Pro opakovatelnou ukázku začni v `q_home` v nové relaci a zadej `puzzle static demo` nebo po zapnutí **Spin** `puzzle spin demo`. Pro rotující ukázku použij periodu 28 s a směr proti směru hodinových ručiček při pohledu shora. Příkaz vystředí základnu na točně na souřadnicích XY `(0.55, 0.0)` m. Pro tuto scénu zkouší napřed dříve ověřenou fázi a znovu ji prověří proti aktuálnímu měření.

U rotace program ještě před prvním pohybem ověří nájezd, časovaný průjezd k základně, výjezd přes volný konec tyče a návrat do čekací polohy. Nejprve zkouší čekací polohy poblíž volného konce, které mají odstup po celou otáčku. V případě potřeby zkusí aktuální nebo výchozí polohu. Během vjezdu i výjezdu zpřesňuje cíle z časovaných snímků. Po úspěchu zůstane obruč v čekací poloze a **Spin** zůstává zapnutý. Při přerušení pohybu simulátor točnu zastaví. Stojící režim plánuje návrat z dosažené polohy po dojezdu k základně. Záznamy jsou v `captures/puzzle_<čas>.json`. U rotace obsahuje `steps` celý průjezd a `depth_step` označuje bod u základny.

### Vlastní umístění po kalibraci

1. Při vypnutém **Spin** spusť `homo run`. `homo status` pak ukáže cestu k naměřené kalibraci.
2. Zadej například `puzzle place 0.54 0.01 90`. Stejné XY lze vybrat posunem točny ve Viseru. Pomocí `aruco detect` můžeš ověřit, že kamera značky vidí.
3. Pro stojící variantu spusť `puzzle static`. Pro rotaci zapni **Spin** a spusť `puzzle spin`. Tyto příkazy použijí zvolenou scénu a aktivní naměřenou kalibraci. Volba `demo` nastavuje ukázkové umístění.

## Kalibrace pomocí obruče

`homo run` při vypnutém **Spin** dočasně odloží držák a tyč za robotem a po kalibraci je vrátí na zvolené místo. Robot projede pevnou mřížku 3 × 3 ve výškách 75, 130 a 190 mm nad stolem. V každé poloze počká na zastavení. Střed obruče v souřadnicích robota se získá z odečtených kloubů a dopředné kinematiky. Střed v pixelech detekuje `Homografy` ve snímku simulované kamery.

Snímky `point_XX.png` a `homography.json` s maticí `P` a naměřenými dvojicemi najdeš v `captures/homography_<čas>/`. Konzole vypíše chybu přepočtu na kontrolních bodech. Po úspěšném průjezdu použije `aruco detect` naměřenou matici do ukončení Mainframu. Nová relace začíná se syntetickou kalibrací simulátoru.

## Dráha rotujícího držáku

V panelu **Turntable** vyber **Random viable placement** a zapni **Spin**. Pro měření můžeš použít výchozí periodu 28 s. V konzoli zadej `homo orbit` a po doběhnutí třeba `homo predict 2`. Během sběru a následné předpovědi ponech točnu běžet stejnou rychlostí a držák na stejném místě vůči točně.

Program používá aktivní homografii a rovinu ArUco značek 64,6 mm nad stolem. Čas snímku odhaduje odečtením latence kamery 0,11 s od času návratu `grab_image`. Výpočet potřebuje aspoň čtyři platné detekce a natočení aspoň o 20°. Mezi dvěma platnými snímky musí točna urazit méně než půl otáčky, aby šel určit směr. U prakticky vystředěného držáku se dráha zjednoduší na bod, natočení se dál předpovídá.

Výstupy jsou v `captures/orbit_<čas>/`. `orbit.png` ukazuje dráhu a naměřené středy na posledním snímku se značkami. `orbit.json` obsahuje vzorky, parametry `TurntableModel` v metrech, radiánech a sekundách, poloměr, znaménkovou rychlost i chybu přizpůsobení. Nulová fáze odpovídá poslednímu snímku s detekcí a predikce v konzoli používá monotónní čas stejného běhu.

Po `homo orbit` si `aruco detect` pamatuje poslední platnou detekci ze série. Každý další snímek se značkou ji může aktualizovat. Při zakrytí značek konzole vypíše odhad XY a natočení pro čas pořízení snímku a stáří poslední detekce. Po nové kalibraci kamery nebo změně polohy či rychlosti točny změř dráhu znovu pomocí `homo orbit`.
