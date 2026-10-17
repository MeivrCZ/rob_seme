# Obrazově řízená homografie CRS93 s hoopem

Tento režim nahrazuje pevnou XY mřížku **mřížkou v pixelech**. Robot nepotřebuje předem znát světové X/Y hranice zorného pole. Při malých zkušebních pohybech odhadne lokální Jacobian `d(u,v)/d(X,Y)`, potom s kamerou v uzavřené smyčce koriguje každý další krok. Každý zpracovaný snímek patří k zastavenému robotu a skutečné FK pozici středu hoopu.

## Důležité předpoklady

- Stůl má fyzicky ověřenou výšku `TABLE_Z` v soustavě CRS93 BASE. Číslo `-0.0248` z Prokopa je **odhad**, nikoliv bezpečně změřená výška.
- Skutečný hoop je namontován podle modelu a jeho fyzický offset vůči přírubě je ověřen. Nominální předpoklad: `0.13491 0 0.010` m.
- Robot je před `guided plan` **už v požadované výšce** (výška skutečného středu hoopu, nikoli příruby), s vodorovným hoopem, a celý prstenec je vidět v kameře. Režim ho z neznámé polohy NEHLEDÁ naslepo.
- Jedna výška/rovina => homografie platí pouze pro body této výšky. Známe-li výšku detekovaných bodů objektu, můžeme zvolit stejnou kalibrační rovinu.
- Real-world překážky a kabely nemusejí být zahrnuté v geometrii. Vizualizace a softwarový model nejsou bezpečnostní ochrana; člověk sleduje průjezd u fyzického E-STOPu.

## Příkazy

```text
mode real nohome    # pouze pokud je reference opravdu platná
load all
camera open
homo clear
homo offset 0.13491 0 0.010
homo scan test
homo scan guided plan -0.0248 10 5 5
homo scan guided status
homo scan guided run
homo status
homo map 960 600
```

`-0.0248` je na řádku jen jako **ilustrační hodnota** z modelu, která se musí nahradit skutečným měřením; bez něj skutečný sken nespouštěj. `guided plan` vůbec nepohne robotem: ověří 10cm výšku hoopu, orientaci, modelový odstup celé geometrie od stolu a přítomnost detekovatelného kruhu. Pokud zůstává hoop z minulého neúspěšného pokusu uříznutý na kraji, nejdříve ho bezpečně přesuň do polohy, ve které je celý viditelný.

`guided run` pořídí dvě nezávislé malé pohybové sondy pro odhad pixelového Jacobianu, postupně najíždí směrem do středu kamery a dál do mřížky 5×5 bodů. Každý povolený pohyb je maximálně 12 mm v XY a prochází fyzicky stejnými kontrolami jako ostatní robotické příkazy. Po každém pohybu znovu detekuje střed a opraví Jacobian (průběžná korekce). Pokud kruh zmizí nebo se stane nespolehlivým, robot se vrací **na bezprostředně předchozí ověřenou kloubovou polohu**, zkontroluje opětovnou viditelnost a zmenší krok; pokud se návrat nepovede, sken okamžitě ukončí.

Neproveditelná IK, úhly mimo kloubové limity nebo porušení odstupu od stolu se nikdy nepřekračují; příslušný cíl sken přeskočí. Na závěr se při dostatku bodů a prostorovém pokrytí spočítá `homography.npz`; při nedostatečném pokrytí zůstávají snímky a korespondence dostupné pro diagnostiku, bez platné homografie.

## Limity

- Automaticky nelze zajistit úspěch, pokud robot nemá pohybový rozsah potřebný pro pokrytí zorného pole, není zkalibrovaný model montáže hoopu nebo kamera nemá ostrý kontrastní obraz.
- Kamera nepozná výšku stolu ani skutečné překážky z obrazu hoopu. Výška a bezpečnost se nikdy „nedoučí“ ze ztráty detekce.
- Na fyzickém zařízení se tento režim **dosud netestoval**; před rutinním provozem proveď dozorovaný první experiment.
