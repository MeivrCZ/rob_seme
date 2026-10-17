# CRS93: automatická homografie s kruhovým hoopem

**Pro tuto kalibraci NEPOTŘEBUJEME ChArUco ani intrinsics.** Robot s pevně
namontovaným kruhovým hoopem projde plánované body v jedné rovině, Basler
najde kruh a algoritmus získá pixel → X/Y (v souřadnicích BASE CRS93).

## Význam 1 cm

Požadavek je interpretován jako **nejmenší přípustný odstup pohyblivých
článků a hoopu od horizontální roviny stolu**: `Z >= TABLE_Z + 0.010 m`.
Upevněná základna robota je logická výjimka. Kontrola zahrnuje STL geometrii
článků 1–6 z `Prokop/crs97.urdf`, polohy kloubů 2–6 a konzervativní obálku
celého hoopu (37 mm okolo úsečky mezi přírubou a středem kruhu).
Před pohybem se MUSÍ shodovat FK Prokopova modelu s FK řadiče CRS93.
Jestli model nesouhlasí, kalibrace se bezpečně zablokuje.

**To není certifikovaná prevence kolizí:** nepozná předměty na stole,
nářadí, lidi, klec, kabely, chybu upevnění hoopu, nerovný stůl ani skutečnou
trajektorii při zpoždění pohonů. Při fyzickém testu musí zůstat obsluha u
stanice a mít přístup k fyzickému E-STOP. Proveďte nejprve průjezd ve větší
výšce bez překážek. `Ctrl+C` NENÍ zaručené nouzové zastavení motorů.
`mode real home` a `robot release` jsou zvláštní hardwarové operace;
homing a brzdění se nedají ověřit touto kontrolou trajektorie.

## Předpoklady a rozměry

- Skutečný **Z povrchu stolu v soustavě BASE robota** (v metrech). Není to
  libovolná výška nad stolem. Zjisti ji fyzickým měřením.
- Skutečný offset středu kruhu od TCP/příruby v lokálním souřadném systému
  TCP **XYZ v metrech** (příkaz `homo offset X Y Z`). Z výkresu je známa
  vzdálenost 135 mm od středu montážních otvorů, ale směr/posun vůči TCP
  závisí na připevnění hoopu. Nehádej znaménka ani osy.
- Správně referencovaný robot, funkční kamera, prázdná pracovní oblast,
  ověřené mechanické dorazy a E-STOP.
- Poloha hoopu při kalibraci musí být viditelná kamerou, musí zůstat ve
  stejné rovině a optimálně zůstat kruh přibližně čelní vůči kameře.

## Zapnutí skutečného pohybu

V existujícím `config.json` **přidej jen** sekci níže; ostatní nastavení zachovej:

```json
"motion": {"real_motion_enabled": true}
```

**Nezapínej na fyzické stanici před ověřením podmínek výše.** Hardwarové
limity úhlů ze `ctu_crs` se nadále kontrolují; odstranili jsme pouze
aplikací vnucovaný limit 3° na jeden příkaz a požadavek zadávat `MOVE`
pro každý `jog`/`goto` a automatickou kalibraci.

## Postup

1. V projektu `rob_seme/Vojtěch/rob_crs93_v4` spusť `python main.py`.
2. Připoj známým způsobem robota (pozor na fyzický pohyb při homingu), otevři
   kameru a ověř stav: `mode status`, `robot limits`, `camera status`.
3. Nastav přesný offset středu hoopu (XYZ vůči TCP):
   `homo offset X Y Z` — **X,Y,Z nahraď změřenými hodnotami v metrech**.
4. Zkus `homo auto` v jedné stojící pozici; automatické HoughCircles musí
   opravdu najít právě hoop, ne jiné kroužky v okolí. Když ne, uprav osvětlení
   nebo kontrast, nespouštěj zatím automatický průjezd. `homo auto` vyžaduje
   Enter pro výběr detekce, ale **není to potvrzení pohybu**.
5. Vymaž starou kalibraci: `homo clear`.
6. Zadej bezpečný obdélník pro střed hoopu **v souřadnicích BASE robota**:

```text
homo scan plan XMIN XMAX YMIN YMAX TABLE_Z 10 5 5
```

`TABLE_Z` je skutečná výška desky stolu vůči robotovi v **metrech**,
`10` je požadovaná výška **středu hoopu** 10 cm nad touto rovinou a
`5 5` znamená 25 poloh v mřížce. Zadaný obdélník musí fyzicky patřit do
oblastí viditelných kamerou a dosažitelných robotem. 48 × 30 cm je
**velikost obrazu pracovní plochy**, nikoliv dostatečná definice
XMIN/XMAX/YMIN/YMAX vůči BASE.

`plan` **s ničím nepohne**. Ověří IK, mechanické limity, shodu URDF/FK,
trajektorie jednotlivých článků a minimální vzdálenost 1 cm od plochy.
Pokud kontrola neprojde, **nespouštěj RUN** — oprav měření/rozsah/orientaci.

7. Zkontroluj plán v `homo scan status`. Udržuj osobní dohled nad robotem.
   Po fyzické kontrole celé trasy spusť:

```text
homo scan run
```

`run` už **nečeká na textové potvrzení**. Projede jednotlivé ověřené
mezikroky, na každé pozici vyfotí hoop, automaticky detekuje kruh, uloží
dvojici (pixel, skutečný XYZ středu hoopu). Na konci sám zavolá `homo solve`
a uloží `data/homography/homography.npz`.

Pokud v jednom snímku kruh nejde jednoznačně detekovat, **neposílá další
pohyb**, uloží chybový snímek a nechá již správné dvojice uložené.
Před novým startem `homo clear` a znovu `homo scan plan ...`.

8. Zkontroluj `homo status` a přesnost v **nezávislých** měřených bodech;
   pouhá RMS chyba z kalibračních bodů nestačí.

## Oprava planovani z home (2026-10-17)

Puvodni `hoop_auto_scan.py` drzel orientaci priruby z okamziku planovani.
Z `home` to pri nominalnim namontovani hoopu znamena prakticky svisly
krouzek a caste chyby `IK/limity`. Aktualni verze nastavuje vodorovnou
rovinu hoopu podle nominalniho Prokopova adapteru (normal krouzku miri
svisle nahoru, smer od priruby ke stredu krouzku vede smerem k zakladne).

Pro staticke overeni mřížky v modelu se osvedcil **jen navrh k PLAN**:

```text
homo scan plan 0.28 0.43 -0.17 0.17 -0.0248 10 5 5
homo scan status
```

Tato cisla NEJSOU overene fyzicke souradnice. `TABLE_Z=-0.0248 m`
pochazi z neovereneho placeholderu v Prokop/placement.py. Pred prikazem
`homo scan run` zmer presnou vysku desky stolu v BASE robota, zkontroluj
skutecnou montaz hoopu, orientaci, volny pohyb celeho ramene a pokryti
kamerou. Spatne nastaveni vysky stolu nebo montaze invaliduje simulovanou
kontrolu odstupu. Vypocet IK na modelu nezarucuje dosazitelnost realneho
hardware ani bezkoliznost stanice.

Pro danou pevnou orientaci krouzku muze byt rozsah uzsi nez cele pole
kamery; body mimo zmerenou oblast vyzaduji dalsi overeni presnosti nebo
dalsi kalibracni postup s jinou orientaci a vice referencnimi body.

## Omezení

Homografie platí **jen pro jednu kalibrovanou rovinu Z středu hoopu**.
Jestli se horní/pozorovaný bod objektu nachází v jiné výšce, objeví se
paralaxa. Měření rozměrů předmětu neodstraní nutnost správně zvolit rovinu.

Není zaručeno, že všechny naplánované body jsou skutečně viditelné; IK a
kontrola odstupu nevědí, co je v obraze. Když HoughCircles vrací falešné
pozice, může vzniknout chybná kalibrace i při malé RMS, proto nezávislá
validace a vizuální inspekce snímků zůstávají nutné.
