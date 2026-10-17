# CRS93: kalibrace homografii a Prokopuv 6. kloub

Tento adresar obsahuje hotovou integraci homografie do stavajiciho hostitele `main.py`.
Driv pouzivane `charuco_calibration`, `camera_intrinsics`, `eye_to_hand` byly ze
zdrojovych souboru a nabidky `load all` odstraneny. Homografie nepotrebuje
intrinsics ani extrinsics kamery; slouzi k prevodu pixelu **do jedne roviny Z**.

## Spusteni

V terminalu ve slozce `Vojtech/rob_crs93_v4` (ve Windows slozka `Vojtěch`):

```powershell
python -m pip install -r requirements.txt
python main.py
```

Standardni prikazy hostitele (`mode`, `robot`, `camera`, `load all`, `pview start`)
zustaly. Prikaz `homo` automaticky nacte `homography_calibration` do pameti,
stejne jako `pview start` pro vizualizer. `homo help` vypise podrobnosti.

## Kalibrace krok po kroku

1. **Fyzicky bezpecne priprav robota a kameru.** Postupuj dle pravidel stanice.
   `mode real nohome` smi byt pouzito **jen** pokud robot zustal referencovany;
   jinak jen opravnene proved `mode real home` (pohybuje skutecnym robotem).
   `homo capture` nepracuje v simulaci, protoze potrebuje skutecne dvojice.
2. Upevni jednoznacny vizualni terc na koncovy efektor (treba jasne kontrastni
   kolecko), ktery kamera vidi. Prave **stred terce** je kalibrovany bod.
   Pokud neni ve flange/TCP pocatku, zmer posun v lokalni soustave TCP a pred
   porizovanim dat zadej treba `homo offset 0.00 0.00 0.05` v METRECH.
   Nepouzivej tyto hodnoty naslepo: jde jen o priklad!
3. Presouvani proved existujicimi `robot jog` / `robot goto` s nastavenymi
   **stale platnymi limity**, kazdy REAL pohyb potrebuje vyzadane potvrzeni.
   Neprovadej nahodne velke skoky, nesmi dojit ke kolizi ani k poruseni limitu.
4. Udrzuj **stred terce ve stejne vysce Z** vuci zakladne robota a pokryj
   8-20+ rozlicnych poloh rozlozenych do cele pouzivane plochy kamery.
   Pracovni prostor cca 48 x 30 cm zacina 22,2 cm pred telem stanice, ale
   tyto rozmery samy o sobe nestanovuji osy robota; nic neodvozuj bez overeni.
5. Na kazde zastavene pozici zadej `homo capture`. Na obrazku klikni presne na
   stred terce a potvrd Enter. Pro jediny kruhovy terc muze pomoci `homo auto`
   (i zde je potreba potvrzeni a rucni korekce). Stisk Escape zrusi snimek.
6. `homo solve` spocita `H` ze skutecnych dvojic **pixel, X/Y ze skutecneho FK**.
   Vypise RMS v pixelech, pocet inlieru, vysku Z a ulozi homografii. Pokud
   pokryti/plosna geometrie nestaci, kalibraci odmitne.
7. `homo map 800 500` prepocita pixel (800,500) na X/Y zakladny v METRECH
   **v kalibrovane rovine Z**. Pred uchopovanim over nezavislymi testovacimi
   body realnou odchylku v milimetrech; RMS fitu sama nestaci.

### Prikazy

```text
homo help            napoveda
homo offset X Y Z    offset stredu terce vuci TCP [m], jen pred snimanim
homo capture         kliknutim vyber stred v nove fotce
homo auto            najdi jediny kruh a rucne potvrdit
homo status          body, Z, offset, RMS
homo solve           vypocet a ulozeni H
homo map U V         pixel -> world (x,y) [m] pri zkalibrovane vysce
homo load            obnov bodova data z disku
homo undo            zrus posledni bod
homo clear           zrus vsechny body pro novou kalibraci
load all             nacti diagnostiku, homografii, Prokop vizualizer
pview start          zobraz Prokop model; bez pohybu robota
```

Poznamka: `homo clear` nema za ukol mazat drive vypoctenou `.npz`, ale
**zablokuje pouziti stareho H**, dokud neprobehne nove `homo solve`.
Stare snimky terce odstraní, ostatni data ne.

## Ukladani a predpoklady

- `data/homography/samples.json`: korespondence, rozliseni, offset, stav dat;
- `data/homography/sample_###.png`: jednotlive fotky;
- `data/homography/homography.npz`: matice `H` pro `pixel -> robot XY`.
- Zdroj obrazu ma pri pouziti stejne rozliseni a vyrez jako behem kalibrace.
- Kamera ani zorny uhel se nesmeji zmenit. Homografie neresí opticke zkresleni;
  velke zkresleni lze zjistit odchylkou validacnich bodu.
- Rovina **neni** libovolna prostorova vyska. Pokud je terc v Z=0.05 m,
  pixel predmetu na stole v Z=0 m nedava touto H korektni XY.
- Oprava sestého kloubu Prokop vizualizeru je v `../../Prokop/view_models.py`.
  Pouze `robot_vis.update_cfg(display_q(q_now))` pouziva offset `+pi`.
  Nemení se actual q, realne FK/IK ani prikazy motorum.

## Testy bez robota

```powershell
python -m pytest tests -q
```

Testy jsou jen offline: nenahrazuji overeni pozice terce, kolizi a realne
chybove kalibrace na stanici CRS93.
