# CRS93: homografie s ChArUco + opraveny Prokopuv 6. kloub

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

## Doporuceno: ChArUco deska -> homografie v robotich souradnicich

Puvodni desku **vyuzijeme**, ale **ne pro intrinsics/extrinsics**. Je to plošny
referencni vzor pro primou homografii `pixel -> CRS93 (X,Y)`.
Sablona z puvodniho projektu: **10 x 14 ctvercu, DICT_4X4_100, legacy
pattern, 117 vnitřních rohů**. Rozmery tisku ale musi byt fyzicky zmerene.

1. **Poloz desku naplocho a bez pohybu na stul**, do zorneho pole kamery.
   Musi lezet v rovine konstantniho Z vuci bazi CRS93 (deska nesmi byt sikmo).
   Jakmile zachytis snimek, **uz desku nepresouvej** az do dokončení solve.
2. Spust `python main.py`; pro skutecny obraz proved `camera open` (bez
   automatickeho pohybu robota). Volitelne `camera replay DIR` pro test dat.
3. Pomoci pravitka zmer skutecnou delku strany bileho/cerneho **ctverce** a
   strany **ArUco markeru**, napr. cisla 20 a 15 jsou *pouze priklad*:
   ```text
   homo board config 20 15
   homo board capture
   homo board status
   ```
   V `data/homography/charuco_ids.png` otevri obrazek s vykreslenymi ID rohu.
   `capture` automaticky detekuje rohy, nemusis s papirkem tocit ani klikat.
4. Vyber **alespon 3 nekolinearni, navzajem vzdalene** vnitrni rohy (jejich
   ID jsou v obrazku). Pro urceni jejich souradnic v baze robota mas moznosti:
   - **Teach fyzickym robotem**: opravneny clovek mimo tento modul postavi
     skutecnou *znamou spicku TCP* presne do daneho vnitrniho rohu (ve stejne
     XY a Z poloze). Pak po uplnem zastaveni prikaz:
     `homo board teach ID` (napr. `homo board teach 8`). Tento prikaz **jen cte
     FK**, s robotem nijak nehybe. Pokud spicka nelezi v TCP, nejdriv zmer a
     nastav `homo offset X Y Z` ve **souradnicich TCP v metrech**.
   - **Rucni souradnice**: pokud presne znas xyz daneho rohu v baze CRS93,
     zadej `homo board anchor ID X Y Z` (v **metrech**).
   U vsech tri bodu musi **Z odpovidat rovine povrchu desky**, jinak bude
   vysledna homografie prirazena ke spatne vysce.
5. Spust `homo board solve`, pote `homo status` a `homo map U V`. Over vice
   **nezavislych** bodu v mm. Snimek a souřadnice jsou ulozeny v
   `data/homography/` a znovu se nactou po restartu.

### Strucny seznam prikazu

```text
camera open                             otevri skutecnou Basler kameru
homo board help                         vsechny prikazy ChArUco
homo board config SQUARE_MM MARKER_MM   rozmery vzoru [mm]; legacy 10x14
homo board capture                      automaticky snimek + detekce a ID
homo board status                       nalezena ID a referencni body
homo board teach ID                     referencni XYZ ze stojiciho REAL TCP
homo board anchor ID X Y Z              referencni XYZ rucne [m]
homo board solve                        vypocti H a ulozi ji
homo board clear                        smaz snimek a reference, nech rozmery
homo map U V                            prepocti pixel na XYZ roviny (X,Y)
```

**Omezeni:** Homografie funguje pouze pro danou rovinu Z; meni-li se vyska
objektu, meni se i spravny prepočet. Proto je vhodne kalibrovat desku v
rovine, v niz chces detekovat body objektu. Pokud deska pokryva jen cast
zorneho pole kamery, program zamerne **odmitne prepocet mimo oblast desky**
(misto nebezpecne extrapolace do nezmerene casti). Pro plne pokryti plochy
48 x 30 cm potrebujes dostatecne velkou desku nebo dalsi reference v
jejich castich. Bez znalosti polohy desky **v souradnicich robota** nelze z
jedineho snimku vypocitat pixely primo na CRS93 XYZ; 3 referencni body jsou
opravdu nezbytne. Kalibrace nepocita korekci optickeho zkresleni objektivu.

---

## Alternativa: kalibrace kruhovym tercem pripevnenym na robotovi

Nasledujici starsi postup stale funguje, pokud nechces pouzit ChArUco.

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
