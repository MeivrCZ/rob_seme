# CRS93 — automatické kloubové meze z ctu_crs

Tento patch navazuje na `CRS93_v4_robot_motion_patch.zip`, zachovává naše konzolové ovládání i Prokopovu synchronizaci.

## Instalace

1. Počkej, až robot stojí. V host konzoli použij `quit`. Samotný `quit` nevypíná pohony ani nenahrazuje nouzový stop.
2. **Pro jistotu zazálohuj** původní `main.py` a `app/robot.py`.
3. Rozbal tento ZIP do `rob_seme/Vojtěch/rob_crs93_v4/` a přepiš jen uvedené soubory. Neměň `Prokop/`, `data/`, ani své `config.json`.
4. Spusť `python main.py`.
5. Připoj robota v odpovídajícím režimu (`mode virtual` pro offline kontrolu; `mode real nohome` JEN pokud zůstala platná reference a stanice dovoluje použití bez homingu).
6. Zadej `robot limits` a zkontroluj intervaly. Výpis obsahuje **původní** a **účinné** meze ve stupních.

## Jak se meze počítají

Po vytvoření instance `ctu_crs.CRS93` se automaticky přečtou `q_min`, `q_max` [rad]. Data se ověří (6 prvků, konečná čísla, `min < max`). Použije se výchozí rezerva **2° na obou koncích**. Dřívější `motion.joint_limits_deg` lze nechat `null`, nebo použít jako další omezení — meze knihovny NIKDY nerozšiřuje.

Když `q_min/q_max` chybí nebo jsou neplatné, REAL pohyb je softwarově zablokován (fail closed). Virtuální polohy lze nastavovat i bez reálného hardware. Hodnoty se při opětovném připojení načtou znovu.

## Konfigurace

Do existujícího `config.json` lze přidat, nikoliv přepsat ostatní sekce:

```json
"motion": {
  "real_motion_enabled": false,
  "joint_limits_deg": null,
  "max_step_deg": 3.0,
  "limit_margin_deg": 2.0
}
```

- `real_motion_enabled` je **výchozí false**: žádné fyzické pohyby `jog/goto`.
- `max_step_deg` je nejvyšší rozdíl na každém kloubu pro jeden jednotlivý příkaz `jog/goto` v REAL režimu (výchozí 3°).
- `limit_margin_deg` ubírá rezervu od modelových mezí (výchozí 2°). Rezerva se **nenahrazuje** automatickou detekcí překážek.
- `joint_limits_deg` je volitelné *dodatečné zúžení* `[min,max]` ve stupních (šest dvojic).

`real_motion_enabled: true` nastav jen po ověření bezpečného provozu, fyzických limitů a funkčnosti E-STOP s obsluhou stanice. Ruční výpis `q_min/q_max` sám o sobě bezpečnostní povolení nezakládá.

## Příkazy

```text
mode virtual
robot limits
robot jog q0 1
robot goto deg 0 -20 10 0 15 0
```

V REAL režimu `robot jog/goto` vyžaduje potvrzení `MOVE`; ověřuje limit intervalu a velikost kroku, zda robot stojí, a opětovně kontroluje výchozí polohu po potvrzení. Nepohne v režimu `release` ani po nejasně dokončeném pohybu. VIRTUAL režim je jen kinematický model.

**UPOZORNĚNÍ:** Ani tyto kontroly ani `ctu_crs.q_min/q_max` nekontrolují kolize s podstavci, stolem, klecí ani možnost střetu s osobou. `robot soft-home` používá vlastní pohybové API mimo `robot jog/goto` a vyžaduje samostatný bezpečnostní postup. Při nečekaném pohybu použij fyzický nouzový stop.

## Offline testy

```bash
python -m unittest discover -s tests -v
```

Testováno s mockovanou knihovnou `ctu_crs`, nikoliv pohybem skutečného robota.
