CRS93 v4 - DOPLNENI `robot release`

1) Vypni bezici main.py bez pohybu (v konzoli quit, po kontrole ze robot stoji).
2) Zalozni si existujici projekt, vcetne data/calibration.
3) Rozbal obsah tohoto ZIPu PRIAMO do Vojtech/rob_crs93_v4 (presne tam kde main.py).
   Prepis soubory main.py, app/robot.py, app/modes.py a tests/test_mode_switching.py.
4) Spust python main.py, pripoj robota podle stavu stanice (mode real nohome
   jen pokud je reference stale platna).
5) Zadej robot release a pro schvaleni napis RELEASE.

DULEZITE: release zabrzdi a rozpoji zpetnou vazbu; rameno muze klesnout.
Jen u stojiho robota s volnym prostorem. To neni emergency-stop.
Nekdy je potreba obnovit Arm Power, Motion Stop, zavrit klec; pri chybe motoru
postup dle navodu. Po release na stejnem pripojeni nevykonavej pohyb.
Zmeny v main.py a app/ nelze hot-reloadovat pres reload extensions.
Kalibracni soubory a Prokopuv kod zustavaji beze zmeny.
