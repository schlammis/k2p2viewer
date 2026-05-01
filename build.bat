pyinstaller ^
  --paths C:\Users\schlammi\AppData\Local\miniforge3\envs\sci_env\Library\bin ^
  --onedir ^
  --noconfirm ^
  --splash .\src\K2splashscreen.png ^
  -i .\src\k2p2viewer.png ^
  --hidden-import PIL ^
  --collect-submodules PIL ^
  --exclude-module PySide6 ^
  --exclude-module PyQt6 ^
  --exclude-module PySide2 ^
  .\src\k2p2viewer.pyw

copy config.ini dist\k2p2viewer\config.ini
