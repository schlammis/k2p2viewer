#define MyVersion FileRead(FileOpen("version.txt"))

[Setup]
AppName=K2P2 Viewer
AppVersion={#MyVersion}
AppPublisher=schlammis
DefaultDirName={localappdata}\k2p2viewer
DefaultGroupName=K2P2 Viewer
OutputBaseFilename=k2p2viewer_setup_{#MyVersion}
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest

[Files]
Source: "dist\k2p2viewer\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\K2P2 Viewer"; Filename: "{app}\k2p2viewer.exe"
Name: "{userdesktop}\K2P2 Viewer"; Filename: "{app}\k2p2viewer.exe"

[Run]
Filename: "{app}\k2p2viewer.exe"; Description: "Launch K2P2 Viewer"; Flags: postinstall nowait
