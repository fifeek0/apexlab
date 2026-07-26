; Inno Setup script — builds IRacingSuite-Setup.exe from the PyInstaller
; one-dir bundle (dist/iracing-suite). Compile: iscc packaging\installer.iss

#define MyAppName "iRacing Suite"
#define MyAppVersion "0.1.0"
#define MyAppExeName "IRacingAnalysis.exe"

[Setup]
AppId={{8F4B7E62-52C1-4F2E-9B1E-1RACINGSUITE}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\iRacingSuite
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputBaseFilename=IRacingSuite-Setup
OutputDir=dist
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "dist\iracing-suite\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\iRacing Analysis"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Race Engineer (voice)"; Filename: "{app}\iracing-suite.exe"; Parameters: "engineer --tts"
Name: "{group}\Live Overlay"; Filename: "{app}\iracing-suite.exe"; Parameters: "overlay"
Name: "{group}\Telemetry Agent (auto-import)"; Filename: "{app}\iracing-suite.exe"; Parameters: "agent"
Name: "{autodesktop}\iRacing Analysis"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; Flags: unchecked

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch iRacing Analysis"; Flags: nowait postinstall skipifsilent
