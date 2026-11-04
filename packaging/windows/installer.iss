; Inno Setup script for the Windows installer; built by tools/build_windows_app.sh, which passes
; /DAppVersion, /DSourceDir (the PyInstaller output folder), /DOutputDir and /DIconFile.
; Installs per user (no admin prompt), so the in-app updater can rerun it silently.

#define AppName "TV Show Noisemaker"
#define AppExe "TV Show Noisemaker.exe"

[Setup]
AppId={{8F3C2B1E-6A4D-4E7B-9C1F-2D5A7E9B3C41}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=sasha351
AppPublisherURL=https://github.com/sasha351/tvShowNoisemaker
AppUpdatesURL=https://github.com/sasha351/tvShowNoisemaker/releases
DefaultDirName={localappdata}\Programs\{#AppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=TV-Show-Noisemaker-Windows-x64-Setup
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=force
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Clean upgrade: drop the previous version's libraries so no stale files are left behind.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
; Also runs after a silent in-app update, which reopens the app.
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall
