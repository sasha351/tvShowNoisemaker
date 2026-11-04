# PyInstaller spec for the desktop app on macOS (.app) and Windows (folder for the installer).
# Run through tools/build_macos_app.sh or tools/build_windows_app.sh, which set the env below.
import os
import sys

from PyInstaller.utils.hooks import collect_data_files

HERE = os.path.abspath(SPECPATH)
ROOT = os.path.dirname(HERE)
TOOLS = os.environ["TVSN_TOOLS_DIR"]  # ffmpeg, ffprobe, fpcalc fetched by the build script
VERSION = os.environ["TVSN_VERSION"]
ICON = os.environ.get("TVSN_ICON")
NAME = "TV Show Noisemaker"
EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""

a = Analysis(
    [os.path.join(HERE, "app_entry.py")],
    pathex=[ROOT],
    binaries=[(os.path.join(TOOLS, t + EXE_SUFFIX), "bin") for t in ("ffmpeg", "ffprobe", "fpcalc")],
    datas=collect_data_files("customtkinter") + collect_data_files("certifi"),
    hiddenimports=["certifi"],
    excludes=["librosa", "pytest", "static_ffmpeg"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME, console=False, upx=False, icon=ICON)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{NAME}.app",
        icon=ICON,
        bundle_identifier="com.tvshownoisemaker.gui",
        version=VERSION,
        info_plist={
            "CFBundleDisplayName": NAME,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSMinimumSystemVersion": "11.0",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
        },
    )
