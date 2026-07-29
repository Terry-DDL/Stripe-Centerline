"""Build the frozen Stage 3.1 Tk desktop application for macOS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIST_DIR = PROJECT_ROOT / "dist"
APP_NAME = "StripeCenterline"


def pyinstaller_arguments(
    dist_dir: Path,
    work_dir: Path,
    spec_dir: Path,
) -> list[str]:
    """Return the deterministic release packaging arguments."""

    return [
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        APP_NAME,
        "--osx-bundle-identifier",
        "com.stripecenterline.desktop",
        "--paths",
        str(PROJECT_ROOT),
        "--paths",
        str(PROJECT_ROOT / "src"),
        "--add-data",
        f"{PROJECT_ROOT / 'images'}{os.pathsep}images",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        str(PROJECT_ROOT / "tools" / "desktop_app.py"),
    ]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=DEFAULT_DIST_DIR,
    )
    arguments = parser.parse_args(argv)

    try:
        import PyInstaller.__main__ as pyinstaller
    except ImportError as error:
        raise SystemExit(
            "PyInstaller is required: python -m pip install pyinstaller"
        ) from error

    dist_dir = arguments.dist_dir.resolve()
    work_dir = PROJECT_ROOT / "build" / "pyinstaller"
    spec_dir = PROJECT_ROOT / "build" / "spec"
    pyinstaller.run(
        pyinstaller_arguments(dist_dir, work_dir, spec_dir)
    )
    app_path = dist_dir / f"{APP_NAME}.app"
    if not app_path.is_dir():
        raise SystemExit(
            f"expected application bundle was not built: {app_path}"
        )

    archive_path = dist_dir / f"{APP_NAME}-macOS.zip"
    subprocess.run(
        [
            "ditto",
            "-c",
            "-k",
            "--sequesterRsrc",
            "--keepParent",
            str(app_path),
            str(archive_path),
        ],
        check=True,
    )
    print(f"application: {app_path}")
    print(f"archive: {archive_path}")


if __name__ == "__main__":
    main()
