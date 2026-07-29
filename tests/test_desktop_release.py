"""Release packaging and default-debug contract tests."""

from pathlib import Path
import unittest

from tools import build_desktop_release as release


class DesktopReleaseTests(unittest.TestCase):
    def test_packaging_includes_images_and_desktop_entrypoint(self):
        arguments = release.pyinstaller_arguments(
            Path("/tmp/release-dist"),
            Path("/tmp/release-work"),
            Path("/tmp/release-spec"),
        )

        self.assertIn("--windowed", arguments)
        self.assertIn("--add-data", arguments)
        self.assertIn(
            str(release.PROJECT_ROOT / "tools" / "desktop_app.py"),
            arguments,
        )
        self.assertTrue(
            any(
                str(release.PROJECT_ROOT / "images") in argument
                and "images" in argument
                for argument in arguments
            )
        )


if __name__ == "__main__":
    unittest.main()
