import tempfile
import unittest
from pathlib import Path

from heat_equity.cli import main, validate_inputs


class ValidateInputsTests(unittest.TestCase):
    def test_reports_missing_required_and_optional_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / "pyproject.toml").write_text("", encoding="utf-8")
            config_dir = project_root / "configs"
            config_dir.mkdir()
            config_path = config_dir / "test.toml"
            config_path.write_text(
                '[[inputs]]\nname="Required raster"\npath="data/required.tif"\n'
                '[[inputs]]\nname="Optional layer"\npath="data/optional.tif"\nrequired=false\n',
                encoding="utf-8",
            )

            required, optional = validate_inputs(config_path)

        self.assertEqual(len(required), 1)
        self.assertIn("Required raster", required[0])
        self.assertEqual(len(optional), 1)
        self.assertIn("Optional layer", optional[0])

    def test_present_required_input_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / "pyproject.toml").write_text("", encoding="utf-8")
            input_path = project_root / "data" / "input.csv"
            input_path.parent.mkdir()
            input_path.write_text("value\n1\n", encoding="utf-8")
            config_dir = project_root / "configs"
            config_dir.mkdir()
            config_path = config_dir / "test.toml"
            config_path.write_text(
                '[[inputs]]\nname="Required table"\npath="data/input.csv"\n',
                encoding="utf-8",
            )

            required, optional = validate_inputs(config_path)

        self.assertEqual(required, [])
        self.assertEqual(optional, [])

    def test_cli_returns_nonzero_for_missing_required_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / "pyproject.toml").write_text("", encoding="utf-8")
            config_path = project_root / "config.toml"
            config_path.write_text(
                '[[inputs]]\nname="Required table"\npath="missing.csv"\n',
                encoding="utf-8",
            )

            exit_code = main(["validate-inputs", "--config", str(config_path)])

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()