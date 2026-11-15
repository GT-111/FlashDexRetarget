"""Requested runtime paths exist before an optional SDK can use tempfile."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simulator import runtime


def _use_requested_temporary_directory():
    # A real tempfile operation catches the NVRTC prerequisite, without loading
    # an engine or replacing the production directory preparation helper.
    with tempfile.TemporaryFile(dir=os.environ["TMPDIR"]) as stream:
        stream.write(b"runtime path is usable")
        stream.seek(0)
        return stream.read()


def _fake_runtime_launcher(cfg):
    return _use_requested_temporary_directory()


class RuntimeDirectoriesTest(unittest.TestCase):
    def test_requested_runtime_directories_exist_before_native_sdk_and_converter_entry(self):
        from simulator.isaacsim import convert_assets

        with tempfile.TemporaryDirectory() as scratch:
            for entry in ("native", "optional-launcher", "converter"):
                with self.subTest(entry=entry):
                    root = Path(scratch) / entry
                    requested = {
                        "TMPDIR": str(root / "new temporary parent" / "tmp"),
                        "XDG_CACHE_HOME": str(root / "xdg"),
                        "CUDA_CACHE_PATH": str(root / "cuda"),
                        "TORCH_EXTENSIONS_DIR": str(root / "torch-extensions"),
                        "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"),
                        "TRITON_CACHE_DIR": str(root / "triton"),
                        "WARP_CACHE_PATH": "",
                    }
                    cache = Path(requested["XDG_CACHE_HOME"])
                    cache.mkdir(parents=True)
                    sentinel = cache / "existing-entry"
                    sentinel.write_text("retained")
                    with patch.dict(os.environ, requested):
                        if entry == "converter":
                            def before_sdk_import():
                                self.assertEqual(_use_requested_temporary_directory(), b"runtime path is usable")
                                raise RuntimeError("converter parse/SDK boundary reached")

                            # parse_args imports AppLauncher. Stop at that real
                            # boundary after exercising main's actual helper.
                            with patch.object(convert_assets, "parse_args", side_effect=before_sdk_import):
                                with self.assertRaisesRegex(RuntimeError, "converter parse/SDK boundary reached"):
                                    convert_assets.main()
                        else:
                            launcher = None if entry == "native" else f"{__name__}._fake_runtime_launcher"
                            result = runtime.launch_runtime(SimpleNamespace(sim=SimpleNamespace(runtime_launcher=launcher)))
                            if entry == "native":
                                self.assertIsNone(result)
                                self.assertEqual(_use_requested_temporary_directory(), b"runtime path is usable")
                            else:
                                self.assertEqual(result, b"runtime path is usable")
                        self.assertEqual({name: os.environ[name] for name in requested}, requested)
                    for path in requested.values():
                        if path:
                            self.assertTrue(Path(path).is_dir())
                    self.assertEqual(sentinel.read_text(), "retained")
                    self.assertFalse((root / "warp").exists())


if __name__ == "__main__":
    unittest.main()
