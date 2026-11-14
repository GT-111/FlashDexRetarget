"""Exercise real setup/train shell routing with inert external-command stubs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class RuntimeSetupScriptsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "project with space"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "requirements").mkdir()
        for name in ("setup.sh", "train_flashdexretarget.sh"):
            shutil.copyfile(REPO / "scripts" / name, self.repo / "scripts" / name)
        for name in ("isaac_cu128.txt", "native_cu128.txt"):
            shutil.copyfile(REPO / "requirements" / name, self.repo / "requirements" / name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "events.jsonl"
        self.conda = self.root / "conda"
        profile = self.conda / "etc/profile.d/conda.sh"
        profile.parent.mkdir(parents=True)
        profile.write_text('''conda() {
    if [[ "$1" == activate ]]; then export STUB_ACTIVE_ENV="$2"; fi
    "$STUB_BIN/conda" "$@"
}
''')
        stub = f"#!{sys.executable}\n" + '''import json
import os
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
body = sys.stdin.read() if name == "python" and args[:1] == ["-"] else ""
record = {"command": name, "args": args, "body": body,
          "active_env": os.environ.get("STUB_ACTIVE_ENV")}
if name == "pip" and "-r" in args:
    record["requirements"] = Path(args[args.index("-r") + 1]).read_text()
with open(os.environ["STUB_LOG"], "a") as stream:
    stream.write(json.dumps(record) + "\\n")
if name == "conda":
    if args == ["info", "--base"]:
        print(os.environ["STUB_CONDA_BASE"])
    elif args == ["env", "list"]:
        print(os.environ.get("STUB_EXISTING_ENVS", ""))
elif name == "python" and "sys.version_info" in body:
    if args[1] != os.environ["STUB_PYTHON_VERSION"]:
        raise SystemExit("wrong Python for selected profile")
elif name == "python" and "load_robot_asset" in body:
    print(Path(args[1]).resolve().with_suffix(".usd"))
'''
        for name in ("conda", "pip", "git", "python"):
            path = self.bin / name
            path.write_text(stub)
            path.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "STUB_BIN": str(self.bin), "STUB_LOG": str(self.log),
            "STUB_CONDA_BASE": str(self.conda),
        }

    def invoke(self, script, *args, python="3.12", **env):
        return subprocess.run(
            ["bash", str(self.repo / "scripts" / script), *map(str, args)],
            env={**self.env, "STUB_PYTHON_VERSION": python, **env},
            capture_output=True, text=True, check=False,
        )

    def events(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def public_source(self):
        source = self.root / "selected DexForge"
        for package in ("dexforge_core", "dexforge_tracking"):
            path = source / "packages" / package / "pyproject.toml"
            path.parent.mkdir(parents=True)
            path.write_text('[project]\nname = "fixture"\n')
        return source

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_profiles_never_route_through_isaac(self):
        for profile in ("mjwarp", "newton"):
            with self.subTest(profile=profile):
                self.log.unlink(missing_ok=True)
                extra = ["--dexforge-root", self.public_source()] if profile == "newton" else []
                result = self.invoke("setup.sh", "--profile", profile, *extra)
                self.assert_success(result)
                events = self.events()
                installs = [event for event in events if event["command"] == "pip"]
                resolved = [event for event in installs if "requirements" in event]
                self.assertEqual(len(resolved), 1)
                self.assertIn("requirements/native_cu128.txt", resolved[0]["args"])
                self.assertEqual(resolved[0]["requirements"], (self.repo / "requirements/native_cu128.txt").read_text())
                self.assertFalse(any("load_robot_asset" in event["body"] for event in events))
                self.assertFalse(any("convert_assets.py" in " ".join(event["args"]) for event in events))
                self.assertFalse(any("isaacsim==" in event.get("requirements", "") for event in installs))
                local = [event for event in installs if "-e" in event["args"]]
                self.assertTrue(all("--no-deps" in event["args"] for event in local))
                public = [event for event in local if "dexforge_core" in " ".join(event["args"])]
                self.assertEqual(len(public), int(profile == "newton"))
                if public:
                    self.assertIn(str(self.root / "selected DexForge/packages/dexforge_tracking"), public[0]["args"])
                self.assertTrue((self.repo / "env/native.sh").exists())
                self.assertFalse((self.repo / "scripts/env.sh").exists())

    def test_isaac_converts_selected_stock_or_native_config(self):
        configs = [self.root / "renamed hand.yaml", self.root / "another hand.yaml"]
        for config in configs:
            config.write_text("sim: {}\n")
        result = self.invoke("setup.sh", "--profile", "isaac", "--robot-config", configs[0],
                             "--robot-config", configs[1], python="3.11")
        self.assert_success(result)
        events = self.events()
        conversions = [event["args"] for event in events if event["command"] == "python"
                       and event["args"][:1] == ["src/simulator/isaacsim/convert_assets.py"]]
        self.assertEqual(conversions, [["src/simulator/isaacsim/convert_assets.py", "--robot-config", str(config)]
                                       for config in configs])
        freeze = [event["requirements"] for event in events if "requirements" in event]
        self.assertEqual(len(freeze), 1)
        self.assertIn("isaacsim==5.1.0.0", freeze[0])
        self.assertNotIn("tensordict==0.9.0", freeze[0])
        self.log.unlink()
        result = self.invoke("setup.sh", "--profile", "isaac", "--native-scene-config", configs[0], python="3.11")
        self.assert_success(result)
        conversions = [event["args"] for event in self.events() if event["command"] == "python"
                       and event["args"][:1] == ["src/simulator/isaacsim/convert_assets.py"]]
        self.assertEqual(conversions, [["src/simulator/isaacsim/convert_assets.py", "--native-scene-config", str(configs[0])]])

    def test_train_uses_profile_activation_and_passes_real_hydra_selection(self):
        result = self.invoke("train_flashdexretarget.sh", ROBOT="sharpa")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unset ROBOT", result.stderr)
        self.assertIn("envs/robot@robot=sharpa_bimanual", result.stderr)
        self.assertEqual(self.events(), [])
        motion = self.root / "motion directory/motion.pt"
        motion.parent.mkdir()
        motion.touch()
        for profile in ("isaac", "mjwarp", "newton"):
            with self.subTest(profile=profile):
                extra = ["--dexforge-root", self.public_source()] if profile == "newton" else []
                result = self.invoke("setup.sh", "--profile", profile, "--skip_assets", "--env", f"selected-{profile}",
                                     *extra, python="3.11" if profile == "isaac" else "3.12")
                self.assert_success(result)
                self.log.unlink()
                result = self.invoke("train_flashdexretarget.sh", "--profile", profile,
                                     "envs/robot@robot=renamed_asset", "sim.kit_args=operator-owned arguments",
                                     MOTION=str(motion))
                self.assert_success(result)
                call = [event for event in self.events() if event["command"] == "python"]
                self.assertEqual(len(call), 1)
                self.assertEqual(call[0]["active_env"], f"selected-{profile}")
                args = call[0]["args"]
                self.assertEqual(args[:3], ["train.py", "--config_name", "flashdexretarget" if profile == "isaac" else "native"])
                self.assertIn("envs/robot@robot=renamed_asset", args)
                self.assertIn("sim.kit_args=operator-owned arguments", args)
                self.assertIn(f"commands.motion_file={motion}", args)
                if profile != "isaac":
                    self.assertIn(f"envs/simulator@simulator={profile}", args)
                self.assertFalse(any("xhand_bimanual" in arg for arg in args))
                if profile == "isaac":
                    self.log.unlink()
                    result = self.invoke("train_flashdexretarget.sh", "--profile", "isaac",
                                         "--config_name", "native", MOTION=str(motion))
                    self.assert_success(result)
                    call = [event for event in self.events() if event["command"] == "python"]
                    self.assertEqual(call[0]["args"][:3], ["train.py", "--config_name", "native"])

    def test_newton_requires_selected_public_packages_before_mutation(self):
        result = self.invoke("setup.sh", "--profile", "newton")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--dexforge-root", result.stderr)
        self.assertEqual(self.events(), [])

    def test_existing_wrong_python_is_rejected_before_package_install(self):
        result = self.invoke("setup.sh", "--profile", "mjwarp", python="3.11",
                             STUB_EXISTING_ENVS="FlashDexRetargetNative")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wrong Python", result.stderr)
        self.assertFalse(any(event["command"] == "pip" for event in self.events()))


if __name__ == "__main__":
    unittest.main()
