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
        for name in ("setup.sh", "train_flashdexretarget.sh", "retarget/kinematic_retargeting.sh"):
            destination = self.repo / "scripts" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / "scripts" / name, destination)
        for name in ("isaac_cu128.txt", "native_cu128.txt"):
            shutil.copyfile(REPO / "requirements" / name, self.repo / "requirements" / name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "events.jsonl"
        self.conda = self.root / "conda"
        profile = self.conda / "etc/profile.d/conda.sh"
        profile.parent.mkdir(parents=True)
        profile.write_text('''conda() {
    if [[ "$1" == activate ]]; then export STUB_ACTIVE_ENV="$2" CONDA_DEFAULT_ENV="$2"; fi
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
          "active_env": os.environ.get("STUB_ACTIVE_ENV"),
          "pythonpath": os.environ.get("PYTHONPATH"),
          "eula": os.environ.get("OMNI_KIT_ACCEPT_EULA"),
          "caches": {key: os.environ.get(key) for key in ("XDG_CACHE_HOME", "CUDA_CACHE_PATH",
              "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "WARP_CACHE_PATH")}}
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
    compile(body, "<selected-profile ABI check>", "exec")
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
                                     MOTION=str(motion), python="3.11" if profile == "isaac" else "3.12")
                self.assert_success(result)
                call = [event for event in self.events() if event["command"] == "python"
                        and event["args"][:1] == ["train.py"]]
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
                                         "--config_name", "native", MOTION=str(motion), python="3.11")
                    self.assert_success(result)
                    call = [event for event in self.events() if event["command"] == "python"
                            and event["args"][:1] == ["train.py"]]
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


    def test_retarget_uses_explicit_profile_and_preserves_hydra_arguments(self):
        for profile in ("isaac", "mjwarp", "newton"):
            with self.subTest(profile=profile):
                extra = ["--dexforge-root", self.public_source()] if profile == "newton" else []
                version = "3.11" if profile == "isaac" else "3.12"
                result = self.invoke("setup.sh", "--profile", profile, "--skip_assets", "--env", "selected-" + profile,
                                     *extra, python=version)
                self.assert_success(result)
                self.log.unlink()
                overrides = ["dataset=" + str(self.root / "prepared data"), "input=prepared_native",
                             "steps.convex=false", "steps.usd=false", "steps.sdf=false"]
                result = self.invoke("retarget/kinematic_retargeting.sh", "--profile", profile, *overrides,
                                     python=version, STUB_ACTIVE_ENV="unrelated-active-env")
                self.assert_success(result)
                calls = [event for event in self.events() if event["command"] == "python"
                         and event["args"][:1] == [str(self.repo / "src/retarget/kinematic_retargeting.py")]]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["active_env"], "selected-" + profile)
                self.assertEqual(calls[0]["args"][1:], overrides)
        result = self.invoke("retarget/kinematic_retargeting.sh", python="3.11", STUB_ACTIVE_ENV="selected-newton")
        self.assert_success(result)
        self.assertEqual(self.events()[-1]["active_env"], "selected-isaac")

    def test_retarget_rejects_unknown_or_missing_profile_activation_before_python(self):
        for profile, message in (("other", "unknown profile"), ("mjwarp", "missing activation")):
            with self.subTest(profile=profile):
                result = self.invoke("retarget/kinematic_retargeting.sh", "--profile", profile)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertEqual(self.events(), [])

    def test_generated_activation_rejects_wrong_python_before_public_entry(self):
        for profile, expected, wrong in (("isaac", "3.11", "3.12"), ("mjwarp", "3.12", "3.11")):
            with self.subTest(profile=profile):
                self.assert_success(self.invoke("setup.sh", "--profile", profile, "--skip_assets", python=expected))
                self.log.unlink()
                result = self.invoke("retarget/kinematic_retargeting.sh", "--profile", profile, python=wrong)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("wrong Python", result.stderr)
                self.assertFalse(any("kinematic_retargeting.py" in " ".join(event["args"]) for event in self.events()))
                activation = self.repo / ("scripts/env.sh" if profile == "isaac" else "env/native.sh")
                # Direct documented source use must return failure even without set -e.
                direct = subprocess.run(["bash", "-c", 'source "$1"; exit "$?"', "activation", str(activation)],
                    env={**self.env, "STUB_PYTHON_VERSION": wrong}, capture_output=True, text=True, check=False)
                self.assertNotEqual(direct.returncode, 0)
                self.assertIn("wrong Python", direct.stderr)

    def test_generated_activation_preserves_operator_caches_license_and_clears_foreign_pythonpath(self):
        caches = {name: str(self.root / "operator cache" / name.lower()) for name in (
            "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR", "WARP_CACHE_PATH")}
        result = self.invoke("setup.sh", "--profile", "isaac", "--skip_assets", python="3.11",
                             PYTHONPATH="foreign-sdk-source", **caches)
        self.assert_success(result)
        self.assertTrue(all(event["pythonpath"] is None for event in self.events() if event["command"] in ("pip", "python")))
        self.log.unlink()
        motion = self.root / "motion.pt"
        motion.touch()
        result = self.invoke("train_flashdexretarget.sh", MOTION=str(motion), python="3.11",
                             PYTHONPATH="foreign-sdk-source", OMNI_KIT_ACCEPT_EULA="YES", **caches)
        self.assert_success(result)
        call = next(event for event in self.events() if event["args"][:1] == ["train.py"])
        self.assertIsNone(call["pythonpath"])
        self.assertEqual(call["eula"], "YES")
        self.assertEqual(call["caches"], caches)

    def test_generated_activation_defaults_cache_under_selected_node_local_environment(self):
        caches = {name: "" for name in ("XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR",
                  "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "WARP_CACHE_PATH")}
        scratch = self.root / "node local"
        scratch.mkdir()
        result = self.invoke("setup.sh", "--profile", "mjwarp", "--skip_assets", "--env", "selected-native")
        self.assert_success(result)
        self.log.unlink()
        result = self.invoke("retarget/kinematic_retargeting.sh", "--profile", "mjwarp", TMPDIR=str(scratch), **caches)
        self.assert_success(result)
        call = next(event for event in self.events() if "kinematic_retargeting.py" in " ".join(event["args"]))
        expected = scratch / ("flashdexretarget-" + str(os.getuid())) / "selected-native"
        self.assertEqual(call["caches"], {
            "XDG_CACHE_HOME": str(expected), "CUDA_CACHE_PATH": str(expected / "cuda"),
            "TORCH_EXTENSIONS_DIR": str(expected / "torch-extensions"), "TORCHINDUCTOR_CACHE_DIR": str(expected / "inductor"),
            "TRITON_CACHE_DIR": str(expected / "triton"), "WARP_CACHE_PATH": str(expected / "warp")})



if __name__ == "__main__":
    unittest.main()
