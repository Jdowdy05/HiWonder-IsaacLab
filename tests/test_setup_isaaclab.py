# SPDX-License-Identifier: BSD-3-Clause
"""Source-only installer acceptance tests; no SDK or network required."""

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "tools/setup_isaaclab.py"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.STDOUT).decode().strip()


class SetupTests(unittest.TestCase):
    def setUp(self):
        # Removing the implementation must fail the suite with a clear assertion.
        assert SCRIPT.is_file(), "Installer has not been implemented"
        spec = importlib.util.spec_from_file_location("setup_isaaclab", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        git(self.upstream, "init", "--quiet")
        git(self.upstream, "config", "user.name", "Installer Test")
        git(self.upstream, "config", "user.email", "installer-test@example.invalid")
        (self.upstream / "pyproject.toml").write_text("# upstream\n")
        (self.upstream / "uv.lock").write_text("# upstream lock\n")
        git(self.upstream, "add", ".")
        git(self.upstream, "commit", "--quiet", "-m", "Fixture")
        git(self.upstream, "tag", "v3.0.0")
        self.repo = self.root / "overlay"
        (self.repo / "integration").mkdir(parents=True)
        payload = {
            "source/isaaclab_hiwonder/robot.usdc": b"robot data\x00",
            "integration/pyproject.toml": b"# integrated\n",
            "integration/uv.lock": b"# pinned lock\n",
        }
        for relative, data in payload.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.checksums = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
        self.pins = {
            "upstream_url": str(self.upstream),
            "upstream_tag": "v3.0.0",
            "upstream_commit": git(self.upstream, "rev-parse", "HEAD"),
            "upstream_pyproject_sha256": hashlib.sha256(b"# upstream\n").hexdigest(),
            "uv_minimum": "0.13.1",
        }
        self.write_metadata()
        self.target = self.root / "installed"

    def write_metadata(self):
        (self.repo / "integration/pins.json").write_text(json.dumps(self.pins))
        (self.repo / "integration/checksums.json").write_text(json.dumps(self.checksums))

    def test_fresh_clone_contains_verified_pin_and_complete_overlay(self):
        self.module.install(self.repo, self.target, skip_sync=True)
        self.assertEqual(git(self.target, "rev-parse", "HEAD"), self.pins["upstream_commit"])
        self.assertEqual((self.target / "source/isaaclab_hiwonder/robot.usdc").read_bytes(), b"robot data\x00")
        self.assertEqual((self.target / "pyproject.toml").read_bytes(), b"# integrated\n")
        self.assertEqual((self.target / "uv.lock").read_bytes(), b"# pinned lock\n")

    def test_windows_long_upstream_paths_survive_staged_clone(self):
        relative = Path("source") / ("deep/" * 8) / ("r" * 140 + ".txt")
        path = self.upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"long-path content")
        git(self.upstream, "-c", "core.longpaths=true", "add", ".")
        git(self.upstream, "commit", "--quiet", "-m", "Long path fixture")
        git(self.upstream, "tag", "-f", "v3.0.0")
        self.pins["upstream_commit"] = git(self.upstream, "rev-parse", "HEAD")
        self.write_metadata()
        self.module.install(self.repo, self.target, skip_sync=True)
        self.assertEqual((self.target / relative).read_bytes(), b"long-path content")

    def test_existing_directory_preserves_user_files(self):
        self.target.mkdir()
        user = self.target / "pyproject.toml"
        user.write_bytes(b"valuable user changes")
        with self.assertRaisesRegex(ValueError, "exist"):
            self.module.install(self.repo, self.target, skip_sync=True)
        self.assertEqual(user.read_bytes(), b"valuable user changes")
        self.assertEqual(list(self.target.iterdir()), [user])

    def test_existing_file_is_untouched(self):
        self.target.write_bytes(b"user file")
        with self.assertRaisesRegex(ValueError, "exist"):
            self.module.install(self.repo, self.target, skip_sync=True)
        self.assertEqual(self.target.read_bytes(), b"user file")

    def test_tampered_asset_fails_before_target_is_created(self):
        (self.repo / "source/isaaclab_hiwonder/robot.usdc").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "[Cc]hecksum"):
            self.module.install(self.repo, self.target, skip_sync=True)
        self.assertFalse(self.target.exists())

    def test_missing_asset_fails_before_target_is_created(self):
        (self.repo / "source/isaaclab_hiwonder/robot.usdc").unlink()
        with self.assertRaisesRegex(ValueError, "[Mm]issing"):
            self.module.install(self.repo, self.target, skip_sync=True)
        self.assertFalse(self.target.exists())

    def test_traversal_payload_is_rejected(self):
        self.checksums["../outside"] = "0" * 64
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "[Uu]nsafe"):
            self.module.install(self.repo, self.target, skip_sync=True)
        self.assertFalse(self.target.exists())

    def test_wrong_revision_does_not_publish_target(self):
        self.pins["upstream_commit"] = "a" * 40
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "revision"):
            self.module.install(self.repo, self.target, skip_sync=True)
        self.assertFalse(self.target.exists())

    def test_unexpected_upstream_manifest_does_not_publish_target(self):
        self.pins["upstream_pyproject_sha256"] = "0" * 64
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.module.install(self.repo, self.target, skip_sync=True)
        self.assertFalse(self.target.exists())

    def test_skip_sync_does_not_require_a_uv_executable(self):
        self.module.install(self.repo, self.target, skip_sync=True, uv="no-such-uv-executable")
        self.assertTrue((self.target / "uv.lock").is_file())

    def test_old_uv_fails_before_clone(self):
        # Only the external executable probe is mocked; filesystem/clone logic remains real.
        real = subprocess.check_output

        def output(command, **kwargs):
            if command == ["uv", "--version"]:
                return b"uv 0.12.9\n"
            return real(command, **kwargs)

        with patch.object(self.module.subprocess, "check_output", side_effect=output):
            with self.assertRaisesRegex(ValueError, "0.13.1"):
                self.module.install(self.repo, self.target)
        self.assertFalse(self.target.exists())

    def test_sync_runs_locked_requested_extras_in_new_checkout(self):
        # A tiny local uv substitute records its received arguments and working directory.
        uv = self.root / "uv.cmd"
        uv.write_text('@echo off\nif "%1"=="--version" (echo uv 0.13.1 & exit /b 0)\necho %* > uv-received.txt\necho %CD% >> uv-received.txt\n')
        if __import__("os").name != "nt":
            uv = self.root / "uv"
            uv.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo "uv 0.13.1"; exit 0; fi\nprintf "%s\\n" "$*" "$PWD" > uv-received.txt\n')
            uv.chmod(0o755)
        self.module.install(self.repo, self.target, uv=str(uv))
        lines = (self.target / "uv-received.txt").read_text().splitlines()
        self.assertEqual(lines[0].strip(), "sync --locked --extra isaacsim --extra rsl-rl --extra test --inexact")
        self.assertEqual(Path(lines[1].strip()).resolve(), self.target.resolve())


if __name__ == "__main__":
    unittest.main()
