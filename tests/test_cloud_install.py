import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def test_reuse_checks_installed_pins_too_and_never_replaces_changed_files(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / "cloud/verify_install.py"
    spec = importlib.util.spec_from_file_location("cloud_verify", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    overlay, installed = tmp_path / "overlay", tmp_path / "installed"
    for root in (overlay, installed):
        (root / "integration").mkdir(parents=True)
        (root / "integration/pins.json").write_text(json.dumps(dict(upstream_commit="abc")))
    content = (overlay / "integration/pins.json").read_bytes()
    (overlay / "integration/checksums.json").write_text(
        json.dumps({"integration/pins.json": hashlib.sha256(content).hexdigest()})
    )
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda command, **kwargs: "abc\n" if command[1] == "rev-parse" else ""
    )
    assert module.verify_install(overlay, installed) == 1
    target = installed / "integration/pins.json"
    target.write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        module.verify_install(overlay, installed)
    assert target.read_text() == "changed"


def test_modified_upstream_code_cannot_be_reused_even_with_matching_head_and_overlay(tmp_path):
    script = Path(__file__).parents[1] / "cloud/verify_install.py"
    spec = importlib.util.spec_from_file_location("cloud_verify_upstream", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    overlay, installed = tmp_path / "overlay", tmp_path / "installed"
    installed.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=installed, check=True)
    launcher = installed / "source/isaaclab/app/launcher.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("pristine\n")
    subprocess.run(["git", "add", "."], cwd=installed, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-qm", "base"],
        cwd=installed,
        check=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=installed, text=True).strip()
    for root in (overlay, installed):
        (root / "integration").mkdir(parents=True)
        (root / "integration/pins.json").write_text(json.dumps(dict(upstream_commit=head)))
    content = (overlay / "integration/pins.json").read_bytes()
    (overlay / "integration/checksums.json").write_text(
        json.dumps({"integration/pins.json": hashlib.sha256(content).hexdigest()})
    )
    assert module.verify_install(overlay, installed) == 1
    launcher.write_text("changed upstream\n")
    with pytest.raises(ValueError, match="upstream"):
        module.verify_install(overlay, installed)
    assert launcher.read_text() == "changed upstream\n"


def test_shell_bootstrap_and_eku_data_are_unix_text():
    root = Path(__file__).parents[1]
    assert b"\r" not in (root / "cloud/bootstrap.sh").read_bytes()
    profile = json.loads((root / "source/isaaclab_hiwonder/configs/eka_v6_rgb_spots.json").read_text())
    assert profile["robot_model"] == "eka_v6"
