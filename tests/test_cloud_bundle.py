import importlib.util
import json
import tarfile
from pathlib import Path

import pytest


def load():
    path = Path(__file__).parents[1] / "cloud/bundle.py"
    spec = importlib.util.spec_from_file_location("cloud_bundle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_includes_only_explicit_project_files_and_manifest(tmp_path):
    source = tmp_path / "repo"
    source.mkdir()
    (source / "README.md").write_text("test")
    (source / ".env").write_text("SECRET=private")
    target = tmp_path / "cloud.tar.gz"
    result = load().create_bundle(source, target, ["README.md"], revision="testrev")
    with tarfile.open(target) as archive:
        assert archive.getnames() == ["HiWonder-IsaacLab/README.md", "HiWonder-IsaacLab/BUNDLE-MANIFEST.json"]
        manifest = json.load(archive.extractfile("HiWonder-IsaacLab/BUNDLE-MANIFEST.json"))
        assert manifest["revision"] == "testrev" and ".env" not in manifest["files"]
    assert result["file_count"] == 1


@pytest.mark.parametrize("name", (".env", ".ssh/id_rsa", "../outside", "logs/run/final.pt", ".git/config"))
def test_even_explicit_secret_or_escape_paths_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        load().create_bundle(tmp_path, tmp_path / "out.tar.gz", [name], revision="testrev")
