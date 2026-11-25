"""
Tests for util.
"""

import logging
import os

from sisyphus import gs, tk

from i6_core.util import get_executable_path, get_g2p_path, get_g2p_python, get_returnn_python_exe


def _setup_sis_python_in_base_dir(tmp_path, monkeypatch) -> str:
    """Sisyphus command python inside the setup dir, as with a .venv in the setup root"""
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    python.chmod(0o755)
    monkeypatch.setattr(gs, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(gs, "WARNING_ABSPATH", True)
    monkeypatch.setattr(gs, "SIS_COMMAND", [str(python), "sis"])
    for name in ["RETURNN_PYTHON_EXE", "G2P_PATH", "G2P_PYTHON"]:
        monkeypatch.delattr(gs, name, raising=False)
    return str(python)


def test_get_executable_path_explicit_no_abspath_warning(tmp_path, monkeypatch, caplog):
    python = _setup_sis_python_in_base_dir(tmp_path, monkeypatch)
    explicit = tk.Path(python, hash_overwrite="python")
    with caplog.at_level(logging.WARNING):
        assert get_returnn_python_exe(explicit) is explicit
        assert get_g2p_python(explicit) is explicit
        assert get_g2p_path(explicit) is explicit
    assert "Creating absolute path" not in caplog.text


def test_get_executable_path_default_used(tmp_path, monkeypatch, caplog):
    python = _setup_sis_python_in_base_dir(tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING):
        assert get_returnn_python_exe(None).get_path() == python
        assert get_g2p_python(None).get_path() == python
        assert get_g2p_path(None).get_path() == os.path.join(os.path.dirname(python), "g2p.py")
    assert f"Creating absolute path inside current work directory: {python} " in caplog.text


def test_get_executable_path_default_path():
    default = tk.Path("/usr/bin/python3")
    assert get_executable_path(None, "I6_CORE_TEST_UNSET", default) is default
