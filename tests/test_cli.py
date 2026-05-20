"""CLI invariants — guards against the v0.2.0 regression where
`os.listdir('tests/')` ran on every invocation and crashed any installed
CLI used outside the repo root."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nakdimon", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_version_outside_repo(tmp_path: Path) -> None:
    """`nakdimon --version` must work from a directory without `tests/`."""
    result = _run("--version", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("nakdimon ")


def test_help_outside_repo(tmp_path: Path) -> None:
    result = _run("--help", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_predict_help_outside_repo(tmp_path: Path) -> None:
    result = _run("predict", "--help", cwd=tmp_path)
    assert result.returncode == 0, result.stderr


def test_run_test_help_outside_repo(tmp_path: Path) -> None:
    """The subcommand that previously needed tests/ to even build its parser."""
    result = _run("run_test", "--help", cwd=tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("subcmd", ["train", "results"])
def test_subcommand_help_outside_repo(subcmd: str, tmp_path: Path) -> None:
    result = _run(subcmd, "--help", cwd=tmp_path)
    assert result.returncode == 0, f"{subcmd} --help failed:\n{result.stderr}"
