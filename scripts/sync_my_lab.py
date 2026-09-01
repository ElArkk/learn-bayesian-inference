"""Safely create or update the local learner notebook."""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


class SyncError(RuntimeError):
    """A safe learner-notebook sync could not be completed."""


class SyncConflict(SyncError):
    """The course and learner changed the same text region."""


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _require_closed_learner(project_dir: Path) -> None:
    lock_path = project_dir / ".my_lab.running"
    if not lock_path.exists():
        return
    try:
        pid = int(lock_path.read_text().strip())
    except (OSError, ValueError):
        raise SyncError(
            "The learner-session lock is invalid. Stop Marimo, then remove "
            f"{lock_path.name}."
        ) from None
    if _process_is_alive(pid):
        raise SyncError(
            "my_lab.py is open in Marimo. Stop that process before you sync. "
            "This prevents the editor from overwriting the merged file."
        )
    lock_path.unlink()


def _validate_notebook(path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "marimo", "check", "--strict", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        raise SyncError(f"Marimo rejected {path.name}:\n{detail}")


def _git_merge_file(
    candidate: Path, base: Path, master: Path
) -> subprocess.CompletedProcess[str]:
    if shutil.which("git") is None:
        raise SyncError("Git is required for course updates but was not found.")
    return subprocess.run(
        ["git", "merge-file", "--diff3", str(candidate), str(base), str(master)],
        check=False,
        capture_output=True,
        text=True,
    )


def sync(project_dir: Path, *, validate: bool = True) -> str:
    """Create or update my_lab.py without changing files after a failed merge."""
    project_dir = project_dir.resolve()
    master = project_dir / "inference_lab.py"
    learner = project_dir / "my_lab.py"
    base = project_dir / ".my_lab.base"
    backup = project_dir / "my_lab.py.backup"

    if not master.is_file():
        raise SyncError(f"Missing clean course notebook: {master}")
    _require_closed_learner(project_dir)
    if validate:
        _validate_notebook(master)

    if not learner.exists():
        shutil.copy2(master, learner)
        shutil.copy2(master, base)
        return "Created my_lab.py from the clean course."

    if not base.is_file():
        raise SyncError(
            "Missing .my_lab.base. my_lab.py was not changed. Restore the base "
            "from backup or ask for a one-time rebase; do not delete your learner notebook."
        )

    if filecmp.cmp(master, base, shallow=False):
        return "my_lab.py already uses the latest course base."

    with tempfile.TemporaryDirectory(prefix=".sync-my-lab-", dir=project_dir) as temp_name:
        temp_dir = Path(temp_name)
        candidate = temp_dir / "my_lab.py"
        old_learner = temp_dir / "old-my_lab.py"
        old_base = temp_dir / "old-base.py"
        next_base = temp_dir / "next-base.py"
        shutil.copy2(learner, candidate)
        shutil.copy2(learner, old_learner)
        shutil.copy2(base, old_base)
        shutil.copy2(master, next_base)

        result = _git_merge_file(candidate, base, master)
        if result.returncode != 0:
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            conflict = project_dir / f"my_lab.py.merge-conflict-{stamp}.py"
            shutil.copy2(candidate, conflict)
            raise SyncConflict(
                "Course and learner edits overlap. my_lab.py and .my_lab.base "
                f"were not changed. Resolve the separate file {conflict.name}."
            )

        if validate:
            _validate_notebook(candidate)

        shutil.copy2(learner, backup)
        try:
            os.replace(candidate, learner)
            os.replace(next_base, base)
        except OSError as exc:
            shutil.copy2(old_learner, learner)
            shutil.copy2(old_base, base)
            raise SyncError(
                "The final file replacement failed. The learner and base were restored."
            ) from exc

    return "Updated my_lab.py and preserved learner exercise edits."


def status(project_dir: Path) -> str:
    project_dir = project_dir.resolve()
    master = project_dir / "inference_lab.py"
    learner = project_dir / "my_lab.py"
    base = project_dir / ".my_lab.base"
    if not learner.exists():
        return "Learner notebook is not initialized."
    if not base.exists():
        return "Learner notebook exists, but its merge base is missing."
    if master.exists() and filecmp.cmp(master, base, shallow=False):
        return "Learner notebook uses the latest course base."
    return "A course update is available. Stop Marimo, then run ./scripts/sync-my-lab.sh."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Report update status only.")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent.parent
    try:
        message = status(project_dir) if args.status else sync(project_dir)
    except SyncError as exc:
        print(f"Sync stopped safely: {exc}", file=sys.stderr)
        return 1
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
