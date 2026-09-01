import ast
import os
from pathlib import Path

import pytest
from marimo._ast.load import load_app

import inference_lab
from scripts.sync_my_lab import SyncConflict, SyncError, sync


def _function_ast(code):
    tree = ast.parse(code)
    return [
        ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _exercise_index(code):
    tree = ast.parse(code)
    values = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "exercise_ready"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    return values[0] if values else None


def test_public_master_contains_only_clean_exercise_skeletons():
    _, definitions = inference_lab.app.run()
    app = load_app(Path(inference_lab.__file__))
    exercises = {
        _exercise_index(code): code
        for code in app._cell_manager.codes()
        if _exercise_index(code) is not None
    }

    assert sorted(exercises) == list(range(24))
    for index, spec in enumerate(definitions["LABS"]):
        assert _function_ast(exercises[index]) == _function_ast(spec["code"])


def test_sync_initializes_an_ignored_learner_copy(tmp_path):
    master = tmp_path / "inference_lab.py"
    master.write_text("clean course\n")

    message = sync(tmp_path, validate=False)

    assert "Created" in message
    assert (tmp_path / "my_lab.py").read_text() == "clean course\n"
    assert (tmp_path / ".my_lab.base").read_text() == "clean course\n"


def test_sync_merges_course_and_learner_edits_then_advances_base(tmp_path):
    base_text = """header
exercise = 'todo'
keep 1
keep 2
keep 3
footer = 'old'
"""
    learner_text = base_text.replace("exercise = 'todo'", "exercise = 'answer'")
    master_text = base_text.replace("footer = 'old'", "footer = 'new'")
    (tmp_path / ".my_lab.base").write_text(base_text)
    (tmp_path / "my_lab.py").write_text(learner_text)
    (tmp_path / "inference_lab.py").write_text(master_text)

    message = sync(tmp_path, validate=False)

    merged = (tmp_path / "my_lab.py").read_text()
    assert "preserved" in message
    assert "exercise = 'answer'" in merged
    assert "footer = 'new'" in merged
    assert (tmp_path / ".my_lab.base").read_text() == master_text
    assert (tmp_path / "my_lab.py.backup").read_text() == learner_text


def test_sync_conflict_preserves_learner_and_base(tmp_path):
    base_text = "value = 'base'\n"
    learner_text = "value = 'learner'\n"
    master_text = "value = 'course'\n"
    (tmp_path / ".my_lab.base").write_text(base_text)
    (tmp_path / "my_lab.py").write_text(learner_text)
    (tmp_path / "inference_lab.py").write_text(master_text)

    with pytest.raises(SyncConflict):
        sync(tmp_path, validate=False)

    assert (tmp_path / "my_lab.py").read_text() == learner_text
    assert (tmp_path / ".my_lab.base").read_text() == base_text
    conflicts = list(tmp_path.glob("my_lab.py.merge-conflict-*.py"))
    assert len(conflicts) == 1
    assert "<<<<<<<" in conflicts[0].read_text()


def test_sync_refuses_to_run_while_learner_launcher_is_active(tmp_path):
    (tmp_path / "inference_lab.py").write_text("clean\n")
    (tmp_path / "my_lab.py").write_text("learner\n")
    (tmp_path / ".my_lab.base").write_text("old\n")
    (tmp_path / ".my_lab.running").write_text(str(os.getpid()))

    with pytest.raises(SyncError, match="open in Marimo"):
        sync(tmp_path, validate=False)

    assert (tmp_path / "my_lab.py").read_text() == "learner\n"
    assert (tmp_path / ".my_lab.base").read_text() == "old\n"


def test_launcher_opens_only_the_learner_notebook():
    launcher = Path("scripts/run-local.sh").read_text()
    command = launcher[launcher.index("uv run marimo edit") :]

    assert "my_lab.py" in command
    assert "inference_lab.py" not in command
