import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import inference_lab

IMPLEMENTATION_NAMES = [
    ("learning_loop",),
    ("normal_log_density",),
    ("log_likelihood", "log_posterior"),
    ("gradient_ascent",),
    ("laplace_sd",),
    ("metropolis",),
    ("autocorrelation",),
    ("covariance",),
    ("numerical_gradient",),
    ("energy",),
    ("leapfrog_step",),
    ("one_hmc_transition",),
    ("is_uturn",),
    ("basic_rhat",),
    ("mc_elbo",),
    ("diagonal_gaussian_sample",),
    ("m_step",),
    ("output_object",),
    ("shrink",),
    ("classify_variables",),
    ("funnel_sample",),
    ("noncenter",),
    ("map_loss",),
    ("choose_method",),
]


def _definitions():
    _, definitions = inference_lab.app.run()
    return definitions


def has(container, phrase):
    """Case- and whitespace-insensitive containment, resilient to prose reflow."""

    def norm(text):
        return " ".join(str(text).lower().split())

    return norm(phrase) in norm(container)


def test_all_reference_solutions_pass():
    definitions = _definitions()
    for index, spec in enumerate(definitions["LABS"]):
        namespace = {"np": np}
        exec(spec["solution"], namespace)
        implementations = tuple(
            namespace[name] for name in IMPLEMENTATION_NAMES[index]
        )
        kind, message = definitions["check_exercise"](index, implementations)
        assert kind == "success", f"Lab {index}: {message}"


def test_tutor_sees_all_saved_exercises():
    definitions = _definitions()
    source = Path(inference_lab.__file__).read_text()
    cells = definitions["exercise_cells_from_text"](source)
    assert sorted(cells) == list(range(len(IMPLEMENTATION_NAMES)))
    for index, names in enumerate(IMPLEMENTATION_NAMES):
        for name in names:
            assert f"def {name}" in cells[index], f"Lab {index}: {name} missing"


def test_tutor_detects_mentioned_lab():
    definitions = _definitions()
    source = Path(inference_lab.__file__).read_text()
    cells = definitions["exercise_cells_from_text"](source)
    mentioned_lab = definitions["mentioned_lab"]

    def chat(*texts):
        return [{"role": "user", "content": text} for text in texts]

    assert mentioned_lab(chat("I am stuck on lab 14, the ELBO is negative"), cells) == 14
    assert mentioned_lab(chat("why does my metropolis reject everything?"), cells) == 5
    assert mentioned_lab(chat("help with Lab 3", "now the next step"), cells) == 3
    assert mentioned_lab(chat("what is a prior?"), cells) is None


def test_tutor_chat_streams_delta_chunks(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    definitions = _definitions()
    tutor = definitions["tutor_model"]
    tutor_globals = tutor.__globals__

    class FakeResponse:
        is_error = False

        def iter_lines(self):
            yield ": OPENROUTER PROCESSING"
            yield 'data: {"choices":[{"delta":{"content":"Momentum "}}]}'
            yield 'data: {"choices":[{"delta":{"content":"refresh."}}]}'
            yield "data: [DONE]"

    class FakeStream:
        def __enter__(self):
            return FakeResponse()

        def __exit__(self, *args):
            return False

    class FakeHttpx:
        @staticmethod
        def stream(*args, **kwargs):
            return FakeStream()

    original_httpx = tutor_globals["httpx"]
    tutor_globals["httpx"] = FakeHttpx
    try:
        chunks = list(tutor([{"role": "user", "content": "explain lab 11"}], None))
    finally:
        tutor_globals["httpx"] = original_httpx

    assert "".join(chunks) == "Momentum refresh."


def test_lab08_rejects_scalar_by_scalar_finite_differences():
    definitions = _definitions()

    def wrong_gradient(fn, point, eps=1e-5):
        point = np.asarray(point, dtype=float)
        grad = [
            (fn(value + eps) - fn(value - eps)) / (2 * eps)
            for value in point
        ]
        return np.asarray(grad)

    kind, message = definitions["check_exercise"](8, (wrong_gradient,))

    assert kind == "danger"
    assert has(message, "copy the full point")


def test_lab09_explains_motion_from_zero_initial_momentum():
    definitions = _definitions()
    guide = definitions["LAB_GUIDES"][9]
    chapter = definitions["COURSE_PROSE"][9]

    prediction = guide["prediction"].lower()
    written_answer = guide["prediction_answer"].lower()
    assert "momentum x" in prediction and "momentum y" in prediction
    assert "set" in prediction and "zero" in prediction
    assert has(written_answer, "gradient changes zero momentum")
    assert has(written_answer, "force-free gray reference stays fixed")
    assert any(term == "Component" for term, _ in chapter["terms"])
    assert "r₁ is horizontal" in " ".join(
        definition for _, definition in chapter["terms"]
    )

    result = definitions["run_experiment_early"](
        9,
        {"momentum x": 0.0, "momentum y": 0.0, "steps": 35},
        2638,
    )
    summary = result.summary.lower()
    assert has(summary, "initial momentum was zero")
    assert has(summary, "posterior gradient changed it")
    assert has(summary, "force-free gray reference stayed at the start")
    assert has(summary, "passed the mode's center line")
    assert "zero r₀ + nonzero gradient" in definitions[
        "make_orientation_view"
    ](9).text
    plt.close("all")


def test_lab10_explains_symmetric_steps_and_metropolis_correction():
    definitions = _definitions()
    guide = definitions["LAB_GUIDES"][10]
    chapter = definitions["COURSE_PROSE"][10]

    prediction = guide["prediction"].lower()
    answer = guide["prediction_answer"].lower()
    assert has(prediction, "half momentum")
    assert has(prediction, "return closer")
    assert has(prediction, "average hmc acceptance")
    assert has(answer, "its own inverse")
    assert has(answer, "negative δh and acceptance 1")
    assert any(term == "Metropolis correction" for term, _ in chapter["terms"])
    assert "accept only when a uniform draw u is smaller than α" in chapter[
        "math_story"
    ]

    result = definitions["run_experiment_early"](
        10, {"step size": 0.22, "steps": 77}, 2739
    )
    summary = result.summary.lower()
    assert has(summary, "backward return error")
    assert "if u<α it accepts" in summary
    assert has(summary, "otherwise the chain repeats its old position")
    assert has(summary, "does not make the non-reversible one-sided comparison")
    assert result.table[0]["method"] == "symmetric leapfrog"
    assert "not valid" in result.table[1]["standard HMC correction"]
    orientation = definitions["make_orientation_view"](10).text
    assert "u &lt; α: accept" in orientation
    assert "Deep dive · Why use a symmetric leapfrog step?" in orientation
    assert has(orientation, "not about choosing a positive or negative momentum")
    assert has(orientation, "same routine can move forward or backward")
    assert has(orientation, "first-order splitting")
    assert has(orientation, "second-order splitting")
    assert has(orientation, "factor of 4")
    assert has(orientation, "does not make an arbitrary non-reversible update")
    plt.close("all")


def test_lab10_rejects_a_non_reversible_one_sided_update():
    definitions = _definitions()

    def asymmetric_step(q, p, step, grad_logp):
        p = p + 0.4 * step * grad_logp(q)
        q = q + step * p
        p = p + 0.6 * step * grad_logp(q)
        return q, p

    kind, message = definitions["check_exercise"](10, (asymmetric_step,))

    assert kind == "danger"
    assert has(message, "retrace when run backward")


def test_lab11_turns_trajectories_into_hmc_chain_transitions():
    definitions = _definitions()
    lab = definitions["LABS"][11]
    guide = definitions["LAB_GUIDES"][11]
    chapter = definitions["COURSE_PROSE"][11]

    assert "trajectory to posterior samples" in lab["title"].lower()
    assert "redraw momentum" in guide["story"].lower()
    assert "store one position each time" in guide["story"].lower()
    assert "central question of this chapter" in chapter["overview"].lower()
    prediction = guide["prediction"].lower()
    assert has(prediction, "before seeing the experiment")
    assert "procedure a" in prediction and "procedure b" in prediction
    for hidden_result_reference in ("time colors", "inset", "position plot"):
        assert hidden_result_reference not in prediction
    assert "result is hidden until you lock a prediction" in chapter[
        "visual_guide"
    ].lower()
    assert "broad-looking coverage" in guide["prediction_answer"].lower()
    assert "plot hides momentum" in guide["prediction_answer"].lower()
    assert "phase error" in guide["prediction_answer"].lower()
    assert any(term == "Energy shell" for term, _ in chapter["terms"])
    assert any(term == "Hamiltonian at a state" for term, _ in chapter["terms"])
    assert any(term == "Position projection" for term, _ in chapter["terms"])
    assert any(term == "Shadow energy" for term, _ in chapter["terms"])
    exercise = guide["exercise"]
    assert has(exercise, "Translate the Hamiltonian into Python")
    assert "old_h = -float(log_target(old_position))" in exercise
    assert "np.dot(old_momentum, old_momentum)" in exercise
    assert has(exercise, "The minus sign converts a high log density")
    assert has(exercise, "unit mass")
    assert "The `float` calls turn NumPy scalar results" in chapter["math_story"]
    intro_html = definitions["render_lab_intro"](
        11, definitions["lab11_ui"]
    ).text
    assert has(intro_html, "Translate the Hamiltonian into Python")
    assert has(intro_html, "language-python codehilite")
    assert [spec[0] for spec in definitions["CONTROL_SPECS"][11]] == [
        "transitions",
        "leapfrog steps",
    ]

    result = definitions["run_experiment_early"](
        11, {"transitions": 300, "leapfrog steps": 12}, 2840
    )
    summary = result.summary.lower()
    assert has(summary, "first path is deterministic")
    assert has(summary, "fresh momentum")
    assert has(summary, "rejection stores the old position again")
    assert has(summary, "can look broad in the q-only plot")
    assert has(summary, "finite leapfrog steps add phase and hamiltonian error")
    assert has(summary, "does not show random movement across joint energy shells")
    assert has(summary, "not one posterior mode")
    energy_span = re.search(r"exact h changed by only ([0-9.e+-]+)", summary)
    assert energy_span is not None
    assert float(energy_span.group(1)) < 0.1
    figure_text = " ".join(
        text.get_text() for axis in result.figure.axes for text in axis.texts
    )
    assert has(figure_text, "Broad-looking q projection")
    assert len(result.figure.axes) >= 4, "The time-color diagnostic needs a colorbar"
    assert {row["object"] for row in result.table} == {
        "leapfrog step",
        "trajectory",
        "HMC transition",
        "HMC chain",
    }

    orientation = definitions["make_orientation_view"](11).text
    assert has(orientation, "One trajectory is not yet an HMC chain")
    assert has(orientation, "draw p₀")
    assert "reject: repeat qₜ" in orientation
    assert has(orientation, "After you lock a prediction")
    assert not has(orientation, "not posterior exploration")
    notebook_source = Path(inference_lab.__file__).read_text()
    assert "hmc_momentum_step" not in notebook_source
    assert "def one_hmc_transition(position, log_target, integrate, rng):" in notebook_source
    assert "# TODO: sample momentum, integrate, compare H, accept or repeat" in notebook_source
    plt.close("all")


def test_lab11_checks_fresh_momentum_acceptance_and_rejection():
    definitions = _definitions()
    namespace = {"np": np}
    exec(definitions["LABS"][11]["solution"], namespace)

    kind, message = definitions["check_exercise"](
        11, (namespace["one_hmc_transition"],)
    )

    assert kind == "success", message


def test_lab12_has_expandable_nuts_logic_deep_dive():
    definitions = _definitions()
    orientation = definitions["make_orientation_view"](12).text
    orientation_lower = orientation.lower()

    assert has(orientation, "Deep dive · How NUTS builds and stops a trajectory")
    assert has(orientation_lower, "balanced binary tree")
    assert "1, 2, 4, 8" in orientation
    assert has(orientation_lower, "left endpoint")
    assert has(orientation_lower, "right endpoint")
    assert has(orientation_lower, "mass matrix")
    assert has(orientation_lower, "check completed subtrees")
    assert has(orientation_lower, "endpoint bias")
    assert has(orientation_lower, "does not have to be the final endpoint")
    assert has(orientation_lower, "divergent leapfrog path")
    assert has(orientation_lower, "maximum depth")
    assert has(orientation_lower, "dual averaging")
    assert has(orientation_lower, "after warmup")


def test_lab12_reference_explanation_answers_the_displayed_question():
    definitions = _definitions()
    question = definitions["LAB_GUIDES"][12]["quiz"]
    answer = definitions["LABS"][12]["answer"].lower()

    assert has(question, "too short")
    assert has(question, "too long")
    assert has(answer, "small move")
    assert has(answer, "turn back")
    assert has(answer, "extra leapfrog steps")


def test_lab13_defines_within_and_between_chain_variance_before_coding():
    definitions = _definitions()
    chapter = definitions["COURSE_PROSE"][13]
    exercise = definitions["LAB_GUIDES"][13]["exercise"]
    math_story = chapter["math_story"]

    assert "chains.var(axis=1, ddof=1)" in math_story
    assert r"W=\frac{1}{m}" in math_story
    assert r"B=\frac{n}{m-1}" in math_story
    assert has(math_story, "sample variance of the chain means")
    assert "multiply by `n`" in math_story
    assert "axis=1" in exercise
    assert "ddof=1" in exercise
    assert "W = 2" in exercise
    assert "B = 4" in exercise
    assert "1.225" in exercise

    namespace = {"np": np}
    exec(definitions["LABS"][13]["solution"], namespace)
    hand_value = namespace["basic_rhat"](
        np.array([[0.0, 2.0], [2.0, 4.0]])
    )
    assert np.isclose(hand_value, np.sqrt(1.5))


def test_lab13_reference_explanation_answers_its_convergence_question():
    definitions = _definitions()
    question = definitions["LAB_GUIDES"][13]["quiz"].lower()
    answer = definitions["LABS"][13]["answer"].lower()

    assert has(question, "what does it converge to")
    assert has(question, "what behavior should several chains share")
    assert has(answer, "stationary sampling regime")
    assert has(answer, "same posterior distribution")
    assert has(answer, "same regions with similar frequencies")
    assert has(answer, "starting points")
    assert has(answer, "not to one parameter value")


def test_all_default_experiments_run():
    definitions = _definitions()
    for index in range(24):
        ui = definitions[f"lab{index:02d}_ui"]
        values = {
            spec[0]: ui.value[f"control::{spec[0]}"]
            for spec in definitions["CONTROL_SPECS"][index]
        }
        runner = (
            definitions["run_experiment_early"]
            if index <= 11
            else definitions["run_experiment_late"]
        )
        result = runner(index, values, 1729 + 101 * index)
        assert result.summary
        plt.close("all")


def test_single_notebook_has_local_feedback_and_real_exercises():
    definitions = _definitions()
    assert "tutor_chat" in definitions
    assert "grade_answer" in definitions
    assert type(definitions["lab01_ui"]).__name__ == "dictionary"
    # The clean course ships Lab 01 as an unsolved stub, so the check must fail.
    assert definitions["check_exercise"](
        1, (definitions["normal_log_density"],)
    )[0] == "danger"


def test_course_opening_explains_first_run_and_tutor_setup():
    source = Path(inference_lab.__file__).read_text()

    assert "Before Lab 0: start the course once" in source
    assert has(source, "Run all stale cells")
    assert has(source, "AI → AI Providers → OpenRouter")
    assert "OPENROUTER_API_KEY=your-key" in source
    assert has(source, "Never paste an API key into a Python or Markdown cell")


def test_course_sidebar_has_an_adjustable_width_control():
    definitions = _definitions()
    source = Path(inference_lab.__file__).read_text()

    width_control = definitions["sidebar_width"]
    assert type(width_control).__name__ == "slider"
    assert width_control.value == 390
    assert "width=f\"{sidebar_width.value}px\"" in source
    assert 'label="Sidebar width"' in source


def test_all_lab_controls_use_reactive_marimo_dictionaries():
    definitions = _definitions()

    for index in range(24):
        ui = definitions[f"lab{index:02d}_ui"]
        assert type(ui).__name__ == "dictionary", f"Lab {index} is not reactive"

        expected_keys = {
            "prediction",
            "run",
            "test",
            "debug",
            "hint_level",
            "answer",
            "done",
            *(
                f"control::{spec[0]}"
                for spec in definitions["CONTROL_SPECS"][index]
            ),
        }
        assert set(ui.value) == expected_keys


def test_run_tests_compiles_the_exact_saved_exercise_code():
    definitions = _definitions()
    notebook_source = Path(inference_lab.__file__).read_text()

    assert "implementations_for_test" in definitions
    assert "<saved learner exercise>" in notebook_source
    assert has(notebook_source, "test compiles the exact saved code")
    assert has(notebook_source, "Do not tell the learner to rerun the cell")
    assert not has(notebook_source, "That button runs your edited cell first")
    assert "Cmd+Enter" in notebook_source
    assert "Ctrl+Enter" in notebook_source


def test_textbook_story_gives_context_before_questions():
    definitions = _definitions()
    labs = definitions["LABS"]
    guides = definitions["LAB_GUIDES"]

    assert len(labs) == len(guides) == 24
    for index, (lab, guide) in enumerate(zip(labs, guides, strict=True)):
        assert len(guide["bridge"].split()) >= 15, f"Lab {index}: short bridge"
        assert len(guide["scenario"].split()) >= 20, (
            f"Lab {index}: short decision context"
        )
        assert len(guide["model"].split()) >= 20, (
            f"Lab {index}: short model explanation"
        )
        assert len(guide["experiment"].split()) >= 18, (
            f"Lab {index}: short experiment guide"
        )
        assert len(guide["goals"]) == 3
        assert not lab["answer"].startswith(("Yes.", "No.")), (
            f"Lab {index}: reference answer depends on missing context"
        )
        learner_text = " ".join(
            [
                lab["title"],
                lab["answer"],
                guide["bridge"],
                guide["scenario"],
                guide["model"],
                guide["experiment"],
                guide["prediction"],
                guide["quiz"],
            ]
        ).lower()
        assert "witness" not in learner_text

    notebook_source = Path(inference_lab.__file__).read_text().lower()
    for stale_frame in ("witness", "alert system", "new incident", "inference laboratory"):
        assert stale_frame not in notebook_source


def test_every_lab_has_long_form_context_terms_and_math_explanations():
    definitions = _definitions()
    chapters = definitions["COURSE_PROSE"]

    assert len(chapters) == 24
    for index, chapter in enumerate(chapters):
        assert len(chapter["overview"].split()) >= 70, (
            f"Lab {index}: chapter overview is too short"
        )
        assert len(chapter["math_story"].split()) >= 50, (
            f"Lab {index}: mathematical walkthrough is too short"
        )
        assert len(chapter["visual_guide"].split()) >= 20, (
            f"Lab {index}: visual-reading guide is too short"
        )
        assert len(chapter["connection"].split()) >= 15, (
            f"Lab {index}: ML connection is too short"
        )
        assert len(chapter["terms"]) >= 4, (
            f"Lab {index}: too few technical terms are defined"
        )
        assert len(chapter["notation"]) >= 3, (
            f"Lab {index}: too few symbols are explained"
        )
        text = " ".join(
            [
                chapter["overview"],
                chapter["math_story"],
                chapter["visual_guide"],
                chapter["connection"],
                *(value for pair in chapter["terms"] for value in pair),
                *(value for pair in chapter["notation"] for value in pair),
            ]
        )
        assert not any(ord(char) < 32 and char not in "\n\r" for char in text), (
            f"Lab {index}: prose contains a broken escape character"
        )

    notebook_source = Path(inference_lab.__file__).read_text()
    renderer = notebook_source[
        notebook_source.index("def render_lab_intro") : notebook_source.index(
            "def render_lab_wrapup"
        )
    ]
    for heading in (
        "Chapter overview",
        "Technical terms in plain language",
        "Read the notation",
        "Walk through the mathematics",
        "What to look for",
        "Connection to machine-learning practice",
    ):
        assert heading in renderer
        assert renderer.index(heading) < renderer.index("## 1 · Predict")


def test_every_lab_shows_a_fixed_visual_before_prediction():
    definitions = _definitions()

    for index in range(24):
        html = definitions["make_orientation_view"](index).text
        assert has(html, "Orientation view · fixed example")
        assert "<svg" in html
        assert has(html, "How to read it.")
        assert has(html, "What the live experiment adds.")

    notebook_source = Path(inference_lab.__file__).read_text()
    renderer = notebook_source[
        notebook_source.index("def render_lab_intro") : notebook_source.index(
            "def render_lab_wrapup"
        )
    ]
    assert renderer.index("make_orientation_view(index)") < renderer.index(
        "## 1 · Predict"
    )
    assert "## Read the experiment" not in renderer


def test_feedback_uses_fixed_low_cost_model_and_robust_json_parsing():
    definitions = _definitions()
    assert definitions["OPENROUTER_MODEL"] == "qwen/qwen3.8-flash"

    raw = """```json
    {
      "summary": "Good start.",
      "what_you_got_right": "You separated density and mass.",
      "what_to_improve": "State that mass is area over an interval.",
      "next_question": "What are the units of density?"
    }
    ```"""
    parsed = definitions["parse_feedback_response"](raw)
    assert parsed["summary"] == "Good start."
    assert parsed["next_question"] == "What are the units of density?"

    with pytest.raises(RuntimeError, match="empty feedback"):
        definitions["parse_feedback_response"](None)

    notebook_source = Path(inference_lab.__file__).read_text()
    assert '"max_tokens": 8192' in notebook_source
    assert '"id": "response-healing"' not in notebook_source
    structured_call = notebook_source[
        notebook_source.index("def _call_openrouter") : notebook_source.index(
            "def _openrouter_headers"
        )
    ]
    assert "timeout=35.0" in structured_call
    assert "openrouter/free" not in notebook_source


def test_all_tutor_modes_require_standalone_display_math():
    source = Path(inference_lab.__file__).read_text()

    assert "_TUTOR_MARKDOWN_RULES" in source
    assert "Never use inline $...$ math" in source
    assert has(source, "heading, list item, table cell, bold label, or link")
    assert "put $$ on a line by itself" in source
    # One definition plus the reasoning, code-failure, and sidebar tutor prompts.
    assert source.count("_TUTOR_MARKDOWN_RULES") == 4


def test_tutor_output_removes_inline_math_and_preserves_display_math():
    definitions = _definitions()
    sanitize = definitions["sanitize_tutor_markdown"]
    raw = (
        "# Why $p(\\theta)$ matters\n"
        "- Compare $\\widehat{R}$ with $W$.\n"
        "| quantity | $\\sigma$ |\n\n"
        "$$\n"
        "p(\\theta \\mid x) \\propto p(x \\mid \\theta)p(\\theta)\n"
        "$$\n"
    )

    cleaned = sanitize(raw)

    assert "# Why p(θ) matters" in cleaned
    assert "- Compare R-hat with W." in cleaned
    assert "| quantity | σ |" in cleaned
    assert "$$\np(\\theta \\mid x) \\propto p(x \\mid \\theta)p(\\theta)\n$$" in cleaned
    assert "$p(" not in cleaned
    assert "$\\widehat" not in cleaned


def test_reasoning_tutor_respects_the_curriculum_boundary():
    definitions = _definitions()
    lab = definitions["LABS"][8]
    guide = definitions["LAB_GUIDES"][8]
    chapters = definitions["COURSE_PROSE"]

    assert "momentum" not in guide["quiz"].lower()
    assert "momentum" not in lab["answer"].lower()
    assert "momentum" not in chapters[8]["overview"].lower()
    assert "Momentum" not in {term for term, _ in chapters[8]["terms"]}
    assert "Momentum" in {term for term, _ in chapters[9]["terms"]}

    notebook_source = Path(inference_lab.__file__).read_text()
    assert has(notebook_source, "Use only concepts listed as introduced through this lab")
    assert has(notebook_source, "Never mark an answer incomplete")
    assert '"curriculum-v2"' in notebook_source


def test_lab_conclusion_is_hidden_before_explanation_submission():
    definitions = _definitions()
    output = definitions["render_lab_wrapup"](
        1,
        (definitions["normal_log_density"],),
        definitions["lab01_ui"],
    )
    html = output.text
    assert not has(html, "What this lab established")
    assert not has(html, "Why the next lab follows")

    notebook_source = Path(inference_lab.__file__).read_text()
    wrapup = notebook_source[
        notebook_source.index("def render_lab_wrapup") : notebook_source.index(
            "tutor_chat ="
        )
    ]
    assert "if submitted_answer:" in wrapup
    assert "*conclusion_parts" in wrapup


def test_prediction_answers_are_written_and_gated_by_the_experiment():
    definitions = _definitions()

    for index, guide in enumerate(definitions["LAB_GUIDES"]):
        assert len(guide["prediction_answer"].split()) >= 25, (
            f"Lab {index}: prediction answer is too short"
        )

    notebook_source = Path(inference_lab.__file__).read_text()
    renderer = notebook_source[
        notebook_source.index("def render_lab_intro") : notebook_source.index(
            "def render_lab_wrapup"
        )
    ]
    result_gate = renderer.index("if not ui[\"run\"].value:")
    written_answer = renderer.index("Written answer to the opening prediction")
    assert written_answer > result_gate
    assert 'guide["prediction_answer"]' in renderer


def test_failed_code_can_request_opt_in_pedagogical_feedback():
    definitions = _definitions()
    source = definitions["implementation_source"](
        (definitions["gradient_ascent"],)
    )
    assert "def gradient_ascent" in source
    assert "TODO" in source
    assert "grade_code_failure" in definitions
    assert "implementation_snapshot" in definitions
    assert "normal_log_density" in definitions["LAB_GUIDES"][2]["exercise"]

    notebook_source = Path(inference_lab.__file__).read_text()
    wrapup = notebook_source[
        notebook_source.index("def render_lab_wrapup") : notebook_source.index(
            "tutor_chat ="
        )
    ]
    assert 'ui["test"].value or ui["debug"].value' in wrapup
    assert has(notebook_source, "Ask tutor about this failure")
    assert has(notebook_source, "Do not provide a complete corrected function")
    assert has(notebook_source, "Do not recommend ")
    assert has(notebook_source, "scipy.stats or another library shortcut")
    # Cached on the exact code and failure so unrelated re-renders do not
    # trigger new paid API calls, while any new failure gets fresh feedback.
    assert "_code_feedback_cache" in notebook_source
    assert "_code_feedback_cache[cache_key]" in notebook_source
    assert "supporting_implementations" in wrapup
    assert "code_snapshot[\"saved_code\"]" in wrapup
    assert "test_bundle[\"snapshot\"]" in wrapup
    assert "implementations_for_test" in wrapup
    assert has(notebook_source, "do not diagnose a stale-kernel problem")
    assert not has(notebook_source, "Tell the learner to run the exercise cell")
    assert "(normal_log_density,)" in notebook_source
    # Prose lives in reformatted literals; check the loaded data, not the source.
    assert has(definitions["LAB_GUIDES"][2]["exercise"], "Do not score the observations a second time")
    assert has(wrapup, "Why the test failed")
    assert has(wrapup, "Your next implementation step")

    calls = []
    grader = definitions["grade_code_failure"]
    grader_globals = grader.__globals__
    call_key = next(key for key in grader_globals if key.endswith("_call_openrouter"))
    original_call = grader_globals[call_key]

    def fake_call(messages, *, structured=False):
        calls.append((messages, structured))
        return json.dumps(
            {
                "summary": "The test found one wrong component.",
                "what_you_got_right": "The likelihood is correct.",
                "what_to_improve": "Score the parameter under the prior.",
                "next_question": "Which value does the prior score?",
            }
        )

    grader_globals[call_key] = fake_call
    try:
        for _ in range(2):
            result = grader(
                2,
                "def log_posterior(...): ...",
                "The prior term is wrong.",
                "def normal_log_density(...): ...",
            )
            assert result["ok"]
        assert len(calls) == 1, (
            "Identical code and failure must be served from the cache, "
            "not with a second paid model call"
        )
        result = grader(
            2,
            "def log_posterior(...): ...",
            "The likelihood term is wrong.",
            "def normal_log_density(...): ...",
        )
        assert result["ok"]
        assert len(calls) == 2, "A new failure must call the model again"
    finally:
        grader_globals[call_key] = original_call

    final_prompt = calls[-1][0][-1]["content"]
    assert "normal_log_density" in final_prompt


def test_lab_two_prediction_can_be_tested_with_new_sales_day_control():
    definitions = _definitions()
    control_names = [spec[0] for spec in definitions["CONTROL_SPECS"][2]]
    assert has(control_names, "new sales day")
    assert "new sales day" in definitions["LAB_GUIDES"][2]["prediction"].lower()

    fixed_values = {
        "prior mean": 55.0,
        "observations": 5,
        "new sales day": 90.0,
    }
    small_prior = definitions["run_experiment_early"](
        2,
        {**fixed_values, "prior width": 2.0},
        1931,
    )
    wide_prior = definitions["run_experiment_early"](
        2,
        {**fixed_values, "prior width": 30.0},
        1931,
    )

    def shift(result):
        match = re.search(r"shift of ([+-]\d+\.\d+)", result.summary)
        assert match is not None
        return float(match.group(1))

    assert shift(wide_prior) > shift(small_prior) > 0
    assert "before one highlighted new day" in definitions[
        "make_orientation_view"
    ](2).text
    plt.close("all")
