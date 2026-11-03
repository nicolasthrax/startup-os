import subprocess

import pytest

from startup_os import frontmatter
from startup_os.agents.ceo import stage_policy
from startup_os.memory import MemoryRepo
from startup_os.venture import Decision, Stage, TransitionError, Venture


@pytest.fixture
def venture(tmp_path, clock):
    m = MemoryRepo(tmp_path / "company")
    m.ensure()
    return Venture(m, now=clock.now)


def D(**kw):
    return Decision(**({"what": "Move on", "why": "Evidence", "alternatives": ["Wait"]} | kw))


def test_new_repo_starts_in_ideation(venture):
    assert venture.load().stage == Stage.IDEATION


def test_full_forward_path_and_iterate_loop(venture):
    for to in ["Validation", "Build", "Launch", "Iterate", "Build"]:
        venture.transition(to, D(), name="X", idea="research/ideas/x.md")
    s = venture.load()
    assert s.stage == Stage.BUILD and [h["to"] for h in s.history] == ["Validation", "Build", "Launch", "Iterate", "Build"]


@pytest.mark.parametrize("path", [["Build"], ["Launch"], ["Validation", "Launch"], ["Validation", "Iterate"]])
def test_skipping_stages_is_refused(venture, path):
    with pytest.raises(TransitionError):
        for to in path:
            venture.transition(to, D(), name="X")


def test_back_to_ideation_clears_venture(venture):
    venture.transition("Validation", D(), name="X", idea="research/ideas/x.md")
    venture.transition("Ideation", D(what="Drop X", why="no-go score 4/10"))
    s = venture.load()
    assert s.stage == Stage.IDEATION and s.name is None and s.idea is None


def test_stage_change_requires_decision_with_alternatives(venture):
    with pytest.raises(TransitionError):
        venture.transition("Validation", {"what": "x", "why": "y", "alternatives": ["z"]})
    with pytest.raises(ValueError):
        Decision(what="Pick X", why="because", alternatives=[])


def test_decision_record_and_venture_committed_together(venture):
    rel = venture.transition("Validation", D(what="Pick X", why="best brief", alternatives=["Y (5/10)", "Z (4/10)"]),
                             name="X", idea="research/ideas/x.md")
    m = venture.memory
    meta, body = frontmatter.parse(m.read(rel))
    assert meta["type"] == "stage_change" and meta["from"] == "Ideation" and meta["to"] == "Validation"
    for part in ["## What", "Pick X", "## Why", "best brief", "## Alternatives considered", "- Y (5/10)"]:
        assert part in body
    files = subprocess.run(["git", "show", "--name-only", "--pretty=", "HEAD"], cwd=m.root,
                           capture_output=True, text=True).stdout.split()
    assert sorted(files) == sorted([rel, "venture.md"])
    assert m.log(1)[0].startswith("ceo: stage Ideation → Validation")
    vmeta, vbody = frontmatter.parse(m.read("venture.md"))
    assert vmeta["stage"] == "Validation" and vmeta["name"] == "X" and rel in vbody


@pytest.mark.parametrize("to,current,score,ok", [
    (Stage.BUILD, Stage.VALIDATION, 7, True),
    (Stage.BUILD, Stage.VALIDATION, 6, False),     # below threshold
    (Stage.BUILD, Stage.VALIDATION, None, False),  # no report
    (Stage.IDEATION, Stage.VALIDATION, 4, True),   # no-go → back to ideation
    (Stage.IDEATION, Stage.VALIDATION, 8, False),  # GO: dropping it needs a human
    (Stage.LAUNCH, Stage.VALIDATION, 9, False),    # not an allowed transition
])
def test_stage_policy(to, current, score, ok):
    report = None if score is None else {"go_score": score}
    assert (stage_policy(to, current, report, threshold=7) is None) == ok
