import subprocess

import pytest

from startup_os.memory import MemoryError, MemoryRepo
from startup_os.search_index import SearchIndex


@pytest.fixture
def repo(tmp_path):
    idx = SearchIndex(tmp_path / "index.db", tmp_path / "company")
    r = MemoryRepo(tmp_path / "company", idx)
    r.ensure()
    return r


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo.root, capture_output=True, text=True, check=True).stdout


def test_first_run_creates_layout_and_initial_commit(repo):
    for p in ["venture.md", "approvals.log", "plans", "decisions", "research", "research/ideas",
              "tickets", "retros"]:
        assert (repo.root / p).exists(), p
    assert repo.log() == ["system: initialise company memory repo"]
    assert git(repo, "status", "--porcelain") == ""


def test_ensure_is_idempotent(repo):
    assert repo.ensure() is False
    assert len(repo.log()) == 1


def test_write_commits_with_clear_message(repo):
    sha = repo.write("research/ideas/x.md", "# X\n\nhello", agent="research", message='add idea brief "X"')
    assert sha == repo.head()
    assert repo.log(1) == ['research: add idea brief "X"']
    assert "File: research/ideas/x.md" in git(repo, "log", "-1", "--pretty=%B")
    assert git(repo, "show", "--name-only", "--pretty=", "HEAD").strip() == "research/ideas/x.md"
    assert git(repo, "status", "--porcelain") == ""


def test_every_write_is_its_own_commit(repo):
    repo.write("plans/a.md", "# A", agent="ceo", message="add plan A")
    repo.write("plans/a.md", "# A v2", agent="ceo", message="revise plan A")
    repo.write("plans/a.md", "# A v2", agent="ceo", message="no-op")  # unchanged → no commit
    assert repo.log(3) == ["ceo: revise plan A", "ceo: add plan A", "system: initialise company memory repo"]


def test_approval_log_append_commits(repo):
    repo.append_approval("approve idea X", agent="ceo")
    assert "approve idea X" in repo.read("approvals.log")
    assert repo.log(1)[0].startswith("ceo: approval log")


def test_write_rejects_path_escape(repo):
    with pytest.raises(MemoryError):
        repo.write("../evil.md", "x", agent="research", message="nope")
    with pytest.raises(MemoryError):
        repo.write(".git/config", "x", agent="research", message="nope")


def test_index_rebuilt_after_write(repo):
    assert repo.index.search("lighthouse") == []
    repo.write("research/ideas/l.md", "# Lighthouse CRM\n\nCRM for lighthouse keepers.",
               agent="research", message="add brief")
    hits = repo.index.search("lighthouse keepers")
    assert hits and hits[0]["path"] == "research/ideas/l.md" and hits[0]["title"] == "Lighthouse CRM"
    assert isinstance(repo.index.search('weird "quotes" AND (syntax'), list)  # sanitised, no crash
