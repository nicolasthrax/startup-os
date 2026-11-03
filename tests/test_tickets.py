import pytest
from pydantic import ValidationError

from startup_os.memory import MemoryRepo
from startup_os.tickets import Ticket, TicketError, TicketStore

SAMPLE = """---
id: T-0007
title: Build landing page
owner: claude-code
status: todo
depends_on: T-0003, A-2
acceptance_criteria:
- Page loads in < 1s
- Has a signup form
plan: 2026-W45
---

# T-0007: Build landing page

A simple static page.

With two paragraphs.

## Acceptance criteria

- [ ] Page loads in < 1s
- [ ] Has a signup form
"""


@pytest.fixture
def store(tmp_path):
    m = MemoryRepo(tmp_path / "company")
    m.ensure()
    return TicketStore(m)


def test_parse_ticket_front_matter():
    t = Ticket.from_markdown(SAMPLE)
    assert t.id == "T-0007" and t.owner == "claude-code" and t.status == "todo"
    assert t.depends_on == ["T-0003", "A-2"]          # comma string → list
    assert t.acceptance_criteria == ["Page loads in < 1s", "Has a signup form"]
    assert t.description == "A simple static page.\n\nWith two paragraphs."
    assert t.plan == "2026-W45"


def test_roundtrip():
    t = Ticket.from_markdown(SAMPLE)
    assert Ticket.from_markdown(t.to_markdown()) == t


@pytest.mark.parametrize("field,value", [
    ("owner", "marketing"),               # only research / claude-code
    ("status", "someday"),
    ("id", "TICKET-1"),
    ("depends_on", ["T-0003", "whatever"]),
    ("acceptance_criteria", []),
])
def test_invalid_fields_rejected(field, value):
    base = dict(id="T-0001", title="A ticket", owner="research", acceptance_criteria=["x"])
    with pytest.raises(ValidationError):
        Ticket(**(base | {field: value}))


def test_missing_front_matter_rejected():
    with pytest.raises(TicketError):
        Ticket.from_markdown("# just markdown\n")


def test_store_ids_save_and_list(store):
    a = store.new(title="Research pricing", owner="research", acceptance_criteria=["3 price points"])
    assert a.id == "T-0001"
    store.save(a, agent="ceo", message="create T-0001")
    b = store.new(title="Build MVP", owner="claude-code", acceptance_criteria=["tests pass"], depends_on=["T-0001"])
    assert b.id == "T-0002"
    store.save(b, agent="ceo", message="create T-0002")
    assert [t.id for t in store.list()] == ["T-0001", "T-0002"]
    assert [t.id for t in store.list(owner="claude-code")] == ["T-0002"]
    assert store.get("T-0002").depends_on == ["T-0001"]
    assert store.memory.log(1) == ["ceo: create T-0002"]


def test_refresh_blocks_on_tickets_and_approvals(store):
    statuses = {1: "pending"}
    a = store.new(title="Research pricing", owner="research", acceptance_criteria=["x"])
    b = store.new(id="T-0002", title="Buy domain then build", owner="claude-code", acceptance_criteria=["x"],
                  depends_on=["T-0001", "A-1"])
    store.memory.write_many(store.files_for([a, b]), agent="ceo", message="tickets")
    assert store.refresh(statuses.get) == ["T-0002"]
    t = store.get("T-0002")
    assert t.status == "blocked" and t.blocked_reason == "T-0001 is todo; A-1 is pending"
    store.save(store.get("T-0001").model_copy(update={"status": "done"}), agent="research", message="done")
    statuses[1] = "approved"
    store.refresh(statuses.get)
    assert store.get("T-0002").status == "todo" and store.get("T-0002").blocked_reason is None
