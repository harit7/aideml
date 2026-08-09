"""Tests for task-level metric direction handling.

Follow-up to #91 / #92: the journal direction should come from the task
(config override or task-description inference) rather than be latched from
the first node's per-node verdict, and the originally-reported per-node
directions should be preserved in journal.json.
"""

import json
import logging

from omegaconf import OmegaConf

import aide.agent as agent_module
from aide.agent import determine_metric_direction
from aide.journal import Journal, Node
from aide.utils import serialize
from aide.utils.metric import MetricValue


def _node(value, maximize):
    return Node(
        code="",
        metric=MetricValue(value, maximize=maximize),
        is_buggy=False,
    )


def _acfg(metric_maximize=None):
    return OmegaConf.create(
        {
            "metric_maximize": metric_maximize,
            "feedback": {"model": "test-model", "temp": 0.5},
        }
    )


def test_task_level_direction_wins_over_conflicting_first_node_verdict():
    # task says lower is better; the first node's verdict says the opposite
    journal = Journal(metric_maximize=False)
    journal.append(_node(0.9, maximize=True))
    journal.append(_node(0.2, maximize=False))

    assert journal.metric_maximize is False
    assert journal.nodes[0].metric.maximize is False
    assert journal.get_best_node().metric.value == 0.2


def test_none_direction_is_canonicalized_without_warning(caplog):
    journal = Journal(metric_maximize=True)
    with caplog.at_level(logging.WARNING, logger="aide"):
        journal.append(_node(0.5, maximize=None))

    assert "Metric direction changed" not in caplog.text
    # still canonicalized so comparisons cannot fail on a None direction
    assert journal.nodes[0].metric.maximize is True


def test_reported_direction_persisted_in_journal_json(tmp_path):
    journal = Journal(metric_maximize=True)
    journal.append(_node(0.9, maximize=True))
    journal.append(_node(0.85, maximize=False))  # conflicting per-node verdict

    path = tmp_path / "journal.json"
    serialize.dump_json(journal, path)
    data = json.loads(path.read_text())

    metrics = [n["metric"] for n in data["nodes"]]
    assert [m["maximize"] for m in metrics] == [True, True]
    assert [m["reported_maximize"] for m in metrics] == [True, False]


def test_config_override_skips_llm_call(monkeypatch):
    def fail_query(*args, **kwargs):
        raise AssertionError("query must not be called when the override is set")

    monkeypatch.setattr(agent_module, "query", fail_query)

    assert (
        determine_metric_direction("some task", _acfg(metric_maximize=False)) is False
    )
    assert determine_metric_direction("some task", _acfg(metric_maximize=True)) is True


def test_direction_inferred_from_task_description(monkeypatch):
    captured = {}

    def fake_query(**kwargs):
        captured.update(kwargs)
        return {"lower_is_better": True}

    monkeypatch.setattr(agent_module, "query", fake_query)

    direction = determine_metric_direction(
        "Predict house prices; report RMSE.", _acfg()
    )
    assert direction is False
    assert captured["func_spec"].name == "submit_metric_direction"


def test_inference_failure_falls_back_to_none(monkeypatch):
    def fake_query(**kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(agent_module, "query", fake_query)

    assert determine_metric_direction("some task", _acfg()) is None
