"""Golden serialized literals: the persisted evaluation wire format.

These keys and values are the storage contract for ``EvaluationPlan`` and
``Score``. Changing any of them changes what a persisted plan or score
means, so the literals are pinned here rather than derived from the models'
field names — only a pinned test catches silent drift of stored identity.
"""

from __future__ import annotations

import json

from _builders import (
    policy,
    procedure,
    question_coordinate,
    repeat_plan,
    repeat_plan_coordinate,
    task_set,
    task_set_coordinate,
)
from dr_code.evaluation import (
    EvaluationCoordinate,
    EvaluationPlan,
    FactCoordinate,
    Score,
)
from dr_code.metrics import MetricFactUnit

_GOLDEN_EVALUATION_PLAN = {
    "plan_id": "plan",
    "version": "1",
    "task_set": {
        "coordinate": {
            "task_set_id": "task-set",
            "version": "1",
            "dataset": {"dataset_id": "dataset", "version": "1"},
        },
        "population": ["t0", "t1", "t2"],
        "selected": ["t0", "t2"],
    },
    "repeat_plan": {
        "coordinate": {"repeat_plan_id": "repeat-plan", "version": "1"},
        "task_count": 2,
        "repeats": 2,
        "seeds": [11, 12, 13, 14],
    },
    "procedure": {
        "preprocessing": {
            "definition_id": "pre",
            "version": "0",
            "steps": [
                {
                    "instance_name": "normalize",
                    "step": "normalize_unicode",
                    "settings": {},
                }
            ],
        },
        "metrics": {
            "definition_id": "met",
            "version": "0",
            "questions": [
                {"metric": "text_stats", "on": "output", "settings": {}}
            ],
        },
    },
    "aggregation": {
        "question": {
            "metric": "text_stats",
            "on_key": "output",
            "settings": [],
        },
        "fact": "char_count",
        "statistic": "mean",
        "not_applicable": "exclude",
        "operator_failure": "fail",
    },
}

_GOLDEN_SCORE = {
    "name": "mean_char_count",
    "value": 12.5,
    "unit": "count",
    "evaluation": {
        "plan_id": "plan",
        "version": "1",
        "task_set": {
            "task_set_id": "task-set",
            "version": "1",
            "dataset": {"dataset_id": "dataset", "version": "1"},
        },
        "repeat_plan": {"repeat_plan_id": "repeat-plan", "version": "1"},
    },
    "sources": [
        {
            "question": {
                "metric": "text_stats",
                "on_key": "output",
                "settings": [],
            },
            "fact": "char_count",
        }
    ],
}


def _golden_plan() -> EvaluationPlan:
    return EvaluationPlan(
        plan_id="plan",
        version="1",
        task_set=task_set(),
        repeat_plan=repeat_plan(seeds=(11, 12, 13, 14)),
        procedure=procedure(),
        aggregation=policy(),
    )


def _golden_score() -> Score:
    return Score(
        name="mean_char_count",
        value=12.5,
        unit=MetricFactUnit.COUNT,
        evaluation=EvaluationCoordinate(
            plan_id="plan",
            version="1",
            task_set=task_set_coordinate(),
            repeat_plan=repeat_plan_coordinate(),
        ),
        sources=(
            FactCoordinate(question=question_coordinate(), fact="char_count"),
        ),
    )


def test_evaluation_plan_serializes_to_the_golden_literals() -> None:
    assert (
        json.loads(_golden_plan().model_dump_json()) == _GOLDEN_EVALUATION_PLAN
    )


def test_golden_plan_literals_load_back_to_an_equal_plan() -> None:
    restored = EvaluationPlan.model_validate(_GOLDEN_EVALUATION_PLAN)
    assert restored == _golden_plan()


def test_score_serializes_to_the_golden_literals() -> None:
    assert json.loads(_golden_score().model_dump_json()) == _GOLDEN_SCORE


def test_golden_score_literals_load_back_to_an_equal_score() -> None:
    assert Score.model_validate(_GOLDEN_SCORE) == _golden_score()
