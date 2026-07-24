from __future__ import annotations

import inspect
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

import dr_code.viewer.app as viewer_app_module
import dr_code.viewer.assets as viewer_assets
from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.app import ViewerAssetsError, create_app
from dr_code.viewer.database import ViewerDatabase
from dr_code.viewer.domain import (
    ANNOTATION_NOTE_MAX_LENGTH,
    ANNOTATION_TAG_IDS_MAX_COUNT,
    TAG_NAME_MAX_LENGTH,
    Annotation,
    ComparisonStage,
    ExampleDetail,
    ExampleSummary,
    FailureGroup,
    Failures,
    IncompatibleRunsError,
    OutcomeTransition,
    Page,
    ReviewPage,
    RunComparison,
    RunSummary,
    Tag,
    Verdict,
    Waterfall,
    WaterfallStage,
)
from viewer.helpers import write_bundle

_DIGEST = "a" * 64
_OUTPUT_DIGEST = "b" * 64


@pytest.fixture
def static_dir(tmp_path: Path) -> Path:
    frontend = tmp_path / "static"
    assets = frontend / "assets"
    assets.mkdir(parents=True)
    (frontend / "index.html").write_text("<main>viewer</main>")
    (assets / "app.js").write_text("console.log('viewer')")
    return frontend


def _run(run_id: str = "baseline") -> RunSummary:
    return RunSummary(
        run_id=run_id,
        label=run_id.title(),
        dataset_id="evalplus/humanevalplus",
        manifest_sha256="c" * 64,
        corpus_sha256=_DIGEST,
        definition_id="functions-v1",
        definition_version="1",
        has_evaluation=False,
        definition_identity="d" * 64,
    )


class FakeService:
    def __init__(self) -> None:
        self.example_kwargs: dict[str, object] | None = None
        self.review_kwargs: dict[str, object] | None = None
        self.put_kwargs: dict[str, object] | None = None
        self.created_tag_names: list[str] = []

    def list_runs(self):
        return (_run(),)

    def waterfall(self, run_id: str):
        return Waterfall(
            run=_run(run_id),
            stages=(
                WaterfallStage(
                    stage_id="source",
                    label="Corpus rows",
                    unit="sample",
                    order=0,
                    count=2,
                    denominator=2,
                    rate=1.0,
                ),
            ),
        )

    def failures(self, run_id: str):
        return Failures(
            run=_run(run_id),
            groups=(
                FailureGroup(
                    failure_code="no_candidate",
                    failed_step="extract_candidates",
                    cause="no code block",
                    count=1,
                ),
                FailureGroup(
                    failure_code="no_candidate",
                    failed_step="extract_candidates",
                    cause=None,
                    count=1,
                ),
                FailureGroup(
                    failure_code="no_candidate",
                    failed_step="extract_candidates",
                    cause="",
                    count=1,
                ),
            ),
            total_count=3,
        )

    def examples(self, run_id: str, **kwargs: object):
        self.example_kwargs = {"run_id": run_id, **kwargs}
        return Page(
            items=(
                ExampleSummary(
                    sample_id="sample-1",
                    task_id="HumanEval/1",
                    decoder_output_sha256=_OUTPUT_DIGEST,
                    outcome="no_candidate",
                    failure_code="no_candidate",
                    failed_step="extract_candidates",
                    decoder_output="raw output",
                    annotation_verdict=Verdict.SHOULD_BE_PARSEABLE,
                ),
            ),
            total=1,
            limit=int(kwargs["limit"]),
            offset=int(kwargs["offset"]),
        )

    def example(self, run_id: str, sample_id: str):
        del run_id
        return ExampleDetail(
            sample_id=sample_id,
            dataset_id="evalplus/humanevalplus",
            task_identity="e" * 64,
            corpus_sha256=_DIGEST,
            decoder_output_sha256=_OUTPUT_DIGEST,
            context={"task_id": "HumanEval/1", "nested": {"ignored": True}},
            outcome="function_candidates_extracted",
            failure_code="no_candidate",
            failed_step="extract_candidates",
            cause="no code block",
            raw_decoder_output="def f(): pass",
            candidates=(
                {
                    "candidate_id": "candidate-1",
                    "candidate_index": 0,
                    "cleaned_source": "def f(): pass",
                    "compile_warnings": None,
                    "origins": [
                        {
                            "path": [
                                {"kind": "plain", "details": {}},
                                {
                                    "kind": "fence",
                                    "details": {"tag": "python"},
                                },
                            ]
                        }
                    ],
                    "top_level_function_names": ["f"],
                },
            ),
            facts=({"step_name": "extract", "facts_json": "{}"},),
            rejections=(),
            annotation=None,
        )

    def review_examples(self, run_id: str, **kwargs: object):
        self.review_kwargs = {"run_id": run_id, **kwargs}
        return ReviewPage(
            items=(self.example(run_id, "sample-1"),),
            total=1,
            limit=int(kwargs["limit"]),
            offset=int(kwargs["offset"]),
        )

    def compare(self, baseline_run_id: str, candidate_run_id: str):
        return RunComparison(
            baseline=_run(baseline_run_id),
            candidate=_run(candidate_run_id),
            stages=(
                ComparisonStage(
                    stage_id="source",
                    label="Corpus rows",
                    unit="sample",
                    baseline_count=2,
                    baseline_denominator_count=2,
                    candidate_count=2,
                    candidate_denominator_count=2,
                    count_delta=0,
                    baseline_rate=1.0,
                    candidate_rate=1.0,
                    rate_delta=0.0,
                ),
            ),
            transitions=(
                OutcomeTransition(
                    baseline_outcome="failure",
                    candidate_outcome="success",
                    count=1,
                ),
            ),
        )

    def list_tags(self):
        return (Tag(tag_id="tag-1", name="Needs fence repair"),)

    def create_tag(self, name: str):
        self.created_tag_names.append(name)
        return Tag(tag_id="tag-2", name=name)

    def put_annotation(
        self,
        corpus_sha256: str,
        sample_id: str,
        decoder_output_sha256: str,
        **kwargs: object,
    ):
        self.put_kwargs = {
            "corpus_sha256": corpus_sha256,
            "sample_id": sample_id,
            "decoder_output_sha256": decoder_output_sha256,
            **kwargs,
        }
        verdict = kwargs["verdict"]
        return Annotation(
            corpus_sha256=corpus_sha256,
            sample_id=sample_id,
            decoder_output_sha256=decoder_output_sha256,
            verdict=Verdict(verdict) if verdict is not None else None,
            note=kwargs.get("note"),
            tags=(Tag(tag_id="tag-1", name="Needs fence repair"),),
        )

    def delete_annotation(self, *args: str):
        self.deleted = args

    def export_annotations(self):
        return [
            {
                "corpus_sha256": _DIGEST,
                "sample_id": "sample-1",
                "decoder_output_sha256": _OUTPUT_DIGEST,
                "verdict": None,
                "note": "reviewed",
                "tags": ["Needs fence repair"],
            }
        ]


def test_read_endpoints_adapt_domain_models_and_forward_exact_filters(
    static_dir: Path,
) -> None:
    service = FakeService()
    client = TestClient(
        create_app(service, static_dir=static_dir),
        base_url="http://127.0.0.1",
    )

    assert client.get("/api/runs").json()[0]["semantic_coordinates"] == {
        "definition_version": "1",
        "definition_identity": "d" * 64,
    }
    assert (
        client.get("/api/waterfall", params={"run_id": "baseline"}).json()[
            "stages"
        ][0]["denominator_count"]
        == 2
    )
    failures = client.get(
        "/api/failures", params={"run_id": "baseline"}
    ).json()["groups"]
    failure = failures[0]
    assert (failure["failure_code"], failure["failed_step"]) == (
        "no_candidate",
        "extract_candidates",
    )
    assert failure["cause"] == "no code block"
    ids_by_cause = {group["cause"]: group["id"] for group in failures}
    assert len(set(ids_by_cause.values())) == 3
    assert ids_by_cause[None] != ids_by_cause[""]

    response = client.get(
        "/api/examples",
        params={
            "run_id": "baseline",
            "failure_code": "no_candidate",
            "failed_step": "extract_candidates",
            "cause": "no code block",
            "search": "fence",
            "limit": 25,
            "offset": 50,
        },
    )
    assert response.status_code == 200
    assert response.json()["limit"] == 25
    assert response.json()["items"][0]["annotation_verdict"] == (
        "should_be_parseable"
    )
    assert service.example_kwargs == {
        "run_id": "baseline",
        "stage_id": None,
        "failure_code": "no_candidate",
        "failed_step": "extract_candidates",
        "cause": "no code block",
        "cause_is_null": False,
        "compare_run_id": None,
        "baseline_outcome": None,
        "candidate_outcome": None,
        "search": "fence",
        "limit": 25,
        "offset": 50,
    }

    null_cause = client.get(
        "/api/examples",
        params={
            "run_id": "baseline",
            "failure_code": "no_candidate",
            "failed_step": "extract_candidates",
            "cause_is_null": "true",
        },
    )
    assert null_cause.status_code == 200
    assert service.example_kwargs is not None
    assert service.example_kwargs["cause"] is None
    assert service.example_kwargs["cause_is_null"] is True

    empty_cause = client.get(
        "/api/examples",
        params={
            "run_id": "baseline",
            "failure_code": "no_candidate",
            "failed_step": "extract_candidates",
            "cause": "",
        },
    )
    assert empty_cause.status_code == 200
    assert service.example_kwargs is not None
    assert service.example_kwargs["cause"] == ""
    assert service.example_kwargs["cause_is_null"] is False

    long_cause = "x" * 300
    long_cause_response = client.get(
        "/api/examples",
        params={
            "run_id": "baseline",
            "failure_code": "no_candidate",
            "failed_step": "extract_candidates",
            "cause": long_cause,
        },
    )
    assert long_cause_response.status_code == 200
    assert service.example_kwargs is not None
    assert service.example_kwargs["cause"] == long_cause

    detail = client.get(
        "/api/example",
        params={"run_id": "baseline", "sample_id": "HumanEval/32"},
    ).json()
    assert detail["raw_decoder_output"] == "def f(): pass"
    assert detail["sample_id"] == "HumanEval/32"
    assert (
        client.get("/api/runs/baseline/examples/HumanEval/32").status_code
        == 404
    )
    assert detail["candidates"][0]["compile_warnings"] == []
    assert detail["candidates"][0]["origins"] == [
        {
            "path": [
                {"kind": "plain", "details": {}},
                {"kind": "fence", "details": {"tag": "python"}},
            ]
        }
    ]
    assert detail["context"] == {"task_id": "HumanEval/1"}

    review = client.get(
        "/api/review-examples",
        params={
            "run_id": "baseline",
            "failure_code": "no_candidate",
            "failed_step": "extract_candidates",
            "cause": "",
            "search": "fence",
            "limit": 25,
            "offset": 50,
        },
    )
    assert review.status_code == 200
    assert review.json()["total"] == 1
    assert review.json()["items"][0]["failure_code"] == "no_candidate"
    assert review.json()["items"][0]["raw_decoder_output"] == "def f(): pass"
    assert service.review_kwargs == {
        "run_id": "baseline",
        "failure_code": "no_candidate",
        "failed_step": "extract_candidates",
        "cause": "",
        "cause_is_null": False,
        "search": "fence",
        "limit": 25,
        "offset": 50,
    }

    null_cause_review = client.get(
        "/api/review-examples",
        params={
            "run_id": "baseline",
            "failure_code": "no_candidate",
            "failed_step": "extract_candidates",
            "cause_is_null": "true",
        },
    )
    assert null_cause_review.status_code == 200
    assert service.review_kwargs is not None
    assert service.review_kwargs["cause"] is None
    assert service.review_kwargs["cause_is_null"] is True

    missing_cause = client.get(
        "/api/review-examples",
        params={
            "run_id": "baseline",
            "failure_code": "no_candidate",
            "failed_step": "extract_candidates",
        },
    )
    conflicting_cause = client.get(
        "/api/review-examples",
        params={
            "run_id": "baseline",
            "failure_code": "no_candidate",
            "failed_step": "extract_candidates",
            "cause": "no code block",
            "cause_is_null": "true",
        },
    )
    assert missing_cause.status_code == 400
    assert conflicting_cause.status_code == 400


def test_query_parameter_run_ids_support_slashes(static_dir: Path) -> None:
    client = TestClient(
        create_app(FakeService(), static_dir=static_dir),
        base_url="http://127.0.0.1",
    )

    response = client.get("/api/waterfall", params={"run_id": "group/run"})

    assert response.status_code == 200
    assert response.json()["run_id"] == "group/run"

    comparison = client.get(
        "/api/compare", params={"baseline": "baseline", "candidate": "new"}
    ).json()
    assert comparison["stages"][0]["candidate_denominator_count"] == 2
    assert comparison["transitions"][0]["id"] == "failure → success"


def test_compare_endpoint_supports_two_preprocessing_only_runs(
    tmp_path: Path,
    static_dir: Path,
) -> None:
    baseline = write_bundle(
        tmp_path / "baseline",
        run_id="baseline",
        with_evaluation=False,
    )
    candidate = write_bundle(
        tmp_path / "candidate",
        run_id="candidate",
        corpus_path=baseline.corpus_path,
        with_evaluation=False,
    )
    with ViewerDatabase(":memory:") as database:
        service = ViewerAnalytics(database, [baseline, candidate])
        response = TestClient(
            create_app(service, static_dir=static_dir),
            base_url="http://127.0.0.1",
        ).get(
            "/api/compare",
            params={"baseline": "baseline", "candidate": "candidate"},
        )

    assert response.status_code == 200
    assert [stage["id"] for stage in response.json()["stages"]] == [
        "lost:output_present",
        "lost:output_nonblank",
        "has_extracted_candidate",
        "has_compilable_candidate",
        "has_top_level_candidate",
    ]


def test_annotation_and_tag_endpoints_are_typed(static_dir: Path) -> None:
    service = FakeService()
    client = TestClient(
        create_app(service, static_dir=static_dir),
        base_url="http://127.0.0.1",
    )
    annotation_path = f"/api/annotations/{_DIGEST}/{_OUTPUT_DIGEST}"
    annotation_params = {"sample_id": "HumanEval/32"}

    tag_response = client.post("/api/tags", json={"name": "  New tag  "})
    assert tag_response.status_code == 201
    assert tag_response.json() == {"tag_id": "tag-2", "name": "New tag"}

    response = client.put(
        annotation_path,
        params=annotation_params,
        json={
            "verdict": "should_be_parseable",
            "note": "reviewed",
            "tag_ids": ["tag-1"],
        },
    )
    assert response.status_code == 200
    assert response.json()["tags"][0]["tag_id"] == "tag-1"
    assert service.put_kwargs is not None
    assert service.put_kwargs["sample_id"] == "HumanEval/32"
    assert service.put_kwargs["tag_ids"] == ["tag-1"]

    unlabeled = client.put(
        annotation_path,
        params=annotation_params,
        json={
            "verdict": None,
            "note": "save without verdict",
            "tag_ids": ["tag-1"],
        },
    )
    assert unlabeled.status_code == 200
    assert unlabeled.json()["verdict"] is None
    assert unlabeled.json()["note"] == "save without verdict"
    assert service.put_kwargs is not None
    assert service.put_kwargs["verdict"] is None

    assert (
        client.delete(annotation_path, params=annotation_params).status_code
        == 204
    )
    exported = client.get("/api/annotations/export").json()[0]
    assert exported["verdict"] is None
    assert exported["tags"] == ["Needs fence repair"]

    invalid = client.put(
        "/api/annotations/not-a-digest/not-a-digest",
        params=annotation_params,
        json={"verdict": "should_be_parseable"},
    )
    assert invalid.status_code == 422


def test_annotation_http_contract_uses_distinct_tags_and_normalized_names(
    static_dir: Path,
) -> None:
    service = FakeService()
    client = TestClient(
        create_app(service, static_dir=static_dir),
        base_url="http://127.0.0.1",
    )
    annotation_path = f"/api/annotations/{_DIGEST}/{_OUTPUT_DIGEST}"
    params = {"sample_id": "HumanEval/32"}
    exact_tag_ids = [
        f"tag-{index:03}" for index in range(ANNOTATION_TAG_IDS_MAX_COUNT)
    ]

    exact = client.put(
        annotation_path,
        params=params,
        json={
            "verdict": None,
            "note": "n" * ANNOTATION_NOTE_MAX_LENGTH,
            "tag_ids": [*exact_tag_ids, exact_tag_ids[0]],
        },
    )
    assert exact.status_code == 200
    assert service.put_kwargs is not None
    assert service.put_kwargs["tag_ids"] == exact_tag_ids

    too_many_tags = client.put(
        annotation_path,
        params=params,
        json={
            "verdict": None,
            "tag_ids": [
                *exact_tag_ids,
                f"tag-{ANNOTATION_TAG_IDS_MAX_COUNT:03}",
            ],
        },
    )
    assert too_many_tags.status_code == 422
    too_long_note = client.put(
        annotation_path,
        params=params,
        json={
            "verdict": None,
            "note": "n" * (ANNOTATION_NOTE_MAX_LENGTH + 1),
        },
    )
    assert too_long_note.status_code == 422

    exact_name = "x" * TAG_NAME_MAX_LENGTH
    accepted_tag = client.post("/api/tags", json={"name": f"  {exact_name}  "})
    assert accepted_tag.status_code == 201
    assert service.created_tag_names[-1] == exact_name
    normalized_short = client.post(
        "/api/tags", json={"name": "left" + " " * 200 + "right"}
    )
    assert normalized_short.status_code == 201
    assert service.created_tag_names[-1] == "left right"
    assert (
        client.post(
            "/api/tags", json={"name": "x" * (TAG_NAME_MAX_LENGTH + 1)}
        ).status_code
        == 422
    )


def test_annotation_http_rejects_non_scalar_text_before_database_binding(
    tmp_path: Path, static_dir: Path
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    with ViewerDatabase(":memory:") as database:
        service = ViewerAnalytics(database, [descriptor])
        target = service.example(descriptor.run_id, "no-code")
        assert target.decoder_output_sha256 is not None
        client = TestClient(
            create_app(service, static_dir=static_dir),
            base_url="http://127.0.0.1",
            raise_server_exceptions=False,
        )

        ordinary_validation = client.post("/api/tags", json={})
        assert ordinary_validation.status_code == 422
        ordinary_detail = ordinary_validation.json()["detail"][0]
        assert ordinary_detail["loc"] == ["body", "name"]
        assert ordinary_detail["type"] == "missing"
        assert ordinary_detail["input"] == {}

        normalized = client.post(
            "/api/tags", json={"name": "left\u0085\u00a0right"}
        )
        assert normalized.status_code == 201
        assert normalized.json()["name"] == "left right"

        surrogate_tag = client.post(
            "/api/tags",
            content='{"name":"\\ud800"}',
            headers={"content-type": "application/json"},
        )
        assert surrogate_tag.status_code == 422
        assert surrogate_tag.json()["detail"][0]["input"] == "\\ud800"
        surrogate_key = client.post(
            "/api/tags",
            content='{"\\ud800":1}',
            headers={"content-type": "application/json"},
        )
        assert surrogate_key.status_code == 422

        annotation_path = (
            f"/api/annotations/{descriptor.corpus_sha256}/"
            f"{target.decoder_output_sha256}"
        )
        surrogate_note = client.put(
            annotation_path,
            params={"sample_id": target.sample_id},
            content='{"verdict":null,"note":"\\ud800"}',
            headers={"content-type": "application/json"},
        )
        assert surrogate_note.status_code == 422
        assert surrogate_note.json()["detail"][0]["input"] == "\\ud800"
        assert (
            database.get_annotation(
                descriptor.corpus_sha256,
                target.sample_id,
                target.decoder_output_sha256,
            )
            is None
        )
        assert [tag.name for tag in database.list_tags()] == ["left right"]


def test_maximum_database_annotation_round_trips_get_to_unchanged_put(
    tmp_path: Path, static_dir: Path
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    with ViewerDatabase(":memory:") as database:
        service = ViewerAnalytics(database, [descriptor])
        target = service.example(descriptor.run_id, "no-code")
        assert target.decoder_output_sha256 is not None
        tags = [
            database.create_tag(f"tag {index:03}")
            for index in range(ANNOTATION_TAG_IDS_MAX_COUNT)
        ]
        stored = database.put_annotation(
            descriptor.corpus_sha256,
            target.sample_id,
            target.decoder_output_sha256,
            verdict=Verdict.SHOULD_BE_PARSEABLE,
            note="n" * ANNOTATION_NOTE_MAX_LENGTH,
            tag_ids=[tag.tag_id for tag in tags],
        )
        client = TestClient(
            create_app(service, static_dir=static_dir),
            base_url="http://127.0.0.1",
        )

        loaded = client.get(
            "/api/example",
            params={
                "run_id": descriptor.run_id,
                "sample_id": target.sample_id,
            },
        )
        loaded_annotation = loaded.json()["annotation"]
        unchanged = client.put(
            f"/api/annotations/{descriptor.corpus_sha256}/"
            f"{target.decoder_output_sha256}",
            params={"sample_id": target.sample_id},
            json={
                "verdict": loaded_annotation["verdict"],
                "note": loaded_annotation["note"],
                "tag_ids": [
                    *(tag["tag_id"] for tag in loaded_annotation["tags"]),
                    loaded_annotation["tags"][0]["tag_id"],
                ],
            },
        )

        assert loaded.status_code == 200
        assert len(loaded_annotation["note"]) == ANNOTATION_NOTE_MAX_LENGTH
        assert len(loaded_annotation["tags"]) == ANNOTATION_TAG_IDS_MAX_COUNT
        assert unchanged.status_code == 200
        assert (
            database.get_annotation(
                descriptor.corpus_sha256,
                target.sample_id,
                target.decoder_output_sha256,
            )
            == stored
        )


def test_delete_annotation_http_rejects_stale_target_without_mutation(
    tmp_path: Path, static_dir: Path
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    stale_output = "0" * 64
    with ViewerDatabase(":memory:") as database:
        service = ViewerAnalytics(database, [descriptor])
        database.put_annotation(
            descriptor.corpus_sha256,
            "no-code",
            stale_output,
            verdict=Verdict.EXPECTED_NO_CODE,
            note="seeded stale row",
        )
        client = TestClient(
            create_app(service, static_dir=static_dir),
            base_url="http://127.0.0.1",
        )

        response = client.delete(
            f"/api/annotations/{descriptor.corpus_sha256}/{stale_output}",
            params={"sample_id": "no-code"},
        )

        assert response.status_code == 400
        assert "not present" in response.json()["detail"]
        assert (
            database.get_annotation(
                descriptor.corpus_sha256, "no-code", stale_output
            )
            is not None
        )


def test_domain_errors_have_useful_http_status(static_dir: Path) -> None:
    class IncompatibleService(FakeService):
        def compare(self, baseline_run_id: str, candidate_run_id: str):
            del baseline_run_id, candidate_run_id
            raise IncompatibleRunsError("corpus fingerprints differ")

    response = TestClient(
        create_app(IncompatibleService(), static_dir=static_dir),
        base_url="http://127.0.0.1",
    ).get("/api/compare", params={"baseline": "a", "candidate": "b"})
    assert response.status_code == 409
    assert response.json() == {"detail": "corpus fingerprints differ"}


def test_api_handlers_keep_shared_duckdb_connection_off_threadpool(
    static_dir: Path,
) -> None:
    application = create_app(FakeService(), static_dir=static_dir)
    endpoints = [
        route.endpoint
        for route in application.routes
        if getattr(route, "path", "").startswith("/api/")
    ]

    assert endpoints
    assert all(inspect.iscoroutinefunction(endpoint) for endpoint in endpoints)


def test_untrusted_host_is_rejected_before_reads_or_mutations(
    static_dir: Path,
) -> None:
    service = FakeService()
    client = TestClient(
        create_app(service, static_dir=static_dir),
        base_url="http://127.0.0.1",
    )

    read = client.get("/api/runs", headers={"host": "attacker.example"})
    review = client.get(
        "/api/review-examples",
        headers={"host": "attacker.example"},
        params={
            "run_id": "baseline",
            "failure_code": "no_candidate",
            "failed_step": "extract_candidates",
        },
    )
    mutation = client.post(
        "/api/tags",
        headers={"host": "attacker.example"},
        json={"name": "must not be created"},
    )

    assert read.status_code == 400
    assert review.status_code == 400
    assert mutation.status_code == 400
    assert service.created_tag_names == []
    assert service.review_kwargs is None
    assert client.get("/api/runs").status_code == 200
    assert (
        client.get("/api/runs", headers={"host": "[::1]"}).status_code == 200
    )


def test_configured_loopback_host_is_trusted(static_dir: Path) -> None:
    client = TestClient(
        create_app(
            FakeService(),
            static_dir=static_dir,
            allowed_host="127.0.0.2",
        ),
        base_url="http://127.0.0.2",
    )

    assert client.get("/api/runs").status_code == 200
    assert (
        client.get("/api/runs", headers={"host": "127.0.0.1"}).status_code
        == 400
    )


def test_static_assets_use_spa_fallback_without_shadowing_api(
    tmp_path: Path,
) -> None:
    frontend = tmp_path / "dist"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>viewer</main>")
    (frontend / "app.js").write_text("console.log('viewer')")
    client = TestClient(
        create_app(FakeService(), static_dir=frontend),
        base_url="http://localhost",
    )

    assert client.get("/").text == "<main>viewer</main>"
    assert client.get("/review/sample-1").text == "<main>viewer</main>"
    assert client.get("/app.js").text == "console.log('viewer')"
    assert client.get("/api/unknown").status_code == 404
    assert "console.log" not in client.get("/%2E%2E/app.js").text


def test_missing_frontend_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ViewerAssetsError, match="no index.html"):
        create_app(FakeService(), static_dir=tmp_path / "missing")


def test_clean_source_archive_materializes_default_frontend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_package = Path(viewer_assets.__file__).parent
    package = tmp_path / "source-package"
    package.mkdir()
    for filename in (
        viewer_assets.ARCHIVE_FILENAME,
        viewer_assets.DIGEST_FILENAME,
    ):
        (package / filename).write_bytes(
            (source_package / filename).read_bytes()
        )
    cache_home = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    monkeypatch.setattr(viewer_app_module, "files", lambda _package: package)

    client = TestClient(create_app(FakeService()), base_url="http://127.0.0.1")

    response = client.get("/")
    assert response.status_code == 200
    assert 'id="root"' in response.text
    frontend = viewer_app_module._default_static_dir()
    assert frontend.is_relative_to(cache_home)
    assert (frontend / "assets").is_dir()


def test_malformed_installed_package_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(viewer_app_module, "files", lambda _package: tmp_path)

    with pytest.raises(ViewerAssetsError, match="installed dr-code package"):
        viewer_app_module._default_static_dir()
