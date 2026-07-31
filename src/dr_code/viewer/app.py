"""Typed HTTP boundary for the local preprocessing viewer."""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast

from fastapi import FastAPI, HTTPException, Path as ApiPath, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
)
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from dr_code.viewer.assets import (
    PrebuiltViewerAssetsError,
    materialize_prebuilt_viewer_assets,
    validate_static_directory,
)
from dr_code.viewer.domain import (
    Annotation as DomainAnnotation,
    ExampleDetail as DomainExampleDetail,
    Failures as DomainFailures,
    IncompatibleRunsError,
    InvalidQueryError,
    Page as DomainPage,
    ReviewPage as DomainReviewPage,
    RunComparison as DomainRunComparison,
    RunNotFoundError,
    RunSummary as DomainRunSummary,
    RunValidationError,
    Tag as DomainTag,
    Verdict as DomainVerdict,
    ViewerError,
    Waterfall as DomainWaterfall,
)

Scalar = str | int | float | bool | None
Verdict = Literal["should_be_parseable", "expected_no_code"]
Unit = Literal["sample", "candidate"]
Sha256Path = Annotated[str, ApiPath(pattern=r"^[0-9a-f]{64}$")]


class ViewerAssetsError(RuntimeError):
    """The installed viewer frontend resource is missing or malformed."""


class ResponseModel(BaseModel):
    """Response model that also accepts transport-neutral dataclasses."""

    model_config = ConfigDict(from_attributes=True)


class RunSummary(ResponseModel):
    run_id: str
    label: str
    manifest_sha256: str
    corpus_sha256: str
    definition_id: str
    has_evaluation: bool
    semantic_coordinates: dict[str, Scalar]


class WaterfallStage(ResponseModel):
    id: str
    label: str
    count: int = Field(ge=0)
    denominator_count: int = Field(ge=0)
    rate: float | None
    unit: Unit
    description: str | None = None
    failure_count: int | None = Field(default=None, ge=0)
    failure_label: str | None = None


class WaterfallResponse(ResponseModel):
    run_id: str
    stages: list[WaterfallStage]


class FailureGroup(ResponseModel):
    id: str
    label: str
    failure_code: str
    failed_step: str
    cause: str | None
    reason_code: str | None
    count: int = Field(ge=0)


class FailuresResponse(ResponseModel):
    run_id: str
    total_count: int = Field(ge=0)
    groups: list[FailureGroup]


class ExampleSummary(ResponseModel):
    sample_id: str
    outcome: str
    raw_preview: str | None
    context: dict[str, Scalar]
    annotation_verdict: Verdict | None


class ExamplesResponse(ResponseModel):
    items: list[ExampleSummary]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class ExtractionOperation(ResponseModel):
    kind: str
    details: dict[str, JsonValue]


class CandidateOrigin(ResponseModel):
    path: list[ExtractionOperation]


class Candidate(ResponseModel):
    candidate_id: str
    candidate_index: int = Field(ge=0)
    cleaned_source: str
    compile_warnings: list[str]
    origins: list[CandidateOrigin]
    top_level_function_names: list[str]


class DiagnosticFact(ResponseModel):
    step_name: str
    facts_json: str


class Rejection(ResponseModel):
    step_name: str
    reason_code: str | None
    details_json: str


class Tag(ResponseModel):
    tag_id: str
    name: str


class Annotation(ResponseModel):
    verdict: Verdict | None
    note: str | None
    tags: list[Tag]


class ExampleDetail(ResponseModel):
    corpus_sha256: str
    sample_id: str
    decoder_output_sha256: str | None
    outcome: str
    failure_code: str | None
    failed_step: str | None
    cause: str | None
    raw_decoder_output: str | None
    context: dict[str, Scalar]
    candidates: list[Candidate]
    facts: list[DiagnosticFact]
    rejections: list[Rejection]
    annotation: Annotation | None


class ReviewExamplesResponse(ResponseModel):
    items: list[ExampleDetail]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class ComparisonStage(ResponseModel):
    id: str
    label: str
    unit: Unit
    baseline_count: int = Field(ge=0)
    baseline_denominator_count: int = Field(ge=0)
    baseline_rate: float | None
    candidate_count: int = Field(ge=0)
    candidate_denominator_count: int = Field(ge=0)
    candidate_rate: float | None
    count_delta: int
    rate_delta: float | None


class OutcomeTransition(ResponseModel):
    id: str
    baseline_outcome: str
    candidate_outcome: str
    count: int = Field(ge=0)


class CompareResponse(ResponseModel):
    baseline_run_id: str
    candidate_run_id: str
    compatible: bool = True
    incompatibility_reason: str | None = None
    stages: list[ComparisonStage]
    transitions: list[OutcomeTransition]


class AnnotationExport(ResponseModel):
    corpus_sha256: str
    sample_id: str
    decoder_output_sha256: str
    verdict: Verdict | None
    note: str | None
    tags: list[str]


class CreateTagRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)

    @field_validator("name")
    @classmethod
    def normalized_name_is_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("tag name must not be blank")
        return value


class PutAnnotationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Verdict | None
    note: str | None = Field(default=None, max_length=10_000)
    tag_ids: list[str] = Field(default_factory=list, max_length=100)


class ViewerService(Protocol):
    """Transport-neutral operations required by the HTTP adapter."""

    def list_runs(self) -> Sequence[DomainRunSummary]: ...

    def waterfall(self, run_id: str) -> DomainWaterfall: ...

    def failures(self, run_id: str) -> DomainFailures: ...

    def examples(
        self,
        run_id: str,
        *,
        stage_id: str | None = None,
        failure_code: str | None = None,
        failed_step: str | None = None,
        cause: str | None = None,
        cause_is_null: bool = False,
        compare_run_id: str | None = None,
        baseline_outcome: str | None = None,
        candidate_outcome: str | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> DomainPage: ...

    def example(self, run_id: str, sample_id: str) -> DomainExampleDetail: ...

    def review_examples(
        self,
        run_id: str,
        *,
        failure_code: str,
        failed_step: str,
        cause: str | None = None,
        cause_is_null: bool = False,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> DomainReviewPage: ...

    def compare(
        self, baseline_run_id: str, candidate_run_id: str
    ) -> DomainRunComparison: ...

    def list_tags(self) -> Sequence[DomainTag]: ...

    def create_tag(self, name: str) -> DomainTag: ...

    def put_annotation(
        self,
        corpus_sha256: str,
        sample_id: str,
        decoder_output_sha256: str,
        *,
        verdict: DomainVerdict | str | None,
        note: str | None = None,
        tag_ids: Sequence[str] = (),
    ) -> DomainAnnotation: ...

    def delete_annotation(
        self,
        corpus_sha256: str,
        sample_id: str,
        decoder_output_sha256: str,
    ) -> bool: ...

    def export_annotations(self) -> Sequence[dict[str, object]]: ...


def _run(value: DomainRunSummary) -> RunSummary:
    return RunSummary(
        run_id=value.run_id,
        label=value.label,
        manifest_sha256=value.manifest_sha256,
        corpus_sha256=value.corpus_sha256,
        definition_id=value.definition_id,
        has_evaluation=value.has_evaluation,
        semantic_coordinates={
            "definition_version": value.definition_version,
            "definition_identity": value.definition_identity,
        },
    )


def _unit(value: str) -> Unit:
    if value not in {"sample", "candidate"}:
        raise ValueError(f"unsupported waterfall unit: {value}")
    return cast(Unit, value)


def _waterfall(value: DomainWaterfall) -> WaterfallResponse:
    return WaterfallResponse(
        run_id=value.run.run_id,
        stages=[
            WaterfallStage(
                id=stage.stage_id,
                label=stage.label,
                count=stage.count,
                denominator_count=stage.denominator,
                rate=stage.rate,
                unit=_unit(stage.unit),
            )
            for stage in value.stages
        ],
    )


def _failures(value: DomainFailures) -> FailuresResponse:
    groups = [
        FailureGroup(
            id=json.dumps(
                [group.failure_code, group.failed_step, group.cause],
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
            label=(group.cause or group.failure_code).replace("_", " "),
            failure_code=group.failure_code,
            failed_step=group.failed_step,
            cause=group.cause,
            reason_code=group.cause,
            count=group.count,
        )
        for group in value.groups
    ]
    return FailuresResponse(
        run_id=value.run.run_id,
        total_count=value.total_count,
        groups=groups,
    )


def _page(value: DomainPage) -> ExamplesResponse:
    return ExamplesResponse(
        items=[
            ExampleSummary(
                sample_id=item.sample_id,
                outcome=item.outcome,
                raw_preview=(
                    item.decoder_output[:300]
                    if item.decoder_output is not None
                    else None
                ),
                context={
                    key: context_value
                    for key, context_value in {
                        "task_id": item.task_id,
                        "failure_code": item.failure_code,
                        "failed_step": item.failed_step,
                    }.items()
                    if context_value is not None
                },
                annotation_verdict=(
                    item.annotation_verdict.value
                    if item.annotation_verdict is not None
                    else None
                ),
            )
            for item in value.items
        ],
        total=value.total,
        limit=value.limit,
        offset=value.offset,
    )


def _tag(value: DomainTag) -> Tag:
    return Tag(tag_id=str(value.tag_id), name=value.name)


def _annotation(value: DomainAnnotation) -> Annotation:
    return Annotation(
        verdict=value.verdict.value if value.verdict is not None else None,
        note=value.note,
        tags=[_tag(tag) for tag in value.tags],
    )


def _detail(value: DomainExampleDetail) -> ExampleDetail:
    context = {
        key: item
        for key, item in value.context.items()
        if isinstance(item, (str, int, float, bool, type(None)))
    }
    candidates = []
    for candidate in value.candidates:
        payload = dict(candidate)
        payload["compile_warnings"] = payload.get("compile_warnings") or []
        payload["origins"] = payload.get("origins") or []
        payload["top_level_function_names"] = (
            payload.get("top_level_function_names") or []
        )
        candidates.append(Candidate.model_validate(payload))
    return ExampleDetail(
        corpus_sha256=value.corpus_sha256,
        sample_id=value.sample_id,
        decoder_output_sha256=value.decoder_output_sha256,
        outcome=value.outcome,
        failure_code=value.failure_code,
        failed_step=value.failed_step,
        cause=value.cause,
        raw_decoder_output=value.raw_decoder_output,
        context=context,
        candidates=candidates,
        facts=[DiagnosticFact.model_validate(fact) for fact in value.facts],
        rejections=[
            Rejection.model_validate(rejection)
            for rejection in value.rejections
        ],
        annotation=(
            _annotation(value.annotation)
            if value.annotation is not None
            else None
        ),
    )


def _review_page(value: DomainReviewPage) -> ReviewExamplesResponse:
    return ReviewExamplesResponse(
        items=[_detail(item) for item in value.items],
        total=value.total,
        limit=value.limit,
        offset=value.offset,
    )


def _comparison(value: DomainRunComparison) -> CompareResponse:
    return CompareResponse(
        baseline_run_id=value.baseline.run_id,
        candidate_run_id=value.candidate.run_id,
        stages=[
            ComparisonStage(
                id=stage.stage_id,
                label=stage.label,
                unit=_unit(stage.unit),
                baseline_count=stage.baseline_count,
                baseline_denominator_count=stage.baseline_denominator_count,
                baseline_rate=stage.baseline_rate,
                candidate_count=stage.candidate_count,
                candidate_denominator_count=stage.candidate_denominator_count,
                candidate_rate=stage.candidate_rate,
                count_delta=stage.count_delta,
                rate_delta=stage.rate_delta,
            )
            for stage in value.stages
        ],
        transitions=[
            OutcomeTransition(
                id=(
                    f"{transition.baseline_outcome} → "
                    f"{transition.candidate_outcome}"
                ),
                baseline_outcome=transition.baseline_outcome,
                candidate_outcome=transition.candidate_outcome,
                count=transition.count,
            )
            for transition in value.transitions
        ],
    )


def _default_static_dir() -> Path:
    resource = files("dr_code.viewer")
    if not isinstance(resource, Path):
        raise ViewerAssetsError(
            "packaged viewer assets are not filesystem-backed"
        )
    package_dir = resource.resolve()
    frontend = package_dir / "static"
    try:
        validate_static_directory(frontend)
    except PrebuiltViewerAssetsError:
        try:
            return materialize_prebuilt_viewer_assets(package_dir)
        except PrebuiltViewerAssetsError as exc:
            raise ViewerAssetsError(
                "installed dr-code package is missing or has invalid "
                "viewer frontend assets"
            ) from exc
    else:
        return frontend


class _IPv6LoopbackHostMiddleware:
    """Normalize safe IPv6 loopback Hosts for Starlette's IPv4-style parser."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        headers = list(scope["headers"])
        for index, (name, value) in enumerate(headers):
            if name.lower() == b"host" and _is_ipv6_loopback_host(value):
                headers[index] = (name, b"localhost")
                normalized_scope = dict(scope)
                normalized_scope["headers"] = headers
                await self.app(normalized_scope, receive, send)
                return
        await self.app(scope, receive, send)


def _is_ipv6_loopback_host(value: bytes) -> bool:
    try:
        host = value.decode("ascii")
    except UnicodeDecodeError:
        return False
    if host.startswith("["):
        closing_bracket = host.find("]")
        if closing_bracket < 0:
            return False
        literal = host[1:closing_bracket]
        port = host[closing_bracket + 1 :]
        if port and (
            not port.startswith(":")
            or not port[1:].isdigit()
            or not 0 < int(port[1:]) <= 65_535
        ):
            return False
    else:
        literal = host
    try:
        address = ipaddress.ip_address(literal)
    except ValueError:
        return False
    return address.version == 6 and address.is_loopback


def create_app(
    service: ViewerService,
    *,
    static_dir: Path | None = None,
    allowed_host: str = "127.0.0.1",
) -> FastAPI:
    """Create an application around one process-local viewer service."""
    application = FastAPI(title="dr-code preprocessing viewer")
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(dict.fromkeys([allowed_host, "localhost"])),
        www_redirect=False,
    )
    # TrustedHostMiddleware splits on the first colon and cannot parse bracketed
    # IPv6 Hosts. Normalize only validated IPv6 loopback spellings before it.
    application.add_middleware(_IPv6LoopbackHostMiddleware)

    _install_error_handlers(application)

    # These async handlers intentionally call the synchronous service without
    # awaiting. All DuckDB work therefore stays serialized on the one event-loop
    # thread instead of FastAPI dispatching a shared connection to its threadpool.

    @application.get("/api/runs", response_model=list[RunSummary])
    async def list_runs() -> list[RunSummary]:
        return [_run(run) for run in service.list_runs()]

    @application.get("/api/waterfall", response_model=WaterfallResponse)
    async def waterfall(
        run_id: Annotated[str, Query(min_length=1, max_length=256)],
    ) -> WaterfallResponse:
        return _waterfall(service.waterfall(run_id))

    @application.get("/api/failures", response_model=FailuresResponse)
    async def failures(
        run_id: Annotated[str, Query(min_length=1, max_length=256)],
    ) -> FailuresResponse:
        return _failures(service.failures(run_id))

    @application.get("/api/examples", response_model=ExamplesResponse)
    async def examples(
        run_id: Annotated[str, Query(min_length=1, max_length=256)],
        stage_id: Annotated[str | None, Query(max_length=128)] = None,
        failure_code: Annotated[str | None, Query(max_length=256)] = None,
        failed_step: Annotated[str | None, Query(max_length=256)] = None,
        cause: Annotated[str | None, Query()] = None,
        cause_is_null: Annotated[bool, Query()] = False,
        compare_run_id: Annotated[str | None, Query(max_length=256)] = None,
        baseline_outcome: Annotated[str | None, Query(max_length=256)] = None,
        candidate_outcome: Annotated[str | None, Query(max_length=256)] = None,
        search: Annotated[str | None, Query(max_length=1_000)] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> ExamplesResponse:
        result = service.examples(
            run_id,
            stage_id=stage_id,
            failure_code=failure_code,
            failed_step=failed_step,
            cause=cause,
            cause_is_null=cause_is_null,
            compare_run_id=compare_run_id,
            baseline_outcome=baseline_outcome,
            candidate_outcome=candidate_outcome,
            search=search,
            limit=limit,
            offset=offset,
        )
        return _page(result)

    @application.get("/api/example", response_model=ExampleDetail)
    async def example(
        run_id: Annotated[str, Query(min_length=1, max_length=256)],
        sample_id: Annotated[str, Query(min_length=1, max_length=10_000)],
    ) -> ExampleDetail:
        return _detail(service.example(run_id, sample_id))

    @application.get(
        "/api/review-examples", response_model=ReviewExamplesResponse
    )
    async def review_examples(
        run_id: Annotated[str, Query(min_length=1, max_length=256)],
        failure_code: Annotated[str, Query(max_length=256)],
        failed_step: Annotated[str, Query(max_length=256)],
        cause: Annotated[str | None, Query()] = None,
        cause_is_null: Annotated[bool, Query()] = False,
        search: Annotated[str | None, Query(max_length=1_000)] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> ReviewExamplesResponse:
        if (cause is None and not cause_is_null) or (
            cause is not None and cause_is_null
        ):
            raise InvalidQueryError(
                "review examples require exactly one of cause or "
                "cause_is_null=true"
            )
        return _review_page(
            service.review_examples(
                run_id,
                failure_code=failure_code,
                failed_step=failed_step,
                cause=cause,
                cause_is_null=cause_is_null,
                search=search,
                limit=limit,
                offset=offset,
            )
        )

    @application.get("/api/compare", response_model=CompareResponse)
    async def compare(
        baseline: Annotated[str, Query(min_length=1, max_length=256)],
        candidate: Annotated[str, Query(min_length=1, max_length=256)],
    ) -> CompareResponse:
        return _comparison(service.compare(baseline, candidate))

    @application.get("/api/tags", response_model=list[Tag])
    async def list_tags() -> list[Tag]:
        return [_tag(tag) for tag in service.list_tags()]

    @application.post("/api/tags", response_model=Tag, status_code=201)
    async def create_tag(request: CreateTagRequest) -> Tag:
        return _tag(service.create_tag(request.name))

    @application.put(
        "/api/annotations/{corpus_sha256}/{decoder_output_sha256}",
        response_model=Annotation,
    )
    async def put_annotation(
        corpus_sha256: Sha256Path,
        decoder_output_sha256: Sha256Path,
        request: PutAnnotationRequest,
        sample_id: Annotated[str, Query(min_length=1, max_length=10_000)],
    ) -> Annotation:
        result = service.put_annotation(
            corpus_sha256,
            sample_id,
            decoder_output_sha256,
            verdict=request.verdict,
            note=request.note,
            tag_ids=request.tag_ids,
        )
        return _annotation(result)

    @application.delete(
        "/api/annotations/{corpus_sha256}/{decoder_output_sha256}",
        status_code=204,
    )
    async def delete_annotation(
        corpus_sha256: Sha256Path,
        decoder_output_sha256: Sha256Path,
        sample_id: Annotated[str, Query(min_length=1, max_length=10_000)],
    ) -> Response:
        service.delete_annotation(
            corpus_sha256, sample_id, decoder_output_sha256
        )
        return Response(status_code=204)

    @application.get(
        "/api/annotations/export", response_model=list[AnnotationExport]
    )
    async def export_annotations() -> list[AnnotationExport]:
        return [
            AnnotationExport.model_validate(item)
            for item in service.export_annotations()
        ]

    frontend = (
        static_dir.expanduser().resolve()
        if static_dir is not None
        else _default_static_dir()
    )
    index = frontend / "index.html"
    if not index.is_file():
        raise ViewerAssetsError(
            f"viewer frontend has no index.html: {frontend}"
        )

    @application.get("/{requested_path:path}", include_in_schema=False)
    def frontend_file(requested_path: str) -> FileResponse:
        if requested_path == "api" or requested_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (frontend / requested_path).resolve()
        if candidate.is_relative_to(frontend) and candidate.is_file():
            return FileResponse(candidate)
        # Client-side routes receive the app shell, never filesystem paths.
        return FileResponse(index)

    return application


def _install_error_handlers(application: FastAPI) -> None:
    """Translate domain errors without making the domain depend on FastAPI."""
    status_by_type = {
        RunNotFoundError: 404,
        InvalidQueryError: 400,
        IncompatibleRunsError: 409,
    }

    @application.exception_handler(ViewerError)
    async def viewer_error_handler(
        _request: object, error: Exception
    ) -> Response:
        status_code = next(
            (
                status
                for error_type, status in status_by_type.items()
                if isinstance(error, error_type)
            ),
            500,
        )
        return JSONResponse(
            status_code=status_code, content={"detail": str(error)}
        )

    @application.exception_handler(RunValidationError)
    async def run_validation_error_handler(
        _request: object, error: Exception
    ) -> Response:
        return JSONResponse(status_code=400, content={"detail": str(error)})
