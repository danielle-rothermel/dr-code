"""FastAPI facade over the explain API (requires the [serve] extra)."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, StrictStr

from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    DEFAULT_CODE_FIELD,
    PARSER_PROFILE_VERSION,
    STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
)
from dr_code.serve.explain import (
    ExplainStage,
    ExtractionExplanation,
    explain_extraction,
)

SERVE_TITLE = "dr-code serve"
SERVE_VERSION = "0.1.0"

PARSER_PROFILE_IDS = [
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
]


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: StrictStr
    profile_id: StrictStr = BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID
    parser_version: StrictStr = PARSER_PROFILE_VERSION
    code_field: StrictStr = DEFAULT_CODE_FIELD
    stages: list[ExplainStage] | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


class ProfilesResponse(BaseModel):
    profile_ids: list[str]
    parser_version: str


def create_app() -> FastAPI:
    app = FastAPI(title=SERVE_TITLE, version=SERVE_VERSION)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=SERVE_VERSION)

    @app.get("/profiles", response_model=ProfilesResponse)
    def profiles() -> ProfilesResponse:
        return ProfilesResponse(
            profile_ids=PARSER_PROFILE_IDS,
            parser_version=PARSER_PROFILE_VERSION,
        )

    @app.post("/explain", response_model=ExtractionExplanation)
    def explain(request: ExplainRequest) -> ExtractionExplanation:
        stages = (
            frozenset(request.stages) if request.stages is not None else None
        )
        try:
            return explain_extraction(
                request.text,
                profile_id=request.profile_id,
                parser_version=request.parser_version,
                code_field=request.code_field,
                stages=stages,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
