"""Local DuckDB-backed preprocessing run viewer."""

from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.database import ViewerDatabase
from dr_code.viewer.domain import (
    Annotation,
    IncompatibleRunsError,
    InvalidQueryError,
    RunDescriptor,
    RunNotFoundError,
    RunValidationError,
    Tag,
    Verdict,
    ViewerError,
)

__all__ = (
    "Annotation",
    "IncompatibleRunsError",
    "InvalidQueryError",
    "RunDescriptor",
    "RunNotFoundError",
    "RunValidationError",
    "Tag",
    "Verdict",
    "ViewerAnalytics",
    "ViewerDatabase",
    "ViewerError",
)
