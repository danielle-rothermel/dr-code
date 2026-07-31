"""Local DuckDB-backed preprocessing run viewer."""

from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.database import DatabaseSchemaError, ViewerDatabase
from dr_code.viewer.domain import (
    Annotation,
    IncompatibleRunsError,
    InvalidQueryError,
    InvalidTaskAnnotationError,
    MachineTaskAnnotationWriteOutcome,
    MachineTaskAnnotationWriteResult,
    RunDescriptor,
    RunNotFoundError,
    RunValidationError,
    Tag,
    TaskAnnotation,
    TaskAnnotationOrigin,
    TaskAnnotationProvenance,
    TaskIdentity,
    TaskNotFoundError,
    Verdict,
    ViewerError,
)

__all__ = (
    "Annotation",
    "DatabaseSchemaError",
    "IncompatibleRunsError",
    "InvalidQueryError",
    "InvalidTaskAnnotationError",
    "MachineTaskAnnotationWriteOutcome",
    "MachineTaskAnnotationWriteResult",
    "RunDescriptor",
    "RunNotFoundError",
    "RunValidationError",
    "Tag",
    "TaskAnnotation",
    "TaskAnnotationOrigin",
    "TaskAnnotationProvenance",
    "TaskIdentity",
    "TaskNotFoundError",
    "Verdict",
    "ViewerAnalytics",
    "ViewerDatabase",
    "ViewerError",
)
