"""Corpus-facing artifacts derived from preprocessing traces."""

from dr_code.corpus.candidate_evaluation import (
    CandidateEvaluationError,
    EvaluationArtifacts,
    evaluate_preprocessing_candidates,
)

from dr_code.corpus.preprocessing_artifacts import (
    AtomicProjectedPartWriter,
    CANDIDATES_SCHEMA,
    PROJECTED_ARTIFACT_SCHEMAS,
    REJECTIONS_SCHEMA,
    RESULTS_SCHEMA,
    STEP_FACTS_SCHEMA,
    ProjectedArtifacts,
    ProjectedPart,
    combine_projected_parts,
    project_preprocessing_result,
    write_projected_part,
)
from dr_code.corpus.preprocessing_analysis import (
    PreprocessingAnalysisArtifacts,
    PreprocessingAnalysisError,
    analyze_preprocessing_corpus,
)

__all__ = (
    "CandidateEvaluationError",
    "EvaluationArtifacts",
    "CANDIDATES_SCHEMA",
    "AtomicProjectedPartWriter",
    "PROJECTED_ARTIFACT_SCHEMAS",
    "REJECTIONS_SCHEMA",
    "RESULTS_SCHEMA",
    "STEP_FACTS_SCHEMA",
    "ProjectedArtifacts",
    "ProjectedPart",
    "PreprocessingAnalysisArtifacts",
    "PreprocessingAnalysisError",
    "analyze_preprocessing_corpus",
    "combine_projected_parts",
    "evaluate_preprocessing_candidates",
    "project_preprocessing_result",
    "write_projected_part",
)
