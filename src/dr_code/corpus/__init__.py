"""Durable corpus projections and candidate evaluation."""

from dr_code.corpus.candidate_evaluation import (
    CandidateEvaluationError,
    EvaluationArtifacts,
    evaluate_preprocessing_candidates,
)
from dr_code.corpus.candidate_evaluation_contract import (
    CANDIDATE_EVALUATION_SCHEMA_VERSION,
    CandidateEvaluationContractError,
    candidate_evaluation_identity,
    candidate_evaluation_key,
    validate_candidate_result,
)
from dr_code.corpus.evaluation_generation import (
    EvaluationGeneration,
    EvaluationGenerationError,
    resolve_current_generation,
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
    validate_origin_paths,
    write_projected_part,
)
from dr_code.corpus.preprocessing_run import (
    CorpusRunError,
    run_preprocessing_corpus,
)
from dr_code.corpus.preprocessing_analysis import (
    PreprocessingAnalysisArtifacts,
    PreprocessingAnalysisError,
    analyze_preprocessing_corpus,
)
from dr_code.corpus.preprocessing_comparison import (
    PreprocessingComparisonArtifacts,
    PreprocessingComparisonError,
    compare_preprocessing_runs,
)
from dr_code.corpus.run_descriptor import (
    RunDescriptor,
    RunValidationError,
)

__all__ = [
    "AtomicProjectedPartWriter",
    "CANDIDATES_SCHEMA",
    "CandidateEvaluationError",
    "CandidateEvaluationContractError",
    "CANDIDATE_EVALUATION_SCHEMA_VERSION",
    "CorpusRunError",
    "EvaluationArtifacts",
    "EvaluationGeneration",
    "EvaluationGenerationError",
    "PROJECTED_ARTIFACT_SCHEMAS",
    "ProjectedArtifacts",
    "ProjectedPart",
    "PreprocessingAnalysisArtifacts",
    "PreprocessingAnalysisError",
    "PreprocessingComparisonArtifacts",
    "PreprocessingComparisonError",
    "REJECTIONS_SCHEMA",
    "RESULTS_SCHEMA",
    "STEP_FACTS_SCHEMA",
    "RunDescriptor",
    "RunValidationError",
    "analyze_preprocessing_corpus",
    "combine_projected_parts",
    "candidate_evaluation_identity",
    "candidate_evaluation_key",
    "compare_preprocessing_runs",
    "evaluate_preprocessing_candidates",
    "project_preprocessing_result",
    "resolve_current_generation",
    "run_preprocessing_corpus",
    "validate_origin_paths",
    "validate_candidate_result",
    "write_projected_part",
]
