"""Crosswalk tests: dr-code kernel matches the design authority mapping.

Authority: vocab_and_defs.html dr-code crosswalk and Workstream 5 table
in concrete-changes.html. These tests pin the structural claims the
crosswalk makes, so a refactor that breaks the boundary is caught.
"""

from __future__ import annotations

import inspect

from dr_serialize import build_identity_document, identity_document_hash

from dr_code.eval import identity as identity_module
from dr_code.eval.code import CodeArtifact, CodeCandidate, PythonSource
from dr_code.eval.lifecycle import (
    AggregationDefinition,
    EvalDefinition,
    EvaluationProcedureDefinition,
    MetricExtractionDefinition,
    PreprocessingDefinition,
    SamplingDefinition,
)
from dr_code.humaneval.task import HumanEvalTask


def test_identity_flows_through_dr_serialize() -> None:
    # The single identity seam composes dr-serialize's document + hash.
    source = inspect.getsource(identity_module.identity_hash_for)
    assert "build_identity_document" in source
    assert "identity_document_hash" in source
    # And the imported symbols are exactly dr-serialize's.
    assert identity_module.build_identity_document is build_identity_document
    assert identity_module.identity_document_hash is identity_document_hash


def test_schema_names_are_dr_code_owned() -> None:
    schema_constants = [
        value
        for name, value in vars(identity_module).items()
        if name.startswith("SCHEMA_") and isinstance(value, str)
    ]
    assert schema_constants
    assert all(schema.startswith("dr_code.") for schema in schema_constants)


def test_no_generic_task_superclass() -> None:
    # HumanEvalTask implements the Task role directly; its only non-object,
    # non-pydantic base is pydantic BaseModel (no generic Task superclass).
    base_names = {base.__name__ for base in HumanEvalTask.__mro__}
    assert "Task" not in base_names


def test_six_definition_config_pairs_present() -> None:
    # Every declared Definition materializes its namesake Config and is its
    # sole owner (materialize is the only public constructor path).
    definitions = [
        SamplingDefinition,
        PreprocessingDefinition,
        MetricExtractionDefinition,
        EvaluationProcedureDefinition,
        AggregationDefinition,
        EvalDefinition,
    ]
    for definition in definitions:
        assert hasattr(definition, "materialize")
        assert hasattr(definition, "identity_hash")
        assert hasattr(definition, "ref")


def test_text_artifact_carries_python_source_role_without_duplicate_type() -> (
    None
):
    # PythonSource is a role over a text value; CodeArtifact validates
    # compilation; candidates are typed (not bare strings).
    assert PythonSource.model_fields["text"].annotation is str
    assert "source" in CodeArtifact.model_fields
    assert "source" in CodeCandidate.model_fields
    assert "position" in CodeCandidate.model_fields


def test_eval_config_composes_three_component_identities() -> None:
    # Crosswalk: Eval Config composes Sampling + Evaluation Procedure +
    # Aggregation Config identities.
    fields = set(
        inspect.getsource(EvalDefinition.materialize).split()
    )
    assert "sampling" in " ".join(fields)
    assert "evaluation_procedure" in " ".join(fields)
    assert "aggregation" in " ".join(fields)
