"""Dump JSON Schemas for library models (input for generated TS types).

The viewer package generates TypeScript types from this output instead of
minting facade endpoints for library-only models (ADR 0002).
"""

import json
from typing import Any

import typer
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue

app = typer.Typer(
    help="dr-code library JSON Schema dumps", no_args_is_help=True
)

HUMANEVAL_SCHEMA_TITLE = "HumanEvalLibrarySchemas"
VALIDATION_MODE = "validation"


class LenientGenerateJsonSchema(GenerateJsonSchema):
    # `schema` is Any because the parent takes pydantic's private
    # CoreSchemaOrField union, which is not importable publicly.
    def handle_invalid_for_json_schema(
        self,
        schema: Any,
        error_info: str,
    ) -> JsonSchemaValue:
        # Runtime-only fields (e.g. ParsedCode's ast.AST) have no JSON form;
        # emit an unconstrained schema so codegen types them as `unknown`
        # instead of failing the whole dump.
        return {}


@app.callback()
def main() -> None:
    """Dump JSON Schemas for library models."""


@app.command()
def humaneval() -> None:
    """Print a JSON Schema bundle for HumanEvalTask and EvaluationCaseSummary."""
    from pydantic.json_schema import models_json_schema

    from dr_code.humaneval.task import EvaluationCaseSummary, HumanEvalTask

    mapping, definitions = models_json_schema(
        [
            (HumanEvalTask, VALIDATION_MODE),
            (EvaluationCaseSummary, VALIDATION_MODE),
        ],
        title=HUMANEVAL_SCHEMA_TITLE,
        schema_generator=LenientGenerateJsonSchema,
    )
    bundle = {
        "title": HUMANEVAL_SCHEMA_TITLE,
        "type": "object",
        "properties": {
            "task": mapping[(HumanEvalTask, VALIDATION_MODE)],
            "case_summary": mapping[(EvaluationCaseSummary, VALIDATION_MODE)],
        },
        "required": ["task", "case_summary"],
        "$defs": definitions["$defs"],
    }
    typer.echo(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    app()
