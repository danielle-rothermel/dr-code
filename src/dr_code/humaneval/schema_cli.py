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
    def handle_invalid_for_json_schema(
        self,
        # Pydantic's CoreSchemaOrField parameter type is private.
        schema: Any,
        error_info: str,
    ) -> JsonSchemaValue:
        # Runtime-only fields have no JSON form and remain unconstrained.
        return {}


@app.callback()
def main() -> None:
    pass


@app.command(
    help=("Print a JSON Schema bundle for HumanEvalTask and EvalCaseSummary.")
)
def humaneval() -> None:
    from pydantic.json_schema import models_json_schema

    from dr_code.humaneval.task import EvalCaseSummary, HumanEvalTask

    mapping, definitions = models_json_schema(
        [
            (HumanEvalTask, VALIDATION_MODE),
            (EvalCaseSummary, VALIDATION_MODE),
        ],
        title=HUMANEVAL_SCHEMA_TITLE,
        schema_generator=LenientGenerateJsonSchema,
    )
    bundle = {
        "title": HUMANEVAL_SCHEMA_TITLE,
        "type": "object",
        "properties": {
            "task": mapping[(HumanEvalTask, VALIDATION_MODE)],
            "case_summary": mapping[(EvalCaseSummary, VALIDATION_MODE)],
        },
        "required": ["task", "case_summary"],
        "$defs": definitions["$defs"],
    }
    typer.echo(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
