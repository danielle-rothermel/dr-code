# Standalone program executed inside the sandbox container via
# ``python -I -c <source>``. It reads one JSON line from stdin and prints a
# JSON list of case results. It must stay dependency-free (no ``dr_code``
# imports) because it runs in a locked-down, interpreter-isolated container,
# and it must NEVER be imported by host code: it has top-level side effects
# (it reads stdin at import time). The host reads this file's text via
# ``importlib.resources`` and executes it as a string; it does not import it.
import json
import time
import traceback

payload = json.loads(input())

FIELD_LIMIT = 8000

def clip(text):
    text = str(text)
    if len(text) > FIELD_LIMIT:
        return text[:FIELD_LIMIT] + "...[truncated]"
    return text

def assertion(actual, expected, atol=0):
    if atol:
        assert abs(actual - expected) <= atol
    else:
        assert actual == expected

def build_namespace():
    namespace = {"assertion": assertion}
    exec(payload["support_code"], namespace)
    exec(payload["candidate_code"], namespace)
    return namespace

def failure_metadata(check):
    metadata = {
        "input_repr": check.get("input_repr", ""),
        "expected_output_repr": check.get("expected_output_repr", ""),
        "actual_output_repr": "",
    }
    try:
        detail_namespace = build_namespace()
        detail_candidate = detail_namespace[payload["function_name"]]
    except BaseException:
        metadata["actual_output_repr"] = clip(
            traceback.format_exc(limit=4)
        )
        return metadata

    try:
        if check.get("actual_output_expr"):
            metadata["actual_output_repr"] = clip(repr(eval(
                check["actual_output_expr"],
                detail_namespace | {"candidate": detail_candidate},
            )))
    except BaseException:
        metadata["actual_output_repr"] = clip(
            traceback.format_exc(limit=4)
        )

    try:
        if check.get("expected_output_expr"):
            metadata["expected_output_repr"] = clip(repr(eval(
                check["expected_output_expr"],
                detail_namespace | {"candidate": detail_candidate},
            )))
    except BaseException:
        metadata["expected_output_repr"] = clip(
            traceback.format_exc(limit=4)
        )

    return metadata

try:
    namespace = build_namespace()
    candidate = namespace[payload["function_name"]]
except BaseException:
    message = clip(traceback.format_exc(limit=4))
    results = []
    for check in payload["checks"]:
        results.append({
            "case_id": check["case_id"],
            "status": "error",
            "message": message,
            "input_repr": check.get("input_repr", ""),
            "expected_output_repr": check.get(
                "expected_output_repr",
                "",
            ),
            "actual_output_repr": message,
            "elapsed_seconds": 0.0,
        })
    print(json.dumps(results))
    raise SystemExit(0)

results = []
for check in payload["checks"]:
    started_at = time.perf_counter()
    try:
        exec(
            compile(
                check["code"],
                f"<generated {check['case_id']}>",
                "exec",
            ),
            namespace | {"candidate": candidate},
        )
    except AssertionError as exc:
        results.append({
            "case_id": check["case_id"],
            "status": "failed",
            "message": clip(exc),
            **failure_metadata(check),
            "elapsed_seconds": time.perf_counter() - started_at,
        })
    except BaseException:
        results.append({
            "case_id": check["case_id"],
            "status": "error",
            "message": clip(traceback.format_exc(limit=4)),
            **failure_metadata(check),
            "elapsed_seconds": time.perf_counter() - started_at,
        })
    else:
        results.append({
            "case_id": check["case_id"],
            "status": "passed",
            "message": "",
            "input_repr": check.get("input_repr", ""),
            "expected_output_repr": check.get(
                "expected_output_repr",
                "",
            ),
            "actual_output_repr": "",
            "elapsed_seconds": time.perf_counter() - started_at,
        })
print(json.dumps(results))
