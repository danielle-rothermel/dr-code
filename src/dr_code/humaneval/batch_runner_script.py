# The HumanEval driver body composed into dr-exec's batch kit and run in a
# hermetic ``python -I -c`` child. It defines ``run_item(item_id, payload)``,
# which the kit calls once per case; the kit owns protocol I/O, prelude, the
# terminal line, and per-item failure capture. This text is read as a resource
# and executed by the kit, so it must stay dependency-free (no ``dr_code``
# imports) and must never be imported by host code.
#
# The candidate namespace is built once at body load: ``build_namespace`` execs
# the support and candidate code and resolves the target function. A load
# failure is raised here and the kit fans it out as one error result per case.
# Each case's ``run_item`` execs the case check against a fresh candidate
# binding; on failure it re-executes the namespace to recover input/expected/
# actual reprs (``failure_metadata``), matching the in-child diagnostics the
# scoring layer persists.
import time
import traceback

FIELD_LIMIT = 8000

# Injected as literals by the request builder; see batch_runner.compose_body.
CANDIDATE_CODE = _HE_CANDIDATE_CODE  # noqa: F821
SUPPORT_CODE = _HE_SUPPORT_CODE  # noqa: F821
FUNCTION_NAME = _HE_FUNCTION_NAME  # noqa: F821


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
    exec(SUPPORT_CODE, namespace)
    exec(CANDIDATE_CODE, namespace)
    return namespace


def failure_metadata(check):
    metadata = {
        "input_repr": check.get("input_repr", ""),
        "expected_output_repr": check.get("expected_output_repr", ""),
        "actual_output_repr": "",
    }
    try:
        detail_namespace = build_namespace()
        detail_candidate = detail_namespace[FUNCTION_NAME]
    except BaseException:
        metadata["actual_output_repr"] = clip(traceback.format_exc(limit=4))
        return metadata

    try:
        if check.get("actual_output_expr"):
            metadata["actual_output_repr"] = clip(
                repr(
                    eval(
                        check["actual_output_expr"],
                        detail_namespace | {"candidate": detail_candidate},
                    )
                )
            )
    except BaseException:
        metadata["actual_output_repr"] = clip(traceback.format_exc(limit=4))

    try:
        if check.get("expected_output_expr"):
            metadata["expected_output_repr"] = clip(
                repr(
                    eval(
                        check["expected_output_expr"],
                        detail_namespace | {"candidate": detail_candidate},
                    )
                )
            )
    except BaseException:
        metadata["expected_output_repr"] = clip(traceback.format_exc(limit=4))

    return metadata


# Built once at load. A load failure raises out of the body, so the kit fans
# out the candidate traceback as one error result per case.
_NAMESPACE = build_namespace()
_CANDIDATE = _NAMESPACE[FUNCTION_NAME]


def _passed_payload(check, elapsed):
    return {
        "status": "passed",
        "message": "",
        "input_repr": check.get("input_repr", ""),
        "expected_output_repr": check.get("expected_output_repr", ""),
        "actual_output_repr": "",
        "elapsed_seconds": elapsed,
    }


def run_item(item_id, payload):
    check = payload
    started_at = time.perf_counter()
    try:
        exec(
            compile(check["code"], "<generated " + item_id + ">", "exec"),
            _NAMESPACE | {"candidate": _CANDIDATE},
        )
    except AssertionError as exc:
        return {
            "status": "failed",
            "message": clip(exc),
            **failure_metadata(check),
            "elapsed_seconds": time.perf_counter() - started_at,
        }
    except BaseException:
        return {
            "status": "error",
            "message": clip(traceback.format_exc(limit=4)),
            **failure_metadata(check),
            "elapsed_seconds": time.perf_counter() - started_at,
        }
    return _passed_payload(check, time.perf_counter() - started_at)
