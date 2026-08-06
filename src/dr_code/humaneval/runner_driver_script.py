import json
import sys
import time
import traceback

FIELD_LIMIT = 8000


def dr_exec_main(request, emit):
    # Results travel on stdout; the protected protocol carries no outputs.
    del emit
    payload = request["payload"]
    # Redirection prevents accidental result collisions, not forgery.
    results_stream = sys.stdout
    sys.stdout = sys.stderr
    setattr(sys, "__stdout__", sys.stderr)

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

    def emit_output(output):
        results_stream.write(json.dumps(output))
        results_stream.write("\n")
        results_stream.flush()

    def build_support_namespace():
        namespace = {"assertion": assertion}
        exec(payload["support_code"], namespace)
        return namespace

    def build_candidate_namespace():
        namespace = build_support_namespace()
        exec(payload["candidate_code"], namespace)
        return namespace

    def failure_metadata(check):
        metadata = {
            "input_repr": check.get("input_repr", ""),
            "expected_output_repr": check.get("expected_output_repr", ""),
            "actual_output_repr": "",
        }
        try:
            detail_namespace = build_candidate_namespace()
            detail_candidate = detail_namespace[payload["function_name"]]
        except BaseException:
            metadata["actual_output_repr"] = clip(
                traceback.format_exc(limit=4)
            )
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
            metadata["actual_output_repr"] = clip(
                traceback.format_exc(limit=4)
            )

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
            metadata["expected_output_repr"] = clip(
                traceback.format_exc(limit=4)
            )

        return metadata

    setup_started_at = time.perf_counter()
    try:
        namespace = build_support_namespace()
    except BaseException:
        emit_output(
            {
                "kind": "harness_failure",
                "message": clip(traceback.format_exc(limit=4)),
                "elapsed_seconds": time.perf_counter() - setup_started_at,
            }
        )
        return

    try:
        exec(payload["candidate_code"], namespace)
        candidate = namespace[payload["function_name"]]
    except BaseException:
        emit_output(
            {
                "kind": "candidate_failure",
                "message": clip(traceback.format_exc(limit=4)),
                "elapsed_seconds": time.perf_counter() - setup_started_at,
            }
        )
        return

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
            results.append(
                {
                    "case_id": check["case_id"],
                    "status": "failed",
                    "message": clip(exc),
                    **failure_metadata(check),
                    "elapsed_seconds": time.perf_counter() - started_at,
                }
            )
        except BaseException:
            results.append(
                {
                    "case_id": check["case_id"],
                    "status": "error",
                    "message": clip(traceback.format_exc(limit=4)),
                    **failure_metadata(check),
                    "elapsed_seconds": time.perf_counter() - started_at,
                }
            )
        else:
            results.append(
                {
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
                }
            )
    emit_output({"kind": "case_results", "results": results})
