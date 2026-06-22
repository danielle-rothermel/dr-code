# Sandlock COW fork mode

This is the version that might become very fast for a huge dataset.

Sandlock supports a COW-fork mode where an init_fn runs once, then many clone processes inherit that initialized state. The README says the clones use raw fork(2) with copy-on-write pages, and the project gives an example where init_fn loads expensive state once and work_fn processes each clone ID. The Sandlock blog explicitly lists “code evaluation at scale” as a use case: load test cases/reference implementations once, then run each candidate in a forked clone with memory caps and process limits.

Conceptually:
```python
from sandlock import Sandbox

CASES = None

def init():
    global CASES
    CASES = load_all_cases_and_tests()  # loaded once in parent/template process

def work(clone_id: int):
    case = CASES[clone_id]
    result = run_candidate_in_this_child(case)
    print(json.dumps(result), flush=True)

sb = Sandbox(
    fs_readable=["/usr", "/lib", "/lib64", "/bin", "/etc", "/proc", "/dev"],
    fs_writable=["/tmp"],
    clean_env=True,
    max_memory="256M",
    max_processes=4,
    init_fn=init,
    work_fn=work,
)

clones = sb.fork(len(CASES))
```

The upside is huge: you can avoid repeatedly importing your harness, repeatedly loading the dataset, and maybe even repeatedly starting a full Python interpreter. The candidate runs in a forked process; if it mutates globals, imports junk, calls sys.exit, leaks memory, or corrupts process-local state, that clone dies and the parent/template is unaffected because of process COW.
