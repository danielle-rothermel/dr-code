# dr-code

## Python execution

`dr_code.execution.run_python_subprocess` runs Python source in a fresh
`sys.executable -I` process with bounded text input, a shared stdout/stderr
limit, a wall-clock deadline, and process-group cleanup. HumanEval uses this
primitive through an injectable batch-runner interface.

This execution boundary provides no operating-system containment. Candidate
code has the worker's filesystem, credential, process, and network permissions.
Process-group cleanup cannot guarantee termination of descendants that detach
from the group. Run evaluations only on disposable workers whose permissions,
network access, resources, and lifetime are constrained externally.
