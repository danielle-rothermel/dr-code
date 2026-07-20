# HumanEval+ sandbox image

The default `SANDBOX_IMAGE` intentionally remains the pinned Python image
without optional packages. Use the image below only for an evaluation that
needs NumPy, and record its local immutable image ID with the evaluation
artifacts.

`docker/humaneval-plus/Dockerfile` pins the Python 3.13.14-slim base image to
the digest used by the sandbox and installs NumPy 2.2.6. NumPy 2.2.6 has
CPython 3.13 wheels.

Build it from the repository root:

```sh
docker build --pull \
  --file docker/humaneval-plus/Dockerfile \
  --tag dr-code-humaneval-plus:py313-numpy2.2.6 \
  .

image_id="$(docker image inspect --format '{{.Id}}' \
  dr-code-humaneval-plus:py313-numpy2.2.6)"
printf 'DR_CODE_SANDBOX_IMAGE=%s\n' "$image_id"
```

The printed value has the exact `sha256:<64 lowercase hex characters>` form.
`DR_CODE_SANDBOX_IMAGE` accepts that local image ID as well as a named
`name@sha256:<digest>` reference; it rejects mutable tags such as
`dr-code-humaneval-plus:py313-numpy2.2.6`.

Before scoring, run this no-network, read-only preflight against that exact
ID. It validates the precise Python and NumPy versions under the same core
container restrictions as the evaluator:

```sh
docker run --rm --pull=never --network=none --read-only \
  --user=65534:65534 --cap-drop=ALL \
  --security-opt=no-new-privileges --pids-limit=1 --cpus=1 \
  --memory=268435456 --memory-swap=268435456 \
  --ulimit fsize=1048576:1048576 --ulimit nofile=64:64 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,uid=65534,gid=65534,mode=700,size=16777216 \
  --workdir=/tmp --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  "$image_id" python -I -c \
  "import sys, numpy; assert sys.version_info[:3] == (3, 13, 14); assert numpy.__version__ == '2.2.6'; print(sys.version.split()[0], numpy.__version__)"
```

If the command prints `3.13.14 2.2.6`, export its ID for the score run:

```sh
export DR_CODE_SANDBOX_IMAGE="$image_id"
DR_CODE_RUN_SANDBOX_TESTS=1 uv run pytest tests/humaneval/test_sandbox.py
```
