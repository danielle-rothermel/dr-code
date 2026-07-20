# A NumPy-enabled HumanEval+ image. The base is pinned to the same immutable
# Python runtime used by the default sandbox image.
FROM python:3.13.14-slim@sha256:eb43ff125d8d58d7449dcba7d336c23bcac412f526d861db493b9994d8010280

# NumPy 2.2.6 publishes CPython 3.13 wheels. Keep this explicit so the image
# can be rebuilt and its resulting local image ID recorded for an evaluation.
RUN python -m pip install --disable-pip-version-check --no-cache-dir numpy==2.2.6
