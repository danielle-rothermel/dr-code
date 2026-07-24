from __future__ import annotations

import hashlib
from pathlib import Path

from dr_code.corpus.stable_files import stable_file


def test_stable_file_consumes_the_exact_captured_bytes_after_source_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mutable"
    source.write_bytes(b"authenticated")

    with stable_file(source) as captured:
        source.write_bytes(b"mutated")
        assert captured.path.read_bytes() == b"authenticated"
        assert captured.sha256 == hashlib.sha256(b"authenticated").hexdigest()
        assert captured.descriptor() == {
            "sha256": captured.sha256,
            "size": len(b"authenticated"),
        }
