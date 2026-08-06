from __future__ import annotations

import gzip

import zstandard

from dr_code.metrics.compression import (
    CompressionMethod,
    compressed_bytes,
    minify_python_source,
    train_zstd_dictionary,
    zstd_compressed_bytes,
)


def test_existing_compressors_keep_their_default_frames() -> None:
    value = b"repeated source text " * 20

    assert compressed_bytes(
        value, method=CompressionMethod.GZIP, level=6
    ) == gzip.compress(value, compresslevel=6)
    assert compressed_bytes(
        value, method=CompressionMethod.ZSTD, level=3
    ) == zstandard.ZstdCompressor(level=3).compress(value)


def test_compact_zstd_frame_supports_a_trained_dictionary() -> None:
    samples = [
        f"def function_{index}(value):\n    return value + {index}\n".encode()
        * 5
        for index in range(40)
    ]
    dictionary = train_zstd_dictionary(samples, dictionary_size=512)
    value = samples[7]

    compressed = zstd_compressed_bytes(
        value,
        level=22,
        dictionary=dictionary,
        compact_frame=True,
    )

    decompressor = zstandard.ZstdDecompressor(
        dict_data=zstandard.ZstdCompressionDict(dictionary)
    )
    assert (
        decompressor.decompress(compressed, max_output_size=len(value))
        == value
    )


def test_minification_preserves_the_public_function_name() -> None:
    source = '''
def answer(value: int) -> int:
    """Return the answer."""
    intermediate = value + 1
    return intermediate
'''

    minified = minify_python_source(source, public_names=("answer",))

    assert "answer" in minified
    assert "Return the answer" not in minified
    assert "int" not in minified
    compile(minified, "<minified>", "exec")
