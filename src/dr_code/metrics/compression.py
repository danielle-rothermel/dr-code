from __future__ import annotations

import gzip
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal, Self

import python_minifier
import zstandard
from pydantic import Field, model_validator

from dr_code.core.models import FrozenModel


class CompressionMethod(StrEnum):
    GZIP = "gzip"
    ZSTD = "zstd"


class GzipConfig(FrozenModel):
    method: Literal[CompressionMethod.GZIP] = CompressionMethod.GZIP
    level: int

    @model_validator(mode="after")
    def validate_level(self) -> Self:
        if not 0 <= self.level <= 9:
            raise ValueError("gzip level must be between 0 and 9")
        return self


class ZstdConfig(FrozenModel):
    method: Literal[CompressionMethod.ZSTD] = CompressionMethod.ZSTD
    level: int

    @model_validator(mode="after")
    def validate_level(self) -> Self:
        if self.level == 0 or self.level > 22:
            raise ValueError("zstd level must be negative or between 1 and 22")
        return self


CompressionConfig = Annotated[
    GzipConfig | ZstdConfig, Field(discriminator="method")
]


def compressed_bytes(
    value: bytes,
    *,
    method: CompressionMethod,
    level: int,
) -> bytes:
    if method is CompressionMethod.GZIP:
        return gzip.compress(value, compresslevel=level)
    if method is CompressionMethod.ZSTD:
        return zstd_compressed_bytes(value, level=level)
    raise ValueError(f"unsupported compression method: {method}")


def zstd_compressed_bytes(
    value: bytes,
    *,
    level: int,
    dictionary: bytes | None = None,
    compact_frame: bool = False,
) -> bytes:
    dictionary_data = (
        zstandard.ZstdCompressionDict(dictionary)
        if dictionary is not None
        else None
    )
    if compact_frame:
        compressor = zstandard.ZstdCompressor(
            level=level,
            dict_data=dictionary_data,
            write_content_size=False,
            write_checksum=False,
            write_dict_id=False,
        )
    else:
        compressor = zstandard.ZstdCompressor(
            level=level,
            dict_data=dictionary_data,
        )
    return compressor.compress(value)


def train_zstd_dictionary(
    samples: Sequence[bytes],
    *,
    dictionary_size: int,
) -> bytes:
    if dictionary_size <= 0:
        raise ValueError("dictionary size must be positive")
    non_empty_samples: list[bytes | bytearray | memoryview[int]] = [
        sample for sample in samples if sample
    ]
    if not non_empty_samples:
        raise ValueError(
            "at least one non-empty dictionary sample is required"
        )
    return zstandard.train_dictionary(
        dictionary_size,
        non_empty_samples,
    ).as_bytes()


def minify_python_source(
    source: str,
    *,
    public_names: Sequence[str] = (),
) -> str:
    minified = python_minifier.minify(
        source,
        remove_annotations=True,
        remove_pass=True,
        remove_literal_statements=True,
        combine_imports=True,
        hoist_literals=True,
        rename_locals=True,
        rename_globals=False,
        preserve_globals=list(public_names) or None,
        remove_object_base=True,
        convert_posargs_to_args=True,
        preserve_shebang=False,
        remove_asserts=False,
        remove_debug=False,
        remove_explicit_return_none=True,
        remove_builtin_exception_brackets=True,
        constant_folding=True,
        prefer_single_line=True,
    )
    if not minified.endswith("\n"):
        minified += "\n"
    return minified
