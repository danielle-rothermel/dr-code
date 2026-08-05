from enum import StrEnum


class MetricName(StrEnum):
    TEXT_STATS = "text_stats"
    CODE_LEAKAGE = "code_leakage"
    PARSE_OUTCOME = "parse_outcome"
    AST_STATS = "ast_stats"
    COMPRESSED_LENGTH = "compressed_length"
    CODE_TEST = "code_test"
