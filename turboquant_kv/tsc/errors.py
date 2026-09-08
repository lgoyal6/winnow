"""Errors for the tensor-schedule compiler, each carrying a source location."""
from __future__ import annotations


class TSCError(Exception):
    """Base class; `line`/`col` are 1-based positions in the source text."""

    def __init__(self, message: str, line: int | None = None,
                 col: int | None = None):
        self.line = line
        self.col = col
        where = f" (line {line}" + (f", col {col}" if col else "") + ")" \
            if line else ""
        super().__init__(message + where)


class TSCSyntaxError(TSCError):
    """Lexing or parsing failed."""


class TSCValidationError(TSCError):
    """The program parsed but violates a type, shape, bound, or order rule."""
