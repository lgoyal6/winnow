"""Lexer and recursive-descent parser for the tensor-schedule DSL.

Grammar (EBNF; `#` starts a comment that runs to end of line):

    program   := "pipeline" IDENT "(" [params] ")" "{" {decl} {stmt} "}"
    params    := param {"," param}
    param     := IDENT "=" INT
    decl      := ("input" | "output") IDENT ":" type
    type      := DTYPE "[" dim {"," dim} "]"
    dim       := INT | IDENT                 # literal, param ref, or symbol
    stmt      := IDENT "=" call | call       # bare call form is for store
    call      := OP "(" arg {"," arg} ")"
    arg       := IDENT "=" value | value
    value     := IDENT | INT | DTYPE
    DTYPE     := "u8" | "f16" | "bf16" | "f32"
    OP        := "load" | "unpack" | "gather" | "rot" | "rescale"
               | "cast" | "store"
"""
from __future__ import annotations

from dataclasses import dataclass

from tsc.errors import TSCSyntaxError
from tsc.syntax import (
    Assign, DTYPES, DtypeLit, ExprStmt, IntLit, OP_ATTRS, OPS, OpCall, Param,
    Program, Ref, TensorDecl, TensorType,
)

_PUNCT = "(){}[],:="
_KEYWORDS = ("pipeline", "input", "output")


@dataclass(frozen=True)
class Token:
    kind: str            # "ident" | "int" | punctuation char | "eof"
    text: str
    line: int
    col: int


def _lex(src: str) -> list[Token]:
    tokens: list[Token] = []
    line, col = 1, 1
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if ch == "\n":
            line += 1
            col = 1
            i += 1
        elif ch in " \t\r":
            i += 1
            col += 1
        elif ch == "#":
            while i < n and src[i] != "\n":
                i += 1
        elif ch in _PUNCT:
            tokens.append(Token(ch, ch, line, col))
            i += 1
            col += 1
        elif ch.isdigit():
            start = i
            while i < n and src[i].isdigit():
                i += 1
            tokens.append(Token("int", src[start:i], line, col))
            col += i - start
        elif ch.isalpha() or ch == "_":
            start = i
            while i < n and (src[i].isalnum() or src[i] == "_"):
                i += 1
            tokens.append(Token("ident", src[start:i], line, col))
            col += i - start
        else:
            raise TSCSyntaxError(f"unexpected character {ch!r}", line, col)
    tokens.append(Token("eof", "", line, col))
    return tokens


class _Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # -- token helpers ------------------------------------------------------
    def peek(self) -> Token:
        return self.tokens[self.pos]

    def next(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind: str, what: str | None = None) -> Token:
        tok = self.next()
        if tok.kind != kind:
            want = what or f"'{kind}'"
            got = tok.text or "end of input"
            raise TSCSyntaxError(f"expected {want}, got {got!r}",
                                 tok.line, tok.col)
        return tok

    def expect_ident(self, what: str) -> Token:
        return self.expect("ident", what)

    # -- grammar ------------------------------------------------------------
    def program(self) -> Program:
        kw = self.expect_ident("'pipeline'")
        if kw.text != "pipeline":
            raise TSCSyntaxError(f"expected 'pipeline', got {kw.text!r}",
                                 kw.line, kw.col)
        name = self.expect_ident("pipeline name")
        self.expect("(")
        params: list[Param] = []
        if self.peek().kind != ")":
            while True:
                pname = self.expect_ident("parameter name")
                self.expect("=")
                pval = self.expect("int", "integer parameter value")
                params.append(Param(pname.text, int(pval.text), pname.line))
                if self.peek().kind != ",":
                    break
                self.next()
        self.expect(")")
        self.expect("{")

        decls: list[TensorDecl] = []
        while self.peek().kind == "ident" and self.peek().text in ("input",
                                                                   "output"):
            decls.append(self.decl())

        stmts: list[object] = []
        while self.peek().kind != "}":
            stmts.append(self.stmt())
        self.expect("}")
        self.expect("eof", "end of input after '}'")
        return Program(name.text, tuple(params), tuple(decls), tuple(stmts))

    def decl(self) -> TensorDecl:
        kind = self.next()
        name = self.expect_ident("tensor name")
        if name.text in _KEYWORDS or name.text in OPS or name.text in DTYPES:
            raise TSCSyntaxError(
                f"{name.text!r} is reserved and cannot name a tensor",
                name.line, name.col)
        self.expect(":")
        return TensorDecl(kind.text, name.text, self.type(), kind.line)

    def type(self) -> TensorType:
        dt = self.expect_ident("a dtype")
        if dt.text not in DTYPES:
            raise TSCSyntaxError(
                f"unknown dtype {dt.text!r}; expected one of {', '.join(DTYPES)}",
                dt.line, dt.col)
        self.expect("[")
        dims: list[object] = []
        while True:
            tok = self.next()
            if tok.kind == "int":
                dims.append(int(tok.text))
            elif tok.kind == "ident":
                dims.append(tok.text)
            else:
                raise TSCSyntaxError("expected a dimension", tok.line, tok.col)
            if self.peek().kind != ",":
                break
            self.next()
        self.expect("]")
        return TensorType(dt.text, tuple(dims))

    def stmt(self):
        first = self.expect_ident("a statement")
        if self.peek().kind == "=":
            self.next()
            call = self.call()
            if call.op == "store":
                raise TSCSyntaxError(
                    "store produces no value and cannot be assigned",
                    first.line, first.col)
            return Assign(first.text, call, first.line)
        if first.text not in OPS:
            raise TSCSyntaxError(
                f"unknown op {first.text!r}; expected one of "
                f"{', '.join(sorted(OPS))}", first.line, first.col)
        call = self.call_body(first)
        if call.op != "store":
            raise TSCSyntaxError(
                f"result of {call.op!r} must be assigned to a name",
                first.line, first.col)
        return ExprStmt(call, first.line)

    def call(self) -> OpCall:
        op = self.expect_ident("an op name")
        if op.text not in OPS:
            raise TSCSyntaxError(
                f"unknown op {op.text!r}; expected one of "
                f"{', '.join(sorted(OPS))}", op.line, op.col)
        return self.call_body(op)

    def call_body(self, op: Token) -> OpCall:
        self.expect("(")
        args: list[object] = []
        attrs: list[tuple[str, object]] = []
        allowed_attrs = OP_ATTRS.get(op.text, ())
        if self.peek().kind != ")":
            while True:
                tok = self.next()
                if tok.kind == "ident" and self.peek().kind == "=":
                    if tok.text not in allowed_attrs:
                        raise TSCSyntaxError(
                            f"op {op.text!r} takes no attribute {tok.text!r}",
                            tok.line, tok.col)
                    self.next()
                    attrs.append((tok.text, self.value()))
                elif tok.kind == "ident":
                    if tok.text in DTYPES:
                        args.append(DtypeLit(tok.text, tok.line))
                    else:
                        args.append(Ref(tok.text, tok.line))
                elif tok.kind == "int":
                    args.append(IntLit(int(tok.text), tok.line))
                else:
                    raise TSCSyntaxError("expected an argument",
                                         tok.line, tok.col)
                if self.peek().kind != ",":
                    break
                self.next()
        close = self.expect(")")
        arity = OPS[op.text]
        if len(args) != arity:
            raise TSCSyntaxError(
                f"op {op.text!r} takes {arity} argument(s), got {len(args)}",
                op.line, op.col)
        for attr in allowed_attrs:
            if attr not in dict(attrs):
                raise TSCSyntaxError(
                    f"op {op.text!r} requires attribute {attr!r}",
                    close.line, close.col)
        return OpCall(op.text, tuple(args), tuple(attrs), op.line)

    def value(self):
        tok = self.next()
        if tok.kind == "int":
            return IntLit(int(tok.text), tok.line)
        if tok.kind == "ident":
            if tok.text in DTYPES:
                return DtypeLit(tok.text, tok.line)
            return Ref(tok.text, tok.line)
        raise TSCSyntaxError("expected a value", tok.line, tok.col)


def parse(src: str) -> Program:
    """Parse DSL source text into an untyped AST. Raises TSCSyntaxError."""
    return _Parser(_lex(src)).program()
