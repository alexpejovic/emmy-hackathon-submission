"""Fast f32 math functions: the hardware approximations ``ex2``, ``lg2``, ``sin`` and ``cos``
(about 2 ulp), as a fast-math build uses them."""

from __future__ import annotations

import math
from collections.abc import Callable

from ptxgen.builder import PTXKernel
from ptxgen.dtypes import lit

ONE, ZERO = lit(1.0, "f32"), lit(0.0, "f32")


def _f(k: PTXKernel, op: str, *srcs: str) -> str:
    return k.pure(op, "f32", *srcs)


def _p(k: PTXKernel, op: str, *srcs: str) -> str:
    return k.pure(op, "pred", *srcs)


def exp(k: PTXKernel, x: str) -> str:
    return _f(k, "ex2.approx.ftz.f32", _f(k, "mul.f32", x, lit(math.log2(math.e), "f32")))


def log(k: PTXKernel, x: str) -> str:
    return _f(k, "mul.f32", _f(k, "lg2.approx.ftz.f32", x), lit(math.log(2.0), "f32"))


def pow_(k: PTXKernel, x: str, y: str) -> str:
    """2^(y * log2|x|), with powf's sign rules: x < 0 gives -r for an odd integer y, r for an
    even one and nan for a fraction; y == 0 gives 1."""
    t = _f(k, "mul.f32", _f(k, "lg2.approx.ftz.f32", _f(k, "abs.f32", x)), y)
    r = _f(k, "ex2.approx.ftz.f32", t)
    whole = _p(k, "setp.eq.f32", _f(k, "cvt.rni.f32.f32", y), y)
    half = _f(k, "mul.f32", y, lit(0.5, "f32"))
    odd = _p(k, "and.pred", whole, _p(k, "setp.neu.f32", _f(k, "cvt.rni.f32.f32", half), half))
    neg = _p(k, "setp.lt.f32", x, ZERO)
    r = _f(k, "selp.f32", _f(k, "neg.f32", r), r, _p(k, "and.pred", neg, odd))
    r = _f(k, "selp.f32", r, lit(math.nan, "f32"), _p(k, "or.pred", whole, _p(k, "not.pred", neg)))
    return _f(k, "selp.f32", ONE, r, _p(k, "setp.eq.f32", y, ZERO))


def tanh(k: PTXKernel, x: str) -> str:
    """1 - 2 / (e^2x + 1): saturates cleanly to +-1 (tanh.approx is only good to 2^-11)."""
    e = _f(k, "ex2.approx.ftz.f32", _f(k, "mul.f32", x, lit(2 * math.log2(math.e), "f32")))
    r = _f(k, "rcp.approx.ftz.f32", _f(k, "add.f32", e, ONE))
    return _f(k, "fma.rn.f32", r, lit(-2.0, "f32"), ONE)


def erf(k: PTXKernel, x: str) -> str:
    """Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7): 1 - poly(t) e^(-x^2) with t = 1/(1 + p|x|),
    signed like x."""
    ax = _f(k, "abs.f32", x)
    t = _f(k, "rcp.approx.ftz.f32", _f(k, "fma.rn.f32", ax, lit(0.3275911, "f32"), ONE))
    poly = lit(1.061405429, "f32")
    for c in (-1.453152027, 1.421413741, -0.284496736, 0.254829592, 0.0):
        poly = _f(k, "fma.rn.f32", t, poly, lit(c, "f32"))
    e = exp(k, _f(k, "neg.f32", _f(k, "mul.f32", ax, ax)))
    return _f(k, "copysign.f32", x, _f(k, "fma.rn.f32", _f(k, "neg.f32", poly), e, ONE))


def sin(k: PTXKernel, x: str) -> str:
    return _f(k, "sin.approx.ftz.f32", x)


def cos(k: PTXKernel, x: str) -> str:
    return _f(k, "cos.approx.ftz.f32", x)


def sigmoid(k: PTXKernel, x: str) -> str:
    e = _f(k, "ex2.approx.ftz.f32", _f(k, "mul.f32", x, lit(-math.log2(math.e), "f32")))
    return _f(k, "rcp.approx.ftz.f32", _f(k, "add.f32", e, ONE))


F32_FUNCS: dict[str, Callable[..., str]] = {
    "exp": exp,
    "log": log,
    "pow": pow_,
    "tanh": tanh,
    "erf": erf,
    "sin": sin,
    "cos": cos,
    "sigmoid": sigmoid,
}
