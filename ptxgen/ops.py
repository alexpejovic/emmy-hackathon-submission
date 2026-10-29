"""Instruction sequences shared by both math modes: arithmetic per dtype, compares, conversions,
memory accesses and shuffles, and Emmy's dtype rule over data values. Float ops round to nearest:
add, sub and mul by default (no modifier, so ptxas may also fuse a multiply and an add into an fma),
the rest with an explicit ``.rn``, which PTX requires there."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from emmy.compiler.ir.stmt.base import dtype_promote

from ptxgen import fastmath
from ptxgen.builder import PTXKernel
from ptxgen.dtypes import BYTES, FLOATS, is_int, lit, mov_type, ptx_type

if TYPE_CHECKING:
    from emmy.compiler.ir.kernel import Assign

Op = Callable[..., str]  # (kernel, *argument registers) -> result register

# Other spellings of Emmy's elementwise op names.
ALIASES = {"sub": "subtract", "mul": "multiply", "div": "divide", "max": "maximum"}
ALIASES |= {"min": "minimum", "sum": "add", "prod": "multiply", "amax": "maximum"}
BINARY = {"add", "subtract", "multiply", "divide", "maximum", "minimum", "pow"}
# Ops an accumulator folds with -> the instruction for a float and for an integer accumulator.
ACCUM = {"add": ("add", "add"), "subtract": ("sub", "sub"), "multiply": ("mul", "mul.lo")}
ACCUM |= {"divide": ("div.rn", "div"), "maximum": ("max", "max"), "minimum": ("min", "min")}

# Emmy compare op -> setp condition; compares produce a bool (a predicate register).
COMPARE = {"equal": "eq", "not_equal": "ne", "greater": "gt", "less": "lt"}
COMPARE |= {"greater_equal": "ge", "less_equal": "le"}
# An expression's compare operator -> setp condition.
CMP_SYMBOL = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}


def canonical(name: str) -> str:
    return ALIASES.get(name, name)


def const(k: PTXKernel, value: float, typ: str) -> str:
    """A register holding ``value`` (16-bit float arithmetic takes no immediates)."""
    return k.pure(f"mov.{mov_type(typ)}", typ, lit(value, typ))


def _bin(m: str, t: str, k: PTXKernel, a: str, b: str) -> str:
    return k.pure(f"{m}.{t}", t, a, b)


def _un(m: str, t: str, k: PTXKernel, x: str) -> str:
    return k.pure(f"{m}.{t}", t, x)


def fma_operands(k: PTXKernel, t: str, a: str, b: str) -> tuple[str, str, str] | None:
    """``(x, y, c)`` when ``a + b`` is ``x * y + c`` with a still-cached ``mul`` of ``t``;
    always None with ``k.no_fusion``."""
    if k.no_fusion or t not in ("f32", "f64"):
        return None
    for prod, c in ((a, b), (b, a)):
        key = k.definition(prod)
        if key and key[0] == f"mul.{t}":
            return key[1], key[2], c
    return None


def _add(t: str, k: PTXKernel, a: str, b: str) -> str:
    if (fma := fma_operands(k, t, a, b)) is not None:
        return k.pure(f"fma.rn.{t}", t, *fma)
    return _bin("add", t, k, a, b)


def float_ops(t: str) -> dict[str, Op]:
    """Arithmetic every float type ``t`` shares, IEEE round-to-nearest."""

    def zero(k: PTXKernel) -> str:  # 16-bit float arithmetic takes no immediates
        return const(k, 0.0, t) if t == "f16" else lit(0.0, t)

    return {
        "add": partial(_add, t),
        "subtract": partial(_bin, "sub", t),
        "multiply": partial(_bin, "mul", t),
        "divide": partial(_bin, "div.rn", t),
        "maximum": partial(_bin, "max", t),
        "minimum": partial(_bin, "min", t),
        "relu": lambda k, x: _bin("max", t, k, x, zero(k)),
        "negative": partial(_un, "neg", t),
        "abs": partial(_un, "abs", t),
        "square": lambda k, x: _bin("mul", t, k, x, x),
        "reciprocal": partial(_un, "rcp.rn", t),
        "sqrt": partial(_un, "sqrt.rn", t),
    }


F32_OPS: dict[str, Op] = {**float_ops("f32"), "rsqrt": partial(_un, "rsqrt.approx", "f32")}
F64_OPS = float_ops("f64")
# f16 computes natively only where Emmy does (simple ops); the rest go through f32.
F16_OPS = {
    n: f for n, f in float_ops("f16").items() if n not in ("divide", "reciprocal", "sqrt", "square")
}
# Integer ops Emmy keeps integer: op -> (mnemonic, whether it takes a .b<bits> suffix).
INT_OPS = {
    "add": ("add", False),
    "subtract": ("sub", False),
    "multiply": ("mul.lo", False),
    "floor_divide": ("div", False),
    "remainder": ("rem", False),
    "mod": ("rem", False),
    "left_shift": ("shl", True),
    "right_shift": ("shr", False),  # .s for signed types: an arithmetic shift, as C's >>
    "bitwise_and": ("and", True),
    "bitwise_or": ("or", True),
    "bitwise_xor": ("xor", True),
}


def _int_op(name: str, t: str, k: PTXKernel, a: str, b: str) -> str:
    m, bits = INT_OPS[name]
    if m in ("shl", "shr") and BYTES[t] != 4:
        raise NotImplementedError(f"{t} op {name}: shift of a non-32-bit integer")
    return k.pure(f"{m}.{f'b{BYTES[t] * 8}' if bits else t}", t, a, b)


def cvt(k: PTXKernel, x: str, src: str, dst: str) -> str:
    """Register ``x`` from PTX type ``src`` to ``dst`` with C's rules: floats round to nearest,
    float to integer truncates. Two non-f32 floats convert through f32 (exact for 16-bit ones)."""
    if src == dst:
        return x
    if src in FLOATS and dst in FLOATS and "f32" not in (src, dst):
        return cvt(k, cvt(k, x, src, "f32"), "f32", dst)
    if src == "bf16" and dst == "f32":  # cvt.f32.bf16 needs sm_90: shift the bits up instead
        wide = k.pure("cvt.u32.u16", "u32", x)
        return k.pure("mov.b32", "f32", k.pure("shl.b32", "u32", wide, "16"))
    if src in FLOATS and dst in FLOATS:
        rnd = "" if BYTES[dst] > BYTES[src] else ".rn"  # widening is exact
    else:
        rnd = ".rn" if dst in FLOATS else ".rzi" if src in FLOATS else ""
    return k.pure(f"cvt{rnd}.{dst}.{src}", dst, x)


# --- data values ----------------------------------------------------------------------


@dataclass(frozen=True)
class Val:
    """A data value: the register holding it and its Emmy dtype (a bool is a predicate)."""

    reg: str
    dtype: str


def convert(k: PTXKernel, v: Val, dtype: str) -> Val:
    """v at Emmy dtype ``dtype``: rounding conversions, bool as 1 / 0, nonzero as true."""
    if v.dtype == dtype:
        return v
    typ = ptx_type(dtype)
    if v.dtype == "bool":
        return select(k, v.reg, lit(1.0, typ), lit(0.0, typ), dtype)
    if dtype == "bool":
        x = convert(k, v, "f32" if ptx_type(v.dtype) in FLOATS else "i32")
        zero = lit(0.0, ptx_type(x.dtype))
        return Val(k.pure(f"setp.ne.{ptx_type(x.dtype)}", "pred", x.reg, zero), dtype)
    return Val(cvt(k, v.reg, ptx_type(v.dtype), typ), dtype)


def select(k: PTXKernel, cond: str, a: str, b: str, dtype: str) -> Val:
    """a where predicate ``cond`` holds, else b; both registers of Emmy dtype ``dtype``."""
    typ = ptx_type(dtype)
    return Val(k.pure(f"selp.{mov_type(typ)}", typ, a, b, cond), dtype)


def apply(k: PTXKernel, name: str, args: Sequence[Val], dtype: str) -> Val:
    """name(args) at result ``dtype``: native where Emmy computes there (f16's simple ops,
    integer ops, f64), else in f32, converted once."""
    name, typ = canonical(name), ptx_type(dtype)
    at, fn = dtype, None
    if typ == "f16" and all(a.dtype == dtype for a in args):
        fn = F16_OPS.get(name)
    elif is_int(dtype) and all(is_int(a.dtype) for a in args) and name in INT_OPS:
        fn = partial(_int_op, name, typ)
    elif dtype == "f64":
        fn = F64_OPS.get(name)
    if fn is None:
        at, fn = "f32", F32_OPS.get(name) or fastmath.F32_FUNCS.get(name)
    if fn is None:
        raise NotImplementedError(f"{at} op {name}")
    arity = 2 if name in BINARY else 1
    if len(args) != arity:
        raise ValueError(f"{at} op {name} takes {arity} argument(s), got {len(args)}")
    out = fn(k, *(convert(k, a, at).reg for a in args))
    return convert(k, Val(out, at), dtype)


def assign(k: PTXKernel, a: Assign, args: Sequence[Val]) -> Val:
    """The value of Assign ``a`` over its argument values (Emmy's dtype rule; compares in f32)."""
    name = canonical(a.op.name)
    dtype = (
        a.dtype.name if a.dtype is not None else dtype_promote(a.op.name, [v.dtype for v in args])
    )
    if name in COMPARE:
        x, y = (convert(k, v, "f32").reg for v in args)
        return convert(k, Val(k.pure(f"setp.{COMPARE[name]}.f32", "pred", x, y), "bool"), dtype)
    if name == "where":
        cond = convert(k, args[0], "bool").reg
        x, y = (convert(k, v, dtype).reg for v in args[1:])
        return select(k, cond, x, y, dtype)
    if name in ("copy", "pad"):
        return convert(k, args[0], dtype)
    return apply(k, name, args, dtype)


# --- memory ---------------------------------------------------------------------------


def chunks(n: int, size: int) -> Iterator[tuple[int, int]]:
    """(first, width) of the vector accesses covering n consecutive elements of ``size`` bytes:
    .v4 / .v2 where they fit in 16 bytes."""
    i = 0
    while i < n:
        width = next((w for w in (4, 2) if n - i >= w and w * size <= 16), 1)
        yield i, width
        i += width


def load(
    k: PTXKernel, space: str, addr: str, off: int, typ: str, n: int, cached: bool
) -> list[str]:
    """``n`` consecutive elements of ``typ`` from [addr+off]. A ``cached`` scalar load reads
    memory nothing writes during the kernel, so an identical earlier one is reused."""
    regs: list[str] = []
    for i, width in chunks(n, BYTES[typ]):
        at = f"[{addr}+{off + i * BYTES[typ]}]"
        if width == 1 and cached:
            regs.append(k.pure(f"ld.{space}.{typ}", typ, at))
            continue
        out = [k.reg(typ) for _ in range(width)]
        dst = out[0] if width == 1 else f"{{{', '.join(out)}}}"
        k.emit(f"ld.{space}{'' if width == 1 else f'.v{width}'}.{typ} {dst}, {at}")
        regs += out
    return regs


def store(k: PTXKernel, space: str, addr: str, off: int, typ: str, values: list[str]) -> None:
    for i, width in chunks(len(values), BYTES[typ]):
        part = values[i : i + width]
        src = part[0] if width == 1 else f"{{{', '.join(part)}}}"
        at = f"[{addr}+{off + i * BYTES[typ]}]"
        k.emit(f"st.{space}{'' if width == 1 else f'.v{width}'}.{typ} {at}, {src}")


def shfl_xor(k: PTXKernel, x: str, typ: str, lane_mask: int, member_mask: str) -> str:
    """One butterfly step of ``x``. shfl moves 32 bits: 64-bit values go as two halves, 16-bit
    ones widen to 32 and back."""

    def shfl(v: str, t: str) -> str:
        out = k.reg(t)
        k.emit(f"shfl.sync.bfly.b32 {out}, {v}, {lane_mask}, 31, {member_mask}")
        return out

    if BYTES[typ] == 4:
        return shfl(x, typ)
    out = k.reg(typ)
    if BYTES[typ] == 8:
        lo, hi = k.reg("u32"), k.reg("u32")
        k.emit(f"mov.b64 {{{lo}, {hi}}}, {x}")
        k.emit(f"mov.b64 {out}, {{{shfl(lo, 'u32')}, {shfl(hi, 'u32')}}}")
    else:
        k.emit(f"cvt.u16.u32 {out}, {shfl(k.pure('cvt.u32.u16', 'u32', x), 'u32')}")
    return out
