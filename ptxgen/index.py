"""Integer indices as affine forms over registers, with value ranges.

Every index stays a ``Lin`` (``sum(coeff * reg) + const``) for as long as possible, so constant
parts never reach a register: ``x[i * 4 + 3]`` is ``[%rd + 12]``, and copy ``k`` of an unrolled
loop shares its base register with every other copy. Each index register has a known range
(``Kernel.ranges``), which turns ``/`` and ``%`` by a constant into nothing, a shift, or a mask:
``(ctaid * 128 + tid) / 128`` is ``ctaid`` when ``tid < 128``. Compares the ranges decide are
Python bools. Indices are never negative (Emmy guarantees it), so unsigned shifts are exact.
"""

from __future__ import annotations

from ptxgen.builder import PTXKernel, Range

_CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}


class Lin:
    """An affine integer form: sum(coeff * register) + const. Immutable."""

    __slots__ = ("const", "terms")

    def __init__(self, terms: dict[str, int] | None = None, const: int = 0):
        self.terms = {r: c for r, c in sorted((terms or {}).items()) if c}
        self.const = int(const)

    @classmethod
    def of(cls, reg: str) -> Lin:
        return cls({reg: 1})

    @property
    def is_const(self) -> bool:
        return not self.terms

    def var_part(self) -> Lin:
        return Lin(self.terms)

    def __add__(self, other: Lin | int) -> Lin:
        other = other if isinstance(other, Lin) else Lin(const=other)
        terms = dict(self.terms)
        for r, c in other.terms.items():
            terms[r] = terms.get(r, 0) + c
        return Lin(terms, self.const + other.const)

    def __sub__(self, other: Lin | int) -> Lin:
        return self + (other * -1 if isinstance(other, Lin) else -other)

    def __mul__(self, k: int) -> Lin:
        return Lin({r: c * k for r, c in self.terms.items()}, self.const * k)

    def range(self, ranges: dict[str, Range]) -> Range | None:
        lo = hi = self.const
        for r, c in self.terms.items():
            if r not in ranges:
                return None
            a, b = ranges[r][0] * c, ranges[r][1] * c
            lo, hi = lo + min(a, b), hi + max(a, b)
        return lo, hi


def operand(k: PTXKernel, x: Lin) -> str:
    """An immediate or a register holding x."""
    return str(x.const) if x.is_const else reg_of(k, x)


def reg_of(k: PTXKernel, x: Lin) -> str:
    """A register holding x."""
    x = _rejoin(k, x)
    if x.is_const:
        return k.pure("mov.u32", "u32", str(x.const), rng=(x.const, x.const))
    if x.const:
        return k.pure(
            "add.s32", "u32", reg_of(k, x.var_part()), str(x.const), rng=x.range(k.ranges)
        )
    (first, c), *rest = x.terms.items()
    acc = first if c == 1 else _mul_imm(k, first, c)
    for reg, c in rest:
        rng = Lin({acc: 1, reg: c}).range(k.ranges)
        if c in (1, -1):
            acc = k.pure("add.s32" if c == 1 else "sub.s32", "u32", acc, reg, rng=rng)
        else:
            acc = k.pure("mad.lo.s32", "u32", reg, str(c), acc, rng=rng)
    return acc


def _mul_imm(k: PTXKernel, reg: str, c: int) -> str:
    rng = Lin({reg: c}).range(k.ranges)
    if c > 0 and not c & (c - 1):
        return k.pure("shl.b32", "u32", reg, str(c.bit_length() - 1), rng=rng)
    return k.pure("mul.lo.s32", "u32", reg, str(c), rng=rng)


def _rejoin(k: PTXKernel, x: Lin) -> Lin:
    """Fold c*m*(X / m) + c*(X % m) back into c*X, as a split flat index re-joins."""
    for q, cq in x.terms.items():
        if (fact := k.divmods.get(q)) is None or fact[0] != "div":
            continue
        _, src, m = fact
        for r, cr in x.terms.items():
            if k.divmods.get(r) == ("mod", src, m) and cq == cr * m:
                return _rejoin(k, x - Lin({q: cq, r: cr}) + Lin.of(src) * cr)
    return x


def mul(k: PTXKernel, a: Lin, b: Lin) -> Lin:
    if a.is_const or b.is_const:
        return b * a.const if a.is_const else a * b.const
    ra, rb = a.range(k.ranges), b.range(k.ranges)
    rng = None
    if ra and rb:
        ends = [x * y for x in ra for y in rb]
        rng = (min(ends), max(ends))
    return Lin.of(k.pure("mul.lo.s32", "u32", reg_of(k, a), reg_of(k, b), rng=rng))


def div(k: PTXKernel, a: Lin, m: Lin | int) -> Lin:
    """a / m for m > 0."""
    return _divmod(k, a, m, "div")


def mod(k: PTXKernel, a: Lin, m: Lin | int) -> Lin:
    """a % m for m > 0."""
    return _divmod(k, a, m, "mod")


def _divmod(k: PTXKernel, a: Lin, m: Lin | int, kind: str) -> Lin:
    """a / m or a % m (``kind``) for m > 0, kept affine where the ranges allow.

    ``+``, ``-`` and ``*`` by a constant map a ``Lin`` to a ``Lin``. ``/`` and ``%`` do not:
    ``(x + y) / m`` is not ``x / m + y / m``, since the low parts can carry into the quotient. A
    real division gives an opaque register, which hides the structure from later folding. So:

    - Terms whose coefficient m divides go straight to the quotient (``a = m*q + rest``). Only
      ``rest`` needs a division.
    - If ``rest`` stays in ``[0, m)``, it never carries: div is ``q``, mod is ``rest``, no
      instruction at all.
    - Otherwise emit ``shr``/``and`` for a power of two, else ``div``/``rem``, and record the
      fact in ``k.divmods`` so ``_rejoin`` can fold ``m*(X / m) + X % m`` back into ``X``.

    The split needs ``rest >= 0``: the PTX ops are unsigned, so a possibly negative ``rest``
    falls back to dividing all of ``a``. A divisor known only at run time always gets a register.
    """
    is_div = kind == "div"

    # Handle a divisor only known at run time (a ragged raster group)
    if isinstance(m, Lin) and not m.is_const:
        op = "div.u32" if is_div else "rem.u32"
        return Lin.of(k.pure(op, "u32", operand(k, a), reg_of(k, m)))
    m = m.const if isinstance(m, Lin) else m
    if m <= 0:
        raise ValueError(f"index {kind} by {m}")

    # Handle a constant dividend
    if a.is_const:
        return Lin(const=a.const // m if is_div else a.const % m)

    # Split a = m*q + rest, q collecting every term divisible by m: only rest needs a division
    q = Lin({r: c // m for r, c in a.terms.items() if c % m == 0}, a.const // m)
    rest = Lin({r: c for r, c in a.terms.items() if c % m}, a.const % m)
    rng = rest.range(k.ranges)
    if rng is None or rng[0] < 0:  # the ops are unsigned, so divide all of a
        q, rest, rng = Lin(), a, a.range(k.ranges)

    # Handle a rest that never carries into the quotient
    if rng is not None and rng[1] < m:
        return q if is_div else rest

    # Emit the division: a shift or a mask for a power of two
    x = reg_of(k, rest)
    pow2 = not m & (m - 1)
    if is_div:
        op, imm = ("shr.u32", m.bit_length() - 1) if pow2 else ("div.u32", m)
        out_rng = (rng[0] // m, rng[1] // m) if rng else None
    else:
        op, imm = ("and.b32", m - 1) if pow2 else ("rem.u32", m)
        out_rng = (0, min(m - 1, rng[1])) if rng else (0, m - 1)
    r = k.pure(op, "u32", x, str(imm), rng=out_rng)
    k.divmods[r] = (kind, x, m)
    return q + Lin.of(r) if is_div else Lin.of(r)


def minimum(k: PTXKernel, a: Lin, b: Lin) -> Lin:
    if a.is_const and b.is_const:
        return Lin(const=min(a.const, b.const))
    return Lin.of(k.pure("min.s32", "u32", reg_of(k, a), operand(k, b)))


def compare(k: PTXKernel, op: str, a: Lin, b: Lin) -> str | bool:
    """A predicate register for ``a op b``, or a bool when the ranges decide it."""
    d = a - b
    rng = d.range(k.ranges)
    if rng is not None:
        lo, hi = rng
        holds = {"<": hi < 0, "<=": hi <= 0, ">": lo > 0, ">=": lo >= 0, "==": lo == hi == 0}
        fails = {"<": lo >= 0, "<=": lo > 0, ">": hi <= 0, ">=": hi < 0, "==": lo > 0 or hi < 0}
        holds["!="], fails["!="] = fails["=="], holds["=="]
        if holds[op] or fails[op]:
            return holds[op]
    return k.pure(f"setp.{_CMP[op]}.s32", "pred", reg_of(k, d.var_part()), str(-d.const))


def select(k: PTXKernel, pred: str, a: Lin, b: Lin) -> Lin:
    ra, rb = a.range(k.ranges), b.range(k.ranges)
    rng = (min(ra[0], rb[0]), max(ra[1], rb[1])) if ra and rb else None
    return Lin.of(k.pure("selp.b32", "u32", operand(k, a), operand(k, b), pred, rng=rng))
