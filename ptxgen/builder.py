"""The PTX builder: virtual registers, labels, value reuse across dominating blocks, rendering.

A kernel is one straight list of instruction lines. Control flow is structured, so a stack of
scopes tracks which results are available: ``pure`` reuses a result cached in the current
block or one that dominates it, and ``region`` opens a block that does not dominate what
follows (its results end with it). Registers are values (written once) or variables (written
in place: accumulators, combine states and loop counters, made by ``variable``). ``forget`` drops
every cached result that read a register about to be written in place.

Inside a runtime loop (``loop``), arithmetic whose sources all predate the innermost loop moves
in front of it, so it runs once instead of every trip. Only arithmetic: a hoisted load would keep
its value live across the whole loop, which measured slower; and only one level, since a value
live across a whole loop nest costs registers that ptxas spends better. Output is deterministic.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ptxgen.dtypes import REG_CLASS, mov_type

Key = tuple[str, ...]
Range = tuple[int, int]
_REG = re.compile(r"%[a-z]+\d+")  # a virtual register inside an operand, as in [%rd3+8]


@dataclass
class _Loop:
    at: int  # body index in front of the loop, where hoisted instructions go
    born: int  # registers numbered from here on belong to the loop
    home: int  # the scope that caches hoisted results


class PTXKernel:
    def __init__(
        self,
        name: str,
        params: list[str],
        max_threads: int,
        no_fusion: bool = False,
        no_cse: bool = False,
    ):
        self.name, self.params, self.max_threads = name, params, max_threads
        self.no_fusion = no_fusion  # no multiply and add fused into an fma (see ops.fma_operands)
        self.no_cse = no_cse  # no common subexpression elimination (see ``pure``)
        self.body: list[str] = []
        self.shared: list[str] = []  # .shared declarations inside the entry
        self.module_decls: list[str] = []  # module-scope lines (.extern .shared)
        self.ranges: dict[str, Range] = {}  # integer register -> [lo, hi] it is known to lie in
        self.variables: set[str] = set()  # registers written in place
        self.divmods: dict[str, tuple[str, str, int]] = {}  # reg -> ("div" | "mod", x, m)
        self._counts: dict[str, int] = {}
        self._labels = 0
        self._scopes: list[dict[Key, str]] = [{}]  # cached result per key; innermost last
        self._defs: dict[str, Key] = {}  # cached result -> its key
        self._readers: dict[str, list[Key]] = {}  # register -> keys that read it
        self._born: dict[str, int] = {}  # register -> its number, in creation order
        self._loops: list[_Loop] = []  # open runtime loops, outermost first
        self._pure: dict[str, str] = {}  # line emitted by ``pure`` -> its result register

    # --- registers and labels ---------------------------------------------------------
    def reg(self, typ: str, rng: Range | None = None) -> str:
        prefix = REG_CLASS[typ][0]
        n = self._counts.get(prefix, 0)
        self._counts[prefix] = n + 1
        r = f"{prefix}{n}"
        self._born[r] = len(self._born)
        if rng is not None:
            self.ranges[r] = rng
        return r

    def variable(self, typ: str, init: str) -> str:
        """A fresh register written in place later, starting at ``init``."""
        r = self.reg(typ)
        self.variables.add(r)
        self.emit(f"mov.{mov_type(typ)} {r}, {init}")
        return r

    def label(self, hint: str) -> str:
        self._labels += 1
        return f"$L_{hint}_{self._labels}"

    def place(self, label: str) -> None:
        self.body.append(f"{label}:")

    def emit(self, instr: str, pred: str | None = None) -> None:
        self.body.append(f"\t{f'@{pred} ' if pred else ''}{instr};")

    # --- value reuse ------------------------------------------------------------------
    def _cached(self, key: Key) -> str | None:
        """The result cached under ``key`` in this block or a dominating one."""
        for scope in reversed(self._scopes):
            if key in scope:
                return scope[key]
        return None

    def _remember(self, key: Key, out: str, home: int = -1) -> None:
        """Cache ``out`` under ``key`` in scope ``home``. Every register in ``key`` must hold its
        value until the result is forgotten."""
        self._scopes[home][key] = out
        self._defs[out] = key
        for src in {r for part in key for r in _REG.findall(part)}:
            self._readers.setdefault(src, []).append(key)

    def pure(self, op: str, typ: str, *srcs: str, rng: Range | None = None) -> str:
        """``op out, *srcs`` for a side-effect-free op, reusing an identical earlier one (never
        with ``no_cse``)."""
        key = (op, *srcs)

        # Hacky way of disabling CSE in case of the --no_cse flag in the compiler
        if self.no_cse:
            key += (f"#{len(self._born)}",)  # unique: no earlier result matches

        # Do not emit PTX operation if value exists in scope
        if (out := self._cached(key)) is not None:
            return out

        # Create new register storing result of this operation
        out = self.reg(typ, rng)
        line = f"\t{op} {out}, {', '.join(srcs)};"
        self._pure[line] = out

        loop = self._hoist_to(op, srcs)
        if loop is None:
            # Append the line to PTX if you can't hoist op outside the loop
            self.body.append(line)
            self._remember(key, out)
        else:
            # Append the line to before loop begin if you can hoist it outside
            self.body.insert(loop.at, line)
            loop.at += 1
            self._born[out] = loop.born - 1  # it now predates the loop
            self._remember(key, out, loop.home)

        return out

    def _hoist_to(self, op: str, srcs: tuple[str, ...]) -> _Loop | None:
        """Return the innermost open loop if op is hoistable and src registers
        are not created or edited within the loop."""
        if not self._loops or op.startswith("ld."):
            return None
        loop = self._loops[-1]
        regs = [r for s in srcs for r in _REG.findall(s)]
        if any(r in self.variables or self._born[r] >= loop.born for r in regs):
            return None
        return loop

    def definition(self, reg: str) -> Key | None:
        """The key of the cached result ``reg``, while it is still reusable here."""
        return self._defs.get(reg)

    def forget(self, reg: str) -> None:
        """``reg`` was just written in place: no cached result that read it stays reusable."""
        for key in self._readers.pop(reg, []):
            for scope in self._scopes:
                if (out := scope.pop(key, None)) is not None:
                    self._defs.pop(out, None)

    def drop_dead(self) -> None:
        """Remove ``pure`` results nothing reads, such as a multiply fused into an fma."""
        while True:
            reads = Counter(r for line in self.body for r in _REG.findall(line))
            live = [ln for ln in self.body if ln not in self._pure or reads[self._pure[ln]] > 1]
            if len(live) == len(self.body):
                return
            self.body = live

    @contextmanager
    def region(self) -> Iterator[None]:
        """A block dominated by the current one that does not dominate what follows."""
        self._scopes.append({})
        try:
            yield
        finally:
            for out in self._scopes.pop().values():
                self._defs.pop(out, None)

    @contextmanager
    def loop(self, entered: bool) -> Iterator[None]:
        """The body of a runtime loop. ``entered``: no thread skips the loop, so the code in
        front of it dominates what follows the loop and hoisted results stay reusable there."""
        home = len(self._scopes) - 1 if entered else len(self._scopes)
        self._loops.append(_Loop(len(self.body), len(self._born), home))
        try:
            with self.region():
                yield
        finally:
            self._loops.pop()

    @contextmanager
    def unless(self, pred: str) -> Iterator[None]:
        """A block skipped when ``pred`` holds (``pred`` may be negated, as ``!%p1``)."""
        skip = self.label("skip")
        self.emit(f"bra {skip}", pred=pred)
        with self.region():
            yield
        self.place(skip)

    # --- trials -----------------------------------------------------------------------
    def snapshot(self) -> dict[str, object]:
        """All builder state, to ``restore`` after emitting code that is then dropped."""
        return copy.deepcopy(self.__dict__)

    def restore(self, state: dict[str, object]) -> None:
        """Back to ``state``, in place: callers keep their reference to this Kernel."""
        self.__dict__ = state

    # --- rendering --------------------------------------------------------------------
    def render(self) -> str:
        params = ",\n".join(f"\t.param .u64 {self.name}_param_{p}" for p in self.params)
        decls = {}
        for prefix, typ in REG_CLASS.values():
            if prefix in self._counts:
                decls[prefix] = f"\t.reg .{typ} {prefix}<{self._counts[prefix]}>;"
        lines = [
            "//\n// Generated by ptxgen\n//\n",
            ".version 8.0\n.target sm_89\n.address_size 64\n",
            *self.module_decls,
            f".visible .entry {self.name}(\n{params}\n)\n.maxntid {self.max_threads}, 1, 1\n{{",
            *decls.values(),
            *self.shared,
            "",
            *self.body,
            "\tret;\n}\n",
        ]
        return "\n".join(lines)
