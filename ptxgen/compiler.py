"""Kernel IR -> PTX. One ``Emitter`` walks the IR; each statement kind has an ``emit_<Node>``.

Names map to typed values: an index is a ``Lin`` (an affine form, see ``index``), data is a
``Val`` (a register and its Emmy dtype; a bool is a predicate register). Data follows Emmy's
dtype rule (see ``ops.apply``): an op computes natively where Emmy does, otherwise in f32 with
one conversion back.

Registers are values (written once) or variables (accumulators, Init state, combine states),
which are written in place. Every side-effect-free result is cached and reused while the block
that made it dominates (``Kernel.reuse``); each in-place write forgets the results that read the
old value, and a runtime loop forgets them before its body, since its back edge brings the
variable round again. Names defined in a block end with it.

Layout: the core (walking the body, block scope, names, expressions) serves every ``emit_*``.
Every other helper is a static method of one section (launch, memory, data, reduce, control)
and calls no other helper; only the control section's loop shapes walk the body again.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from math import prod

from emmy.compiler.backend.cuda.dtype import canonical_from_cuda_name
from emmy.compiler.ir.expr import CastExpr, FuncCallExpr
from emmy.compiler.ir.kernel import (
    Accum,
    Assign,
    BinaryExpr,
    Cond,
    Expr,
    KernelOp,
    Literal,
    Load,
    Loop,
    Stmt,
    StridedLoop,
    Sync,
    TernaryExpr,
    Tile,
    TreeHalve,
    Var,
    WarpShuffle,
    Write,
)
from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait, Smem, pack_smem
from emmy.compiler.ir.stmt import Init, Let
from emmy.compiler.ir.stmt.base import select_to_ternary
from emmy.compiler.ir.stmt.leaves import Select
from emmy.gpu import STATIC_SMEM_CAP

from ptxgen import index, ops
from ptxgen.builder import PTXKernel, Range
from ptxgen.dtypes import BYTES, FLOATS, is_int, lit, mem_type, mov_type, ptx_type
from ptxgen.index import Lin
from ptxgen.memory import Global, Memory, Shared
from ptxgen.ops import Val

# Copies of a loop's trips (see emit_Loop), each measured on the runner's RTX 5090:
UNROLL_SIZE = 8192  # PTX instructions a loop may unroll to: 6.2K ran fine, 16K+ much slower
NESTED_COPIES = 2  # unroll factor of a loop around loops: the best single choice
REREAD_COPIES = 8  # unroll factor of a loop whose loads _rereads re-reads: the best single choice
REG_FILE, REG_CAP = 65536, 255  # registers per block, and per thread
BARRIERS = (Sync, WarpShuffle, TreeHalve, CpAsyncWait)  # every thread must reach these alike
F32_SYMBOL = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}


@dataclass(frozen=True)
class Buffer:
    """A kernel parameter's buffer, as meta.json describes it."""

    shape: tuple[int, ...]
    dtype: str
    is_output: bool


Value = Lin | Val | bool


def compile_module(
    op: KernelOp,
    buffers: Mapping[str, Buffer],
    arg_order: Sequence[str],
    block_threads: int,
    *,
    no_unroll: bool = False,
    no_fusion: bool = False,
    no_cse: bool = False,
) -> str:
    """The PTX module for ``op``, launched with ``block_threads`` threads per block."""
    return Emitter(op, buffers, arg_order, block_threads, no_unroll, no_fusion, no_cse).run()


class Emitter:
    def __init__(
        self,
        op: KernelOp,
        buffers: Mapping[str, Buffer],
        arg_order: Sequence[str],
        block: int,
        no_unroll: bool,
        no_fusion: bool,
        no_cse: bool,
    ):
        self.op, self.block, self.buffers = op, block, dict(buffers)
        self.k = PTXKernel(op.name, list(arg_order), block, no_fusion, no_cse)
        self.env: dict[str, Value] = {}
        self.mem: dict[str, Memory] = {}  # parameter and shared buffers by name
        self.hold = min(REG_CAP, REG_FILE // block) // 2  # loaded values a loop may keep
        self.reread: set[int] = set()  # ids of Loops whose loads _rereads re-reads every trip
        self.measuring = False  # inside a loop's trial unroll (see emit_Loop)
        self.no_unroll = no_unroll  # every loop a runtime loop, one copy per trip

    # --- core: walking the body ---------------------------------------------------------

    def run(self) -> str:
        k = self.k

        # Add registers that point to parameter space
        for name in k.params:
            if name not in self.buffers:
                raise ValueError(f"parameter {name!r} has no buffer")
            raw = k.reg("u64")
            k.emit(f"ld.param.u64 {raw}, [{k.name}_param_{name}]")
            buf = self.buffers[name]
            ptr = k.pure("cvta.to.global.u64", "u64", raw)
            self.mem[name] = Global(buf.shape, buf.dtype, ptr, buf.is_output)

        # Add tid and ctaid regs
        self.tid = k.pure("mov.u32", "u32", "%tid.x", rng=(0, self.block - 1))
        self.ctaid = k.pure("mov.u32", "u32", "%ctaid.x")

        # If the Kernel IR contains reduction ops, add lane and warp helpers
        if any(isinstance(st, WarpShuffle | TreeHalve) for st in self._walk(self.op.body)):
            lane = k.pure("and.b32", "u32", self.tid, "31", rng=(0, min(31, self.block - 1)))
            warp = k.pure("shr.u32", "u32", self.tid, "5", rng=(0, (self.block - 1) >> 5))
            self.env |= {"lane": Lin.of(lane), "warp": Lin.of(warp)}

        # Declare shared memory buffers if they are needed
        self.mem |= self._declare_shared(k, list(self.op.smem_buffers.values()), self.buffers)

        # Walk the Kernel IR, emitting PTX operations as you go
        self.body(self.op.body)

        # Drop unused PTX registers (these come from operator fusion)
        k.drop_dead()
        return k.render()

    def body(self, stmts: Sequence[Stmt]) -> None:
        for st in stmts:
            fn = getattr(self, f"emit_{type(st).__name__}", None)
            if fn is None:
                raise NotImplementedError(f"no PTX emitter for Kernel IR node {type(st).__name__}")
            fn(st)

    @contextmanager
    def _block(self) -> Iterator[None]:
        """Names first defined inside end with it (rebinding an outer name persists)."""
        before = set(self.env)
        yield
        for name in set(self.env) - before:
            del self.env[name]

    @staticmethod
    def _where(k: PTXKernel, p: str | bool) -> AbstractContextManager[None]:
        """A block that runs only where predicate p holds (always when p is True)."""
        return nullcontext() if p is True else k.unless(f"!{p}")

    @staticmethod
    def _walk(body: Sequence[Stmt]) -> Iterator[Stmt]:
        """Every statement in body, nested ones included."""
        for st in body:
            yield st
            for inner in st.nested():
                yield from Emitter._walk(inner)

    # --- core: names --------------------------------------------------------------------

    def _lookup(self, name: str) -> Value:
        if name not in self.env:
            raise KeyError(f"{name!r} is used before it is defined")
        return self.env[name]

    def _data(self, name: str) -> Val:
        """The value of ``name`` as data; an index name becomes an i32."""
        return self._as_val(self._lookup(name))

    def _define(self, name: str, v: Val) -> None:
        """Bind name to v; a name never aliases a variable, so a variable is copied first."""
        if v.reg in self.k.variables:
            typ = ptx_type(v.dtype)
            copy = self.k.reg(typ)
            self.k.emit(f"mov.{mov_type(typ)} {copy}, {v.reg}")
            v = Val(copy, v.dtype)
        self.env[name] = v

    def _init(self, name: str, identity: float | None, dtype: str) -> None:
        """Bind name to a new variable of ``dtype`` that starts at ``identity``."""
        if identity is None:
            raise ValueError(f"{name!r} has no identity to start from")
        typ = ptx_type(dtype)
        self.env[name] = Val(self.k.variable(typ, lit(identity, typ)), dtype)

    def _variable(self, name: str, dtype: str | None = None) -> Val:
        """Make name's value a variable (at ``dtype``), so it can be written in place."""
        v = self._data(name)
        if dtype is not None and v.dtype != dtype:
            v = ops.convert(self.k, v, dtype)
        if v.reg not in self.k.variables:
            v = Val(self.k.variable(ptx_type(v.dtype), v.reg), v.dtype)
            self.env[name] = v
        return v

    def _write(self, name: str, v: Val) -> None:
        """Write v into variable ``name`` in place."""
        var = self._variable(name)
        src = ops.convert(self.k, v, var.dtype).reg
        if src != var.reg:
            self.k.emit(f"mov.{mov_type(ptx_type(var.dtype))} {var.reg}, {src}")
            self.k.forget(var.reg)

    # --- launch -------------------------------------------------------------------------

    def emit_Tile(self, t: Tile) -> None:
        if not t.is_static_grid or t.aux_threads:
            raise NotImplementedError("Tile: only static grids without an aux band")
        n, k = t.n_elements, self.k
        if n > 2**31 - 1:
            raise NotImplementedError("Tile: more than 2^31 - 1 cells")

        k.ranges[self.ctaid] = (0, -(-n // self.block) - 1)
        gid = Lin.of(self.ctaid) * self.block + Lin.of(self.tid)
        past = index.compare(k, ">=", gid, Lin(const=n))
        if past is True:
            return

        with k.unless(past) if past else nullcontext(), self._block():
            self.env |= self._decode(k, t, gid, self._raster(t))
            self.body(t.body)

    def emit_Sync(self, s: Sync) -> None:
        if s.warp:
            if s.barrier_id:
                raise ValueError("Sync(warp=True) cannot carry a named barrier_id")
            self.k.emit("bar.warp.sync -1")
        elif s.barrier_id == 0:
            self.k.emit("bar.sync 0")
        elif s.count is None:
            raise ValueError(f"Sync(barrier_id={s.barrier_id}) requires count")
        else:
            self.k.emit(f"bar.sync {s.barrier_id}, {s.count}")

    @staticmethod
    def _raster(t: Tile) -> tuple[int, int, int] | None:
        """(grouped axis, other axis, group) when the Tile remaps its block axes in groups."""
        names = [a.name for a in t.axes]
        if not (t.raster_group and t.raster_axes and all(n in names for n in t.raster_axes)):
            return None
        mi, ni = (names.index(n) for n in t.raster_axes)
        if ni != mi + 1:
            return None
        gi, oi = (mi, ni) if t.raster_orient == "m" else (ni, mi)
        g = min(t.raster_group, t.extents[gi])
        return (gi, oi, g) if t.extents[gi] > 1 and g > 1 else None

    @staticmethod
    def _decode(
        k: PTXKernel, t: Tile, gid: Lin, raster: tuple[int, int, int] | None
    ) -> dict[str, Lin]:
        """Each axis from the global id: axis_i = gid / stride_i % extent_i, row-major (no % on
        the outermost axis). A ``raster`` pair takes grouped block order instead: with the
        grouped axis a (extent Ea, group G) and the other axis b (Eb), a = grp*G + (sub %
        (G*Eb)) % gsz and b = (sub % (G*Eb)) / gsz, where gsz is G except in a ragged last
        group."""
        extents = list(t.extents)
        strides = [prod(extents[i + 1 :]) for i in range(len(extents))]
        axes: dict[str, Lin] = {}
        for i, (axis, e) in enumerate(zip(t.axes, extents, strict=True)):
            if raster is not None and i in raster[:2]:
                continue
            v = Lin() if e == 1 else index.div(k, gid, strides[i])
            axes[axis.name] = index.mod(k, v, e) if i and e > 1 else v
        if raster is None:
            return axes
        gi, oi, g = raster
        ea, eb = extents[gi], extents[oi]
        sub = index.div(k, gid, strides[max(gi, oi)])
        if min(gi, oi):  # the pair is not outermost: drop the outer axes
            sub = index.mod(k, sub, ea * eb)
        if eb == 1:
            a, b = sub, Lin()
        else:
            grp, r = index.div(k, sub, g * eb), index.mod(k, sub, g * eb)
            gsz = Lin(const=g)
            if ea % g:
                gsz = index.minimum(k, Lin(const=ea) - grp * g, gsz)
            a = grp * g + index.mod(k, r, gsz)
            b = index.div(k, r, gsz)
        axes[t.axes[gi].name], axes[t.axes[oi].name] = a, b
        return axes

    # --- memory -------------------------------------------------------------------------

    def emit_Smem(self, s: Smem) -> None:
        pass  # declared up front, so a Smem in a loop body is still one buffer

    def emit_Load(self, s: Load) -> None:
        mem = self.mem[s.input]
        dtype = s.dtype.name if s.dtype is not None else mem.dtype
        flat = self._flat(s.input, mem.shape, [self._int(e) for e in s.index])
        self.env.update(zip(s.names, mem.load(self.k, flat, dtype, len(s.names)), strict=True))

    def emit_Write(self, s: Write) -> None:
        if s.swizzle != "NONE":
            raise NotImplementedError(f"Write: swizzle {s.swizzle}")
        k, mem = self.k, self.mem[s.output]
        flat = self._flat(s.output, mem.shape, [self._int(e) for e in s.index])
        values = [self._data(v) for v in s.values]
        if not s.atomic:  # values convert to the buffer's dtype as they store
            mem.store(k, flat, mem.dtype, values)
            return
        # split-K finalize: every block adds its partial into the zeroed output
        if isinstance(mem, Shared):
            raise NotImplementedError("Write: atomic to shared memory")
        if mem.dtype not in ("f32", "f64", "f16", "i32", "u32", "u64"):
            raise NotImplementedError(f"Write: atomic add of {mem.dtype}")
        addr, off, _ = mem.address(k, flat, mem.dtype)
        form = "noftz.f16" if mem.dtype == "f16" else ptx_type(mem.dtype)
        for i, v in enumerate(values):
            at = off + i * BYTES[mem_type(mem.dtype)]
            k.emit(f"red.global.add.{form} [{addr}+{at}], {ops.convert(k, v, mem.dtype).reg}")

    def emit_CpAsyncCopy(self, s: CpAsyncCopy) -> None:
        if s.swizzle != "NONE":
            raise NotImplementedError(f"CpAsyncCopy: swizzle {s.swizzle}")
        if getattr(s, "valid", None) is not None:
            raise NotImplementedError("CpAsyncCopy: zero-fill (valid) form")
        if s.nbytes not in (4, 8, 16):
            raise ValueError(f"CpAsyncCopy: nbytes {s.nbytes} is not 4, 8 or 16")
        k, dst, src = self.k, self.mem[s.smem], self.mem[s.src]
        idx = s.smem_index
        if s.lane_index is not None:  # this lane's stripe of the tile
            idx = tuple(BinaryExpr("+", b, ln) for b, ln in zip(idx, s.lane_index, strict=True))
        flat = self._flat(s.smem, dst.shape, [self._int(e) for e in idx])
        d_addr, d_off, _ = dst.address(k, flat, dst.dtype)
        flat = self._flat(s.src, src.shape, [self._int(e) for e in s.src_index])
        s_addr, s_off, _ = src.address(k, flat, src.dtype)
        cache = "cg" if s.nbytes == 16 else "ca"  # .cg (L2 only) exists for 16 bytes alone
        k.emit(f"cp.async.{cache}.shared.global [{d_addr}+{d_off}], [{s_addr}+{s_off}], {s.nbytes}")

    def emit_CpAsyncCommit(self, _s: CpAsyncCommit) -> None:
        self.k.emit("cp.async.commit_group")

    def emit_CpAsyncWait(self, s: CpAsyncWait) -> None:
        self.k.emit(f"cp.async.wait_group {s.group}")

    @staticmethod
    def _declare_shared(
        k: PTXKernel, decls: list[Smem], taken: Mapping[str, Buffer]
    ) -> dict[str, Shared]:
        """Every Smem of the kernel: static .shared arrays, or offsets into one dynamic pool
        once they pass the static cap (Emmy's rule; meta.json's smem_bytes sizes the pool)."""
        offsets, total = pack_smem(decls)
        pool = total > STATIC_SMEM_CAP
        if pool:
            align = max(16, *(int(s.align or 0) for s in decls))
            k.module_decls.append(f".extern .shared .align {align} .b8 smem_pool[];\n")
        shared: dict[str, Shared] = {}
        for s in decls:
            dtype = canonical_from_cuda_name(s.dtype) or s.dtype
            if dtype not in ("f32", "f64", "f16", "bf16", "i32", "u32", "i64", "u64"):
                raise NotImplementedError(f"Smem: dtype {s.dtype}")
            if s.name in taken:
                raise ValueError(f"Smem {s.name!r}: name already declared")
            shape, size = tuple(int(e) for e in s.extents), BYTES[mem_type(dtype)]
            if pool:
                base = k.pure("mov.u32", "u32", "smem_pool")
                shared[s.name] = Shared(shape, dtype, base, offsets[s.name])
                continue
            sym = f"smem_{s.name}"
            k.shared.append(
                f"\t.shared .align {int(s.align or 0) or size} .b8 {sym}[{prod(shape) * size}];"
            )
            shared[s.name] = Shared(shape, dtype, k.pure("mov.u32", "u32", sym), 0)
        return shared

    @staticmethod
    def _flat(buf: str, shape: tuple[int, ...], parts: list[Lin]) -> Lin:
        """Row-major element offset of buf[parts]; a single index is already flat."""
        if len(parts) <= 1:
            return parts[0] if parts else Lin()
        if len(parts) != len(shape):
            raise NotImplementedError(f"{buf}: {len(parts)}-d index into a {len(shape)}-d buffer")
        flat, stride = Lin(), 1
        for i, extent in zip(reversed(parts), reversed(shape), strict=True):
            flat, stride = flat + i * stride, stride * extent
        return flat

    # --- data ---------------------------------------------------------------------------

    def emit_Assign(self, s: Assign) -> None:
        self._define(s.name, ops.assign(self.k, s, [self._data(a) for a in s.args]))

    def emit_Let(self, s: Let) -> None:
        # untyped: an index, or data under a new name
        v = self.expr(s.value) if s.dtype is None else self._value(s.value, s.dtype.name)
        if isinstance(v, Val):
            self._define(s.name, v)
        else:
            self.env[s.name] = v

    def emit_Init(self, s: Init) -> None:
        self._init(s.name, s.identity, s.dtype.name)

    def emit_Select(self, s: Select) -> None:
        self._define(s.name, self._value(select_to_ternary(s), "f32"))

    def emit_Accum(self, s: Accum) -> None:
        k, name = self.k, ops.canonical(s.op.name)
        dtype = s.dtype.name if s.dtype is not None else "f32"
        if name not in ops.ACCUM:
            raise NotImplementedError(f"Accum: op {s.op.name} is not a binary op")
        if s.name not in self.env:  # no seeding loop: start from the identity here
            self._init(s.name, s.op.identity, dtype)
        acc = self._variable(s.name, dtype).reg
        value = ops.convert(k, self._data(s.value), dtype).reg
        base = acc  # or a rescaled one
        if s.base is not None:
            base = ops.convert(k, self._data(s.base), dtype).reg
        typ = ptx_type(dtype)
        if name == "add" and (fma := ops.fma_operands(k, typ, value, base)) is not None:
            k.emit(f"fma.rn.{typ} {acc}, {', '.join(fma)}")
        elif typ == "bf16" or (typ == "f16" and name == "divide"):  # no native form: in f32
            f32 = [Val(ops.cvt(k, r, typ, "f32"), "f32") for r in (base, value)]
            k.emit(f"cvt.rn.{typ}.f32 {acc}, {ops.apply(k, name, f32, 'f32').reg}")
        else:
            k.emit(f"{ops.ACCUM[name][typ not in FLOATS]}.{typ} {acc}, {base}, {value}")
        k.forget(acc)

    # --- reduce: cross-thread combines --------------------------------------------------
    # A combine program's Assigns to the states write them in place; the rest define names.

    def emit_WarpShuffle(self, s: WarpShuffle) -> None:
        if getattr(s, "prefix", False):
            raise NotImplementedError("WarpShuffle: prefix (shfl.up) form")
        if not 2 <= s.length <= 32 or s.length & (s.length - 1):
            raise ValueError(f"WarpShuffle: length {s.length} is not a power of two in [2, 32]")
        if len(s.state) != len(s.state_b):
            raise ValueError("WarpShuffle: state and state_b differ in length")
        k, dtype = self.k, s.dtype.name
        for st in s.state:
            self._variable(st, dtype)
        mask = k.reg("u32")
        k.emit(f"activemask.b32 {mask}")
        step = s.length // 2
        while step:
            for st, sb in zip(s.state, s.state_b, strict=True):
                other = ops.shfl_xor(k, self._data(st).reg, ptx_type(dtype), step, mask)
                self.env[sb] = Val(other, dtype)
            for a in s.combine_states:
                v = ops.assign(k, a, [self._data(x) for x in a.args])
                (self._write if a.name in s.state else self._define)(a.name, v)
            step //= 2

    def emit_TreeHalve(self, s: TreeHalve) -> None:
        """Halving tree over shared slots: slot t folds in slot t + half under a barrier, until
        slot 0 holds the result, which every thread then reads. Segmented (``inner``): slot t is
        buf[t * scale + inner], each level halving the segment index."""
        if s.length < 2 or s.length & (s.length - 1):
            raise ValueError(f"TreeHalve: length {s.length} is not a power of two >= 2")
        if not len(s.bufs) == len(s.state) == len(s.state_b):
            raise ValueError("TreeHalve: bufs, state and state_b differ in length")
        k = self.k
        trees = [
            (self.mem[b], st, sb) for b, st, sb in zip(s.bufs, s.state, s.state_b, strict=True)
        ]
        for mem, st, _ in trees:
            self._variable(st, mem.dtype)
        t = self._int(Var(s.tid_var))
        scale, root = (1, Lin()) if s.inner is None else (s.inner[1], self._int(Var(s.inner[0])))
        sync = Sync(barrier_id=s.barrier_id, count=s.barrier_count)
        for half in (s.length >> i for i in range(1, s.length.bit_length())):
            # threads t < half fold slot mine + stride into slot mine
            mine, stride = t * scale + root, half * scale
            active = index.compare(k, "<", t, Lin(const=half))  # never False: t starts at 0
            with self._where(k, active), self._block():
                for mem, st, sb in trees:
                    self._write(st, mem.load(k, mine, mem.dtype)[0])
                    self.env[sb] = mem.load(k, mine + stride, mem.dtype)[0]
                for a in s.combine_states:
                    v = ops.assign(k, a, [self._data(x) for x in a.args])
                    (self._write if a.name in s.state else self._define)(a.name, v)
                for mem, st, _ in trees:
                    mem.store(k, mine, mem.dtype, [self._data(st)])
            self.emit_Sync(sync)
        for mem, st, _ in trees:  # every thread takes the result
            self._write(st, mem.load(k, root, mem.dtype)[0])

    # --- control ------------------------------------------------------------------------

    def emit_Cond(self, s: Cond) -> None:
        p = self._pred(s.cond)
        if isinstance(p, bool):
            with self._block():
                self.body(s.body if p else s.else_body)
            return
        k = self.k
        if not s.else_body:
            with self._where(k, p), self._block():
                self.body(s.body)
            return
        other, end = k.label("else"), k.label("endif")
        k.emit(f"bra {other}", pred=f"!{p}")
        with k.region(), self._block():
            self.body(s.body)
            k.emit(f"bra.uni {end}")
        k.place(other)
        with k.region(), self._block():
            self.body(s.else_body)
        k.place(end)

    def emit_Loop(self, s: Loop | StridedLoop) -> None:
        """for (axis = start; axis < end; axis += step) body. Seeds the Accums of its own body,
        then takes the first shape that fits:

        1. Threads run different trip counts (``_trips`` is None): a runtime loop, one copy per
           trip.
        2. At most one trip: unrolled.
        3. Loads that ``_rereads`` re-reads every outer trip: unroll factor REREAD_COPIES.
        4. Any other: a trial unroll, copy by copy, until every trip is out or it passes
           UNROLL_SIZE lines. If every trip fit, the unroll stays. Else it is dropped, and a
           loop around loops takes unroll factor NESTED_COPIES, any other the largest that
           fits UNROLL_SIZE. Inside another trial it only unrolls, and that trial decides.

        With unroll factor f, a loop counting 0..trips (so ptxas sees its trip count) runs the
        trips every thread runs, then the rest follow as copies. Copies of independent trips
        issue their loads together, so a loop waits on memory once per f trips; but more code
        costs more to fetch."""
        k = self.k

        # Read the loop's start, step and end
        end: Lin | None
        if isinstance(s, StridedLoop):
            step = self._int(s.step) if isinstance(s.step, Expr) else Lin(const=s.step)
            start, end = self._int(s.start), None if s.end is None else self._int(s.end)
        else:
            if getattr(s, "carries", None):
                raise NotImplementedError("Loop: carried state")
            start, step, end = Lin(), Lin(const=1), None
        if end is None:
            if not s.axis.extent.is_static:
                raise NotImplementedError(f"loop over {s.axis.name}: symbolic extent")
            end = Lin(const=s.axis.extent.as_static())

        # Seed the Accums of its own body
        accums: dict[str, Accum] = {}
        for st in s.body:
            if not isinstance(st, Accum) or accums.setdefault(st.name, st) is not st:
                continue
            if s.seed:
                self._init(
                    st.name, st.op.identity, st.dtype.name if st.dtype is not None else "f32"
                )
            elif st.name not in self.env:
                raise ValueError(f"Accum {st.name!r}: seed=False but no enclosing seed")

        # Names the body writes in place live in variables made here, across every trip; no
        # result computed from them before the loop survives into it
        stmts = list(self._walk(s.body))
        names = sorted({n for st in stmts for n in self._written(st)} & set(self.env))
        written = [self._variable(n).reg for n in names]

        # 1. Threads run different trip counts: a runtime loop, one copy per trip
        trips = None if self.no_unroll else self._trips(k.ranges, start, step, end, stmts)
        if trips is None:
            self.reread |= self._rereads(s, self.mem, self.hold)
            rng, end_rng = start.range(k.ranges), end.range(k.ranges)
            counter = None
            if rng is not None and end_rng is not None:
                counter = (rng[0], end_rng[1] - 1)
            self._counted(s, start, step, end, 1, written, counter, lambda c: c)
            return
        lo, hi = trips
        n = step.const

        # 2. At most one trip: unrolled
        if hi <= 1:
            self._unrolled(s, start, n, end, lo, range(hi))
            return

        # 3. Loads that _rereads re-reads every outer trip: unroll factor REREAD_COPIES
        self.reread |= self._rereads(s, self.mem, self.hold)
        f: int | None
        if id(s) in self.reread:
            f = REREAD_COPIES
        # 4. Any other: a trial unroll decides f (None: the unroll stays, or another trial decides)
        else:
            f = self._trial(s, start, n, end, lo, hi, stmts)
            if f is None:
                return

        # With unroll factor f: a counted loop over the trips every thread runs, then the
        # rest as copies
        if f >= hi:
            self._unrolled(s, start, n, end, lo, range(hi))
            return
        m = lo // f * f  # the trips the loop runs
        one, trip = Lin(const=1), Lin(const=m)
        self._counted(s, Lin(), one, trip, f, written, (0, m - f), lambda c: start + c * n)
        self._unrolled(s, start, n, end, lo, range(m, hi))

    emit_StridedLoop = emit_Loop

    def _trial(
        self,
        s: Loop | StridedLoop,
        start: Lin,
        n: int,
        end: Lin,
        lo: int,
        hi: int,
        stmts: list[Stmt],
    ) -> int | None:
        """Case 4 of emit_Loop: unroll copy by copy until every trip is out or it passes
        UNROLL_SIZE lines. None when the unroll stays (or, inside another trial, always); else
        the unroll is dropped and this returns the unroll factor."""
        k = self.k
        outer = not self.measuring
        if outer:
            state, env = k.snapshot(), dict(self.env)
            self.measuring = True
        mark = len(k.body)
        j = self._unrolled(s, start, n, end, lo, range(hi), budget=UNROLL_SIZE)
        lines = len(k.body) - mark
        if not outer:
            return None
        self.measuring = False
        if j == hi and lines <= UNROLL_SIZE:
            return None
        k.restore(state)
        self.env = env
        nested = any(isinstance(st, Loop | StridedLoop) for st in stmts)
        return NESTED_COPIES if nested else max(1, UNROLL_SIZE * j // lines)

    def _unrolled(
        self,
        s: Loop | StridedLoop,
        start: Lin,
        n: int,
        end: Lin,
        lo: int,
        trips: range,
        budget: int | None = None,
    ) -> int:
        """Copy j of the body for each j in ``trips``, with the axis at start + j*n; a copy from
        ``lo`` on runs only where start + j*n < end. Stops before a copy once the copies made
        pass ``budget`` lines; returns where it stopped (``trips.stop`` once no trip is left)."""
        k, mark = self.k, len(self.k.body)
        for j in trips:
            if budget is not None and len(k.body) - mark > budget:
                return j
            p = True if j < lo else index.compare(k, "<", start + j * n, end)
            if p is False:
                break
            with self._where(k, p), self._block():
                self.env[s.axis.name] = start + j * n
                self.body(s.body)
        return trips.stop

    def _counted(
        self,
        s: Loop | StridedLoop,
        start: Lin,
        step: Lin,
        bound: Lin,
        copies: int,
        written: list[str],
        counter: tuple[int, int] | None,
        at: Callable[[Lin], Lin],
    ) -> None:
        """A bottom-tested loop over a new counter i, which starts at ``start``: ``copies``
        copies of the body each time round (copy j at i + j*step, so their offsets share one
        base address) while i < bound. The body sees the axis at ``at(c)`` for counter value c,
        and the counter in the range ``counter`` (when known)."""
        k = self.k
        i = k.variable("u32", index.operand(k, start))
        step_op, bound_op = index.operand(k, step * copies), index.operand(k, bound)
        enters = index.compare(k, "<", start, bound)
        if enters is False:
            return
        top, done = k.label("loop"), k.label("loop_end")
        if enters is not True:
            k.emit(f"bra {done}", pred=f"!{enters}")
        if counter is not None:
            k.ranges[i] = counter
        with k.loop(entered=enters is True):
            k.place(top)
            for reg in written:
                k.forget(reg)
            for j in range(copies):
                with self._block():
                    self.env[s.axis.name] = at(Lin.of(i) + step * j)
                    self.body(s.body)
        k.ranges.pop(i, None)
        again = k.reg("pred")
        k.emit(f"add.s32 {i}, {i}, {step_op}")
        k.emit(f"setp.lt.s32 {again}, {i}, {bound_op}")
        k.emit(f"bra {top}", pred=again)
        k.place(done)

    @staticmethod
    def _trips(
        ranges: dict[str, Range], start: Lin, step: Lin, end: Lin, stmts: list[Stmt]
    ) -> tuple[int, int] | None:
        """The trips every thread runs and the trips some thread runs, when they are at most a
        grid stride (one trip) apart; a barrier needs every thread on every trip. Else None."""
        rng = start.range(ranges)
        if not (step.is_const and step.const > 0 and end.is_const and rng is not None):
            return None
        n = step.const
        lo = max(0, -(-(end.const - rng[1]) // n))
        hi = max(0, -(-(end.const - rng[0]) // n))
        barrier = any(isinstance(st, BARRIERS) for st in stmts)
        if hi - lo > 1 or (barrier and lo != hi):
            return None
        return lo, hi

    @staticmethod
    def _written(st: Stmt) -> Sequence[str]:
        """Names a statement writes in place."""
        if isinstance(st, Accum):
            return (st.name,)
        if isinstance(st, WarpShuffle | TreeHalve):
            return tuple(st.state)
        return ()

    @staticmethod
    def _rereads(s: Loop | StridedLoop, mem: Mapping[str, Memory], hold: int) -> set[int]:
        """The read-only values alike on every trip of runtime loop ``s`` stay in registers
        across it while they fit in ``hold``. Past that they would spill: the ids of the inner
        Loops that index them, to be re-read every trip of ``s`` instead, with unroll factor
        REREAD_COPIES and ptxas left to unroll them."""

        def loads(
            body: Sequence[Stmt], loops: tuple[Loop, ...] = ()
        ) -> Iterator[tuple[Load, tuple[Loop, ...]]]:
            """Every Load in body outside a Cond, with the static Loops around it."""
            for st in body:
                if isinstance(st, Load):
                    yield st, loops
                elif isinstance(st, Loop) and st.axis.extent.is_static:
                    yield from loads(st.body, (*loops, st))
                elif not isinstance(st, Cond):
                    for inner in st.nested():
                        yield from loads(inner, loops)

        stmts = list(Emitter._walk(s.body))
        bound = {s.axis.name} | {n for st in stmts for n in (*st.defines(), *st.binds_axes())}
        seen: set[tuple[str, tuple[Expr, ...]]] = set()
        values, inner = 0, set()
        for ld, around in loads(s.body):
            buf = mem.get(ld.input)
            if not isinstance(buf, Global) or buf.is_output:
                continue
            axes = {lp.axis.name: lp for lp in around}  # an inner axis shadows an outer one
            used = set(ld.deps()) & bound
            if not used <= axes.keys():
                continue  # changes across s's trips
            inner |= {id(axes[n]) for n in used}
            if (ld.input, ld.index) not in seen:
                seen.add((ld.input, ld.index))
                values += len(ld.names) * prod(axes[n].axis.extent.as_static() for n in used)
        return inner if values > hold else set()

    # --- core: expressions --------------------------------------------------------------

    def expr(self, e: Expr) -> Value:
        """An index expression as a Lin, data as a Val, a decided predicate as a bool."""
        k = self.k
        if isinstance(e, Var):
            return self._lookup(e.name)
        if isinstance(e, Literal):
            if e.dtype == "bool" or isinstance(e.value, bool):
                return bool(e.value)
            if e.dtype == "int" or isinstance(e.value, int):
                return Lin(const=int(e.value))
            return Val(ops.const(k, float(e.value), "f32"), "f32")
        if isinstance(e, BinaryExpr):
            return self._binary(e)
        if isinstance(e, TernaryExpr):
            p = self._pred(e.cond)
            if isinstance(p, bool):
                return self.expr(e.if_true if p else e.if_false)
            a, b = self.expr(e.if_true), self.expr(e.if_false)
            if isinstance(a, Lin) and isinstance(b, Lin):
                return index.select(k, p, a, b)
            a, b = self._as_val(a), self._as_val(b)
            dtype = a.dtype if a.dtype == b.dtype else "f32"
            x, y = ops.convert(k, a, dtype).reg, ops.convert(k, b, dtype).reg
            return ops.select(k, p, x, y, dtype)
        if isinstance(e, FuncCallExpr):
            return ops.apply(k, e.name, [self._value(a, "f32") for a in e.args], "f32")
        if isinstance(e, CastExpr) and e.dtype == "float":
            return self._value(e.expr, "f32")
        raise NotImplementedError(f"f32 expression {e!r}")

    def _binary(self, e: BinaryExpr) -> Value:
        k, op = self.k, e.op
        # Handle boolean expressions
        if op in ("&&", "||"):
            pa, pb = self._pred(e.left), self._pred(e.right)
            if isinstance(pa, str) and isinstance(pb, str):
                return Val(k.pure("and.pred" if op == "&&" else "or.pred", "pred", pa, pb), "bool")
            const, other = (pa, pb) if isinstance(pa, bool) else (pb, pa)
            # True && x and False || x are x; otherwise the constant decides
            p = other if const == (op == "&&") else const
            return p if isinstance(p, bool) else Val(p, "bool")

        # Recursively eval expression
        a, b = self.expr(e.left), self.expr(e.right)

        # Handle integer index expressions
        if isinstance(a, Lin) and isinstance(b, Lin):
            if op in ops.CMP_SYMBOL:
                p = index.compare(k, op, a, b)
                return p if isinstance(p, bool) else Val(p, "bool")
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return index.mul(k, a, b)
            if op in ("/", "//"):
                return index.div(k, a, b)
            if op == "%":
                return index.mod(k, a, b)
            raise NotImplementedError(f"index expression {e!r}")

        # Handle floating point expression
        x, y = (ops.convert(k, self._as_val(v), "f32") for v in (a, b))
        if op in ops.CMP_SYMBOL:
            return Val(k.pure(f"setp.{ops.CMP_SYMBOL[op]}.f32", "pred", x.reg, y.reg), "bool")
        if op in F32_SYMBOL:
            return ops.apply(k, F32_SYMBOL[op], [x, y], "f32")
        raise NotImplementedError(f"f32 expression {e!r}")

    def _as_val(self, v: Value) -> Val:
        if isinstance(v, Val):
            return v
        if isinstance(v, bool):
            one = index.reg_of(self.k, Lin(const=int(v)))
            return Val(self.k.pure("setp.ne.s32", "pred", one, "0"), "bool")
        return Val(index.reg_of(self.k, v), "i32")

    def _value(self, e: Expr, dtype: str) -> Val:
        """Expression e as data of ``dtype``; a literal is a constant of that type."""
        if isinstance(e, Literal) and e.dtype != "bool" and not isinstance(e.value, bool):
            return Val(ops.const(self.k, float(e.value), ptx_type(dtype)), dtype)
        return ops.convert(self.k, self._as_val(self.expr(e)), dtype)

    def _int(self, e: Expr) -> Lin:
        """Expression e as an index. Integer data is copied out of a variable first, since a
        Lin may outlive the variable's current value."""
        v = self.expr(e)
        if isinstance(v, Lin):
            return v
        if isinstance(v, bool) or not is_int(v.dtype):
            raise NotImplementedError(f"index expression {e!r} is not an integer")
        reg = ops.convert(self.k, v, "i32").reg
        if reg in self.k.variables:
            copy, reg = reg, self.k.reg("u32")
            self.k.emit(f"mov.b32 {reg}, {copy}")
        return Lin.of(reg)

    def _pred(self, e: Expr) -> str | bool:
        """Expression e as a predicate register, or a bool when it is decided here."""
        v = self.expr(e)
        if isinstance(v, bool):
            return v
        if isinstance(v, Lin):
            return index.compare(self.k, "!=", v, Lin())
        return ops.convert(self.k, v, "bool").reg
