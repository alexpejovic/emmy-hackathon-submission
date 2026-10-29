# ptxgen: Emmy Kernel IR to PTX

This directory holds our compiler. It takes one Emmy **Kernel IR** kernel and writes **PTX**, the text form of
NVIDIA GPU code. `ptxas` (or the driver) then turns that PTX into machine code.

The compiler has one main goal: produce correct PTX that runs fast on the runner's GPU. Each main optimization has a
flag that turns it off, so its effect can be measured on its own.

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Files](#files)
- [Optimizations](#optimizations)
- [Limits](#limits)

## Quick start

First, ensure `python --version` gives something >=3.12. Then,

```sh
make .venv
git submodule update --init --recursive
.venv/bin/python compile.py problems/p01_vector_add/k_add_pointwise.ir.json > out.ptx
```

The compiler reads two files:

- `<kernel>.ir.json` — the Kernel IR graph.
- `meta.json` — in the same directory. It gives the order of the kernel parameters, the block size, and each
  buffer's shape, dtype and whether it is an output.

It prints the full `.ptx` module to stdout.

Three flags turn off one optimization each:

| Flag | Effect |
| --- | --- |
| `--no_loop_unroll` | Every loop is a runtime loop with one copy of the body per trip |
| `--no_fusion` | We never fuse a float multiply and add into one `fma` (`ptxas` still may) |
| `--no_cse` | Identical side-effect-free instructions are emitted again instead of reused |

From the repo root:

- `make compile-check` compiles every problem and reports which ones fail.
- `make ptxas-check` assembles every problem's PTX with `ptxas`, for sm_89 and sm_120.
- `make myptx` writes `*.my.ptx` next to each IR, so you can diff it against the nvcc references `*.ref.ptx` and
  `*.ref.fast.ptx` (built with `--use_fast_math`, which is closer to what we emit for math functions).

## How it works

The compiler is a single walk over the Kernel IR tree. There is no separate optimization pass over PTX. Each
optimization happens while the PTX is being written, or in one small cleanup at the end.

```
<kernel>.ir.json + meta.json
        │
        ▼
compile.py: to_ptx           read the graph, find the KernelOp that meta.json names,
                             take arg order / block size / buffers from meta.json
        │
        ▼
compiler.py: compile_module  build an Emitter for the kernel
        │
        ▼
Emitter.run
  1. load every pointer parameter and convert it to a global address
  2. thread id and block id; lane / warp ids (only if the kernel uses warp shuffles or tree reductions)
  3. declare every shared-memory buffer up front
  4. body(stmts): emit_<NodeType> for each node in the kernel body, which may recurse into nested bodies
  5. Kernel.drop_dead          drop results nothing reads (multiplies fused into an fma)
        │
        ▼
builder.py: Kernel.render    header, register declarations, shared memory, body
        │
        ▼
PTX text
```

### Nodes become instructions

`Emitter.body` looks up a method named `emit_<NodeType>` for each Kernel IR node. Supporting a new node means adding
one method. A node with no method raises `NotImplementedError`.

| Kernel IR node | What it becomes |
| --- | --- |
| `Tile` | Global thread id, tail guard (only when the last block is partial), and the decode of the id into the tile's axes |
| `Load` / `Write` | `ld` / `st` from global or shared memory, scalar or vector (`.v2`, `.v4`); `red.global.add` for split-K atomics |
| `Assign`, `Let`, `Select` | Arithmetic, conversions, `selp` for `where` and ternaries |
| `Loop`, `StridedLoop` | A loop, fully unrolled, partly unrolled, or a runtime loop (see [Loop unrolling](#loop-unrolling)) |
| `Accum`, `Init` | A variable register, seeded once and updated in place |
| `Cond` | `if` / `else` with predicated branches; only one side when the compare is known |
| `Smem` | Nothing: every buffer is declared up front, as a static `.shared` array or an offset in one dynamic pool |
| `Sync` | `bar.sync` or `bar.warp.sync` |
| `WarpShuffle` | `activemask`, then a `shfl.sync.bfly` ladder (xor 16, 8, 4, 2, 1) |
| `TreeHalve` | A shared-memory tree reduction, fully unrolled |
| `CpAsyncCopy`, `CpAsyncCommit`, `CpAsyncWait` | `cp.async` global → shared copies and their group waits |

### Values and variables

Every value gets a new virtual register (`%f12`, `%r7`, ...). We never pick physical registers; `ptxas` does that.
`Emitter.env` maps each IR name to what holds it.

Registers come in two kinds:

- **Values** are written once. Almost every register is a value.
- **Variables** are written in place: accumulators, `Init` state, the states of a warp shuffle or tree reduction,
  and loop counters. `Kernel.variable` makes them.

A name defined inside a block (a loop body, an `if` body) ends with that block.

### Three kinds of values

- **Index values** are a `Lin`: an affine form `sum(coeff * register) + const`. They are unsigned 32-bit integer
  math for addresses and loop counters. This is where constant folding, strength reduction and address folding
  happen (see `index.py`).
- **Data values** are a `Val`: a register and its Emmy dtype. They follow Emmy's type rules.
- **Decided predicates** are a Python `bool`. A compare the compiler can decide never reaches a register.

### Data types

Data may be `f32`, `f64`, `f16`, `bf16`, `bool` or an integer type. The rules follow Emmy:

- An op runs natively when Emmy runs it natively: simple `f16` ops (`F16_OPS`), integer ops (`INT_OPS`).
- `f64` stays in `f64` (`F64_OPS`).
- Every other op converts its inputs to `f32`, computes, and converts the result back once.
- Basic float ops (`add`, `mul`, `div`, `sqrt`, ...) are IEEE round-to-nearest. `add`, `sub` and `mul` carry no
  rounding modifier, as in nvcc's output, so `ptxas` may fuse a multiply and an add into an `fma` on its own.
  `fma`, `div`, `rcp`, `sqrt` and rounding `cvt`s keep an explicit `.rn`, because PTX requires one there.
- Math functions (`exp`, `log`, `pow`, `tanh`, `erf`, `sin`, `cos`, `sigmoid`) and `rsqrt` use the fast hardware
  approximations, as a fast-math build does. They are not bit for bit equal to nvcc's default build.
- `cvt` follows C's rounding rules. `bf16 → f32` uses a bit shift, because the direct `cvt` needs sm_90.

## Files

### `compile.py`

The command-line entry point. It loads the IR graph, reads `meta.json`, and finds the `KernelOp` whose name
`meta.json` lists. `meta.json` is the contract for parameter order, launch size and buffers: each buffer's shape,
dtype and output flag come from it, not from the IR. It also parses the three flags.

### `ptxgen/compiler.py`

The core. `Emitter` walks the Kernel IR and calls into the other modules. It is split into sections:

- **core** — walking the body (`run`, `body`, `_block`, `_walk`), names (`_lookup`, `_define`, `_variable`,
  `_write`), and expressions (`expr`, `_binary`, `_value`, `_int`, `_pred`).
- **launch** — `emit_Tile`, `emit_Sync`, the thread decode (`_decode`, `_raster`).
- **memory** — `emit_Load`, `emit_Write`, the `cp.async` nodes, `_declare_shared`, `_flat`.
- **data** — `emit_Assign`, `emit_Let`, `emit_Init`, `emit_Select`, `emit_Accum`.
- **reduce** — `emit_WarpShuffle`, `emit_TreeHalve`.
- **control** — `emit_Cond`, `emit_Loop` and its loop shapes (`_trial`, `_unrolled`, `_counted`, `_trips`,
  `_rereads`).

The unroll limits at the top (`UNROLL_SIZE`, `NESTED_COPIES`, `REREAD_COPIES`) were measured on the runner's
RTX 5090.

### `ptxgen/builder.py`

A thin PTX text builder. It knows nothing about Emmy.

- `Kernel` holds the instruction list, register counters, labels, shared-memory declarations and known index
  ranges.
- `reg(typ)` returns a new virtual register; `variable(typ, init)` returns one that is written in place.
- `pure(op, typ, *srcs)` emits a side-effect-free instruction, or returns an earlier identical one. This is the
  main value-reuse mechanism (see [Value reuse](#value-reuse-cse)). Inside a runtime loop it also moves
  loop-invariant arithmetic in front of the loop (see [Loop-invariant hoisting](#loop-invariant-hoisting)).
- `region()`, `loop()` and `unless()` open blocks, so `pure` only reuses values that are always defined.
- `forget(reg)` drops reusable results that read a register about to change.
- `snapshot()` and `restore()` save and roll back all state, for a trial unroll.
- `drop_dead()` removes `pure` results that nothing reads.
- `render()` writes the final text: header, `.reg` declarations, shared memory, body.

### `ptxgen/index.py`

Integer index math. `Lin` is an affine form over registers. `reg_of` puts one into a register only when needed.
`mul`, `div`, `mod`, `minimum`, `compare` and `select` keep results affine where the known ranges allow it.
`_rejoin` folds a split flat index back together.

### `ptxgen/memory.py`

Buffers the kernel reads or writes. `Memory` knows a buffer's shape and dtype, and how to load and store element `i`.
Its two subclasses differ only in `address`:

- `Global` — a parameter buffer. Read-only buffers load through `ld.global.nc`.
- `Shared` — a `.shared` buffer, or an offset in the dynamic pool.

### `ptxgen/ops.py`

Instruction sequences and op tables, keyed by Emmy op name:

- Tables: `ALIASES`, `BINARY`, `ACCUM`, `COMPARE`, `CMP_SYMBOL`, `F32_OPS`, `F64_OPS`, `F16_OPS`, `INT_OPS`.
- Emmy's dtype rule over data: `Val`, `apply`, `assign`, `convert`, `select`, `cvt`, `const`.
- Fusion: `fma_operands`.
- Memory and shuffles: `load`, `store`, `chunks` (vector widths), `shfl_xor`.

### `ptxgen/dtypes.py`

Data, not logic. `PTX_TYPE` maps each Emmy dtype to its register type and its `ld` / `st` type. `REG_CLASS` gives the
register prefix (`%f`, `%r`, `%rd`, `%rs`, `%p`, ...). `BYTES` gives sizes. `lit` writes constants as exact hex
literals, so they survive bit for bit.

### `ptxgen/fastmath.py`

The `f32` math functions in `F32_FUNCS`, built on `ex2.approx`, `lg2.approx`, `sin.approx`, `cos.approx` and
`rcp.approx` (about 2 ulp). `pow` follows `powf`'s sign rules. `tanh` is `1 - 2 / (e^2x + 1)`, so it saturates
cleanly to ±1. `erf` uses the Abramowitz-Stegun 7.1.26 formula.

## Optimizations

A note on scope first. Emmy already does **kernel fusion**: it merges many operations into one kernel before we see
it. Our compiler works on one kernel at a time. Its optimizations are about the instructions inside that kernel.

### Value reuse (CSE)

**What.** If the same side-effect-free instruction with the same inputs was already emitted, and its result is still
valid here, reuse its register. This is common subexpression elimination. It saves instructions and registers.
`--no_cse` turns it off.

**Where.** `Kernel.pure`, `Kernel.region`, `Kernel.forget` in `builder.py`.

**How.**

- `pure` keys each instruction by `(op, *sources)` and stores the result register in a scope.
- Scopes form a stack. `region()` opens a new scope for code that runs only sometimes, such as an `if` body or a loop
  body. When the region closes, its results are dropped, because they do not exist on every path after it.
- Lookups search from the innermost scope outwards, so code can reuse anything from blocks that always run before it.
- When a variable is about to change in place, `forget` drops every stored result that read it.
- A runtime loop forgets the variables its body writes before the body starts, since the back edge brings them
  round again with new values.

Index math, address math, conversions, compares and constants all go through `pure`. So an address like `base + i*4`
is computed once, even when many loads use it.

**Repeated loads.** Scalar loads from buffers the kernel never writes (`ld.global.nc`) also go through `pure`. When
unrolling copies a load of the same broadcast value (for example a bias or `eps`), only the first copy loads it.

### Loop-invariant hoisting

**What.** Inside a runtime loop, arithmetic whose inputs all exist before the loop moves in front of it. It then runs
once instead of on every trip.

**Where.** `Kernel.loop` and `Kernel._hoist_to` in `builder.py`.

**How.** `loop()` records where the loop starts and which registers already exist. When `pure` emits an instruction
whose sources are all older than the loop and none is a variable, it inserts it in front of the loop instead.

Two limits, both measured:

- **Arithmetic only.** A hoisted load keeps its value live across the whole loop. That ran slower.
- **One level only.** A value live across a whole loop nest costs registers that `ptxas` spends better.

### Instruction fusion (fma)

**What.** `a * b + c` becomes one `fma.rn` instead of a `mul` and an `add`. This is faster, and it matches nvcc,
which makes the same contraction by default. `--no_fusion` turns off our fusion. Plain `add` and `mul` carry
no rounding modifier, so `ptxas` may still fuse them on its own.

**Where.** `fma_operands` and `_add` in `ops.py`; `emit_Accum` in `compiler.py`; `Kernel.drop_dead` in `builder.py`.

**How.**

1. When an `add` is about to be emitted, `fma_operands` checks whether one input is the result of a `mul` of the
   same type that is still cached (`Kernel.definition`).
2. If so, it emits `fma.rn` with the `mul`'s inputs and the other `add` input. An `Accum` that adds does the same.
3. If a `mul` source is rewritten in place, `forget` drops the `mul` from the cache, so it can no longer fuse. We
   never read a stale value.
4. At the end, `drop_dead` deletes each `mul` whose result nothing else reads.

```
before                         after
mul.f32 %f3, %f1, %f2;         fma.rn.f32 %f4, %f1, %f2, %f0;
add.f32 %f4, %f3, %f0;
```

Only `f32` and `f64` fuse.

### Loop unrolling

**What.** Copy the loop body so the loop runs fewer times, or not at all. This removes branches and counter updates,
and lets constant offsets fold into addresses. Copies of independent trips also issue their loads together, so the
loop waits on memory once per group of trips instead of once per trip. `--no_loop_unroll` turns it off.

**Where.** `emit_Loop`, `_trips`, `_trial`, `_unrolled`, `_counted`, `_rereads` in `compiler.py`.

**How.** `_trips` first works out the range of the start value, and so the trip count every thread runs (`lo`) and
the trip count some thread runs (`hi`). Then `emit_Loop` takes the first shape that fits:

| Shape | When |
| --- | --- |
| Runtime loop, one copy per trip | Trip counts are unknown or differ by more than one between threads; a barrier and differing trip counts; or `--no_loop_unroll` |
| Full unroll | At most one trip |
| Unroll factor `REREAD_COPIES` (8) | The loop's read-only loads would not fit in registers across an outer loop (`_rereads`) |
| Trial unroll (`_trial`) | Any other loop. Unroll copy by copy. If every trip fits in `UNROLL_SIZE` (8192) lines, keep it. Otherwise roll back and use unroll factor `NESTED_COPIES` (2) for a loop around loops, or the largest factor that fits in `UNROLL_SIZE` for any other |

Details that keep it correct and cheap:

- **Unroll factor.** With unroll factor `f`, the counted loop holds `f` copies of the body, so each time round it
  runs `f` trips. It covers the first `lo // f * f` trips, counting from 0 so `ptxas` sees its trip count. The trips
  left over follow as plain copies.
- **Guards.** Every thread runs at least `lo` trips, so those copies need no check. Copies between `lo` and `hi` are
  wrapped in an `if (axis < end)`. A copy that can never run is skipped.
- **Offsets fold into addresses.** In copy `j`, the axis is `i + j*step`. `Memory.address` moves that constant into
  the load's `[reg+imm]` offset, so all copies share one address register.
- **Barriers.** A loop that contains a barrier (`Sync`, `WarpShuffle`, `TreeHalve`, `CpAsyncWait`) is only unrolled
  when every thread runs the same number of trips. Otherwise some threads would wait at a barrier the others never
  reach.
- **Nested trials.** Inside another loop's trial unroll, an inner loop only unrolls. The outer trial decides.
- **Register pressure.** `_rereads` counts the read-only values a runtime loop would keep live across its trips. Past
  half the per-thread register budget they would spill, so the inner loops that load them re-read them every trip
  instead.

### Range analysis and compile-time compares

**What.** Track the known `[lo, hi]` range of index registers, and decide compares at compile time when possible.

**Where.** `Kernel.ranges` in `builder.py`; `Lin.range` and `compare` in `index.py`.

**How.** The thread id is known to lie in `[0, block-1]`, and the block id in `[0, blocks-1]`. Lane and warp ids,
loop counters and the results of `shr`, `and`, `div` and `rem` get ranges too. `Lin.range` adds up the ranges of an
affine form.

The ranges give the loop unroller its trip counts. `compare` returns a Python `bool` whenever the range of `a - b`
decides it. `Cond` and ternaries then emit only the side that runs. This matters most after full unrolling, where
many compares become constant.

### Constant folding and strength reduction

**What.** Compute constant index math at compile time. Replace divide and modulo by a constant with cheap bit
operations, or with nothing at all.

**Where.** `Lin`, `div`, `mod`, `_divmod`, `mul`, `_mul_imm` in `index.py`.

**How.**

- `+`, `-` and `*` by a constant stay inside the `Lin`, with no instruction. Constant parts never reach a register.
- `_divmod` splits `a = m*q + rest`, where `q` collects every term that `m` divides. Only `rest` needs a division.
- If `rest` stays in `[0, m)`, it never carries into the quotient: no instruction at all. For example,
  `(ctaid * 128 + tid) / 128` is just `ctaid` when `tid < 128`.
- Otherwise `x / 2^k` → `shr.u32`, `x % 2^k` → `and.b32`, `x * 2^k` → `shl.b32`.
- Index values are never negative, so unsigned shifts are safe.

### Address folding

**What.** Keep constant parts of an index out of registers and put them into the load or store offset.

**Where.** `_flat` in `compiler.py`; `Memory.address` in `memory.py`; `_rejoin` in `index.py`.

**How.**

- `_flat` flattens a multi-dimensional index to one row-major `Lin`.
- `address` puts the `Lin`'s constant into the `imm` of `[reg+imm]`. A fully constant index needs no address math at
  all.
- `_rejoin` notices when the IR split one flat index into `(x / m) * m + x % m` and joins it back into `x`. This
  removes a divide, a modulo, a multiply and an add.
- In global memory the address is one `mul.wide.u32` and one `add.s64`. In shared memory it is one `mad.lo` on
  32-bit addresses.

### Thread decode and tail guard

**Where.** `emit_Tile`, `_decode` and `_raster` in `compiler.py`.

- The tail guard (`if (idx >= n) exit`) is a range compare, so it disappears when every block is full.
- Each axis is decoded from the thread id with `index.div` and `index.mod`, so a power-of-two extent costs a shift
  or a mask, and an axis the ranges already decide costs nothing.
- The outermost axis skips the modulo.
- Grouped block ordering (rasterization) for matmul tiles is decoded as Emmy does it, including a ragged last group.

### Memory and reductions

These follow what the IR asks for, emitted in the cheapest form:

- **Vector loads and stores.** Consecutive values become `.v4` or `.v2` accesses, up to 16 bytes each (`chunks`).
- **Read-only loads.** Buffers the kernel never writes load through `ld.global.nc`.
- **`cp.async`.** 16-byte copies use `.cg` (skip L1); 4- and 8-byte copies use `.ca`, as Emmy does.
- **Warp shuffles.** A `shfl.sync.bfly` ladder under `activemask`. 64-bit values move as two halves; 16-bit values
  widen to 32 bits.
- **Tree reductions.** `TreeHalve` has a fixed length, so its halving loop is fully unrolled.
- **Large shared memory.** Past 48 KB (`STATIC_SMEM_CAP`), all shared buffers become offsets in one dynamic pool,
  sized at launch.
- **Atomics.** Split-K partial sums use `red.global.add`, which does not return the old value.

## Limits

These raise `NotImplementedError`, so failures are loud and easy to find:

- Kernel IR nodes with no `emit_<Node>` method.
- Launches that are not 1-D, and by-value scalar arguments.
- Tiles with a dynamic grid, an aux thread band, or more than 2^31 − 1 elements.
- Loops with a symbolic (non-static) extent, or with carried state.
- Swizzled writes and swizzled `cp.async`.
- `cp.async` zero-fill and `shfl.up` prefix forms (newer Emmy only).
- Atomic adds to shared memory, or of dtypes other than `f32`, `f64`, `f16`, `i32`, `u32`, `u64`.
- Shared buffers of dtypes other than `f32`, `f64`, `f16`, `bf16`, `i32`, `u32`, `i64`, `u64`.
- A multi-dimensional index whose rank differs from the buffer's.
- Shifts of integers that are not 32 bits wide.
- Index expressions that are not integers.
- Dtypes, ops and expressions with no entry in the tables above.
