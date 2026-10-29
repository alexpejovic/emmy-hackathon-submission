# Common subexpression elimination in our compiler

## What is here

Each folder holds one problem's kernel, compiled two ways by `compile.py`:

| File | What it is |
|---|---|
| `<kernel>.ir.json`, `<kernel>.ir.txt` | The Kernel IR (JSON for the compiler, text to read) |
| `meta.json` | Launch shape and buffers; the compiler needs it beside the IR |
| `<kernel>.cse.ptx` / `.sass` | Our compiler as it is: an op that was already computed is not emitted again |
| `<kernel>.nocse.ptx` / `.sass` | Same, with `--no_cse`: every op is emitted each time it is needed |

The SASS is for `sm_120` (the runner's RTX 5090), from `ptxas -arch=sm_120` and `cuobjdump -sass`.

The three `cse` builds are byte for byte the `unroll` builds in `../loop_unroll`. The `a01_attention` one is
also the `fusion` build in `../op_fusion`.

To regenerate one folder, for example `r02_big_sum`:

```sh
cd r02_big_sum
python ../../../compile.py k_sum_1_reduce.ir.json          > k_sum_1_reduce.cse.ptx
python ../../../compile.py k_sum_1_reduce.ir.json --no_cse > k_sum_1_reduce.nocse.ptx
ptxas -arch=sm_120 k_sum_1_reduce.cse.ptx -o /tmp/k.cubin && cuobjdump -sass /tmp/k.cubin > k_sum_1_reduce.cse.sass
```

`ptxas` comes from the `nvidia-cuda-nvcc` wheel and `cuobjdump` from the `triton` wheel, both in `.venv`.

## Results

Median kernel time on the runner, in µs (`tools/bench.py`, 3 to 5 rounds, best median). All variants pass
`submit.py`, with the same max error in both builds.

| Kernel | CSE | No CSE | Slowdown | Registers (CSE / no CSE) |
|---|---|---|---|---|
| `a01_attention` | 51.87 | 146.08 | 2.82× | 255 (316 B spilled) / 56 |
| `r02_big_sum` | 6.85 | 6.85 | 1.00× | 38 / 38 |
| `m01_matmul_512` | 15.04 | 15.04 | 1.00× | 46 / 64 |

PTX instructions leave out declarations. A "wait" is a point where a warp stops until earlier loads arrive,
counted with `tools/sass_waits.py` (for the no-CSE build, give it a `--compiler` script that adds
`--no_cse`).

| Kernel | CSE | No CSE |
|---|---|---|
| `a01_attention` | 582 PTX instructions. Inner loop body: 33–35. Per trip of the `a1` loop (2 steps), ptxas unrolls the inner loops: 1,010 instructions, 130 global loads, 138 spill loads, 27 waits | 1,108 PTX instructions. Inner loop body: 75–91. Per trip of the `a1` loop, ptxas keeps 8 inner loops of 8 passes: 2,260 instructions, 1,026 global loads, no spill loads, 138 waits |
| `r02_big_sum` | 1,084 PTX instructions | 2,622 PTX instructions. The SASS is byte for byte the same: 1,080 instructions, 21 waits |
| `m01_matmul_512` | 4,206 PTX instructions, no loop. SASS: 2,976 instructions, 134 waits | 3,654 PTX instructions, a loop with 2 copies per trip. ptxas unrolls it. SASS: 3,184 instructions, 83 waits |

### a01_attention

4 blocks of 128 threads, so few warps run and memory latency is hard to hide. The main loop runs over `a1`
(64 steps, 2 per trip). Each step has four inner loops over `a2`, each a dot product of a row of `q` with a
column of `kt`. Our compiler keeps those inner loops as PTX loops with 8 copies per trip (see
`../loop_unroll`).

With CSE, the 8 copies of a trip share one address per array. Each load is that address plus a constant:

```
mad.lo.s32 %r14, %r3, 2048, %r13;
mad.lo.s32 %r15, %r5, 256, %r14;
mul.wide.u32 %rd14, %r15, 4;
add.s64 %rd15, %rd3, %rd14;
ld.global.nc.f32 %f52, [%rd15+256];      // q
...
ld.global.nc.f32 %f55, [%rd15+260];      // q, same base
ld.global.nc.f32 %f56, [%rd17+256];      // kt, same base
```

Without it, every copy computes its two addresses again, 4 or 5 instructions per load. The loop body grows
from about 35 PTX instructions to 75–91. The 16 loads per trip stay the same.

ptxas removes the copied address math again: each inner loop is 32 SASS instructions per pass. But it no
longer unrolls the inner loops. With CSE it unrolls them fully. Then it sees that the 256 `q` values do not
change across `a1`, and loads them once before the loop (255 registers, some spills). Without CSE, the 8
inner loops of each trip run 8 passes each, and every pass loads its 8 `q` and 8 `kt` values again. That is
1,026 global loads per trip instead of 130, and 138 waits instead of 27. No spills and 56 registers do not
make up for it: 2.82× slower.

We think ptxas decides whether to unroll a loop by its size before it removes the copies. We have not
confirmed it. The `--no_loop_unroll` build in `../loop_unroll` loses the same way, and less: it reloads `q`
too, with 321 loads per step, in 82.62 µs.

### r02_big_sum

One block of 128 threads sums 65,536 floats, 512 per thread. Our compiler unrolls it fully: 512 loads in a
straight line. With CSE, the address is computed once and each load adds a constant:

```
mad.lo.s32 %r5, %r1, 128, %r0;
mul.wide.u32 %rd4, %r5, 4;
add.s64 %rd5, %rd1, %rd4;
ld.global.nc.f32 %f1, [%rd5+0];
add.rn.f32 %f0, %f0, %f1;
ld.global.nc.f32 %f2, [%rd5+512];
add.rn.f32 %f0, %f0, %f2;
```

Without it, the same three instructions come before every load:

```
mad.lo.s32 %r5, %r1, 128, %r0;
mul.wide.u32 %rd4, %r5, 4;
add.s64 %rd5, %rd1, %rd4;
ld.global.nc.f32 %f1, [%rd5+0];
add.rn.f32 %f0, %f0, %f1;
mad.lo.s32 %r6, %r1, 128, %r0;
mul.wide.u32 %rd6, %r6, 4;
add.s64 %rd7, %rd1, %rd6;
ld.global.nc.f32 %f2, [%rd7+512];
add.rn.f32 %f0, %f0, %f2;
```

That is 1,538 more instructions, 2.4× the PTX. ptxas sees that all 512 addresses are the same and removes
every copy. The SASS of both builds is byte for byte the same, and so is the time.

### m01_matmul_512

Here CSE changes a choice our compiler makes. The outer loop over `_ks` (8 trips) holds the inner loop over
`_ki` (64 trips). Our compiler tries to unroll a loop fully and gives up once the PTX passes 8,192
instructions (see `../loop_unroll`). With CSE the full unroll fits, so it stays (4,206 instructions in the
final PTX). Without it, the copies make it pass 8,192. The compiler then keeps `_ks` as a loop with 2 copies
per trip.

ptxas unrolls that loop itself, since its trip count is known. The SASS has no loop in either build. It ends
up a little different: 3,184 instructions and 64 registers instead of 2,976 and 46, and 83 waits instead of
134. The time does not move. The loads come from shared memory, which is fast, and the `cp.async` copies
already overlap global loads with compute.

### Other kernels we checked

We built every open problem both ways. In 8 of the 15, the SASS is byte for byte the same:

| Kernel | PTX instructions (CSE / no CSE) |
|---|---|
| `a02_attention_128` | 227 / 578 |
| `r03_mean` | 576 / 1,346 |
| `r01_array_sum` | 76 / 102 |
| `r04_row_sum` | 70 / 83 |
| `p00_scalar_add` | 25 / 39 |
| `p01_vector_add` | 22 / 36 |
| `p02_scale_shift` | 27 / 48 |
| `p03_relu_mul` | 32 / 62 |

The other 7 have different SASS. We timed them (3 rounds; `m02_matmul_rect` in three runs):

| Kernel | PTX instructions | SASS | CSE | No CSE |
|---|---|---|---|---|
| `a01_attention` | 582 / 1,108 | see above | 51.87 | 146.08 |
| `m01_matmul_512` | 4,206 / 3,654 | see above | 15.04 | 15.04 |
| `f01_rmsnorm` | 120 / 214 | 224 / 232 instructions, 28 / 34 registers | 4.80 | 4.80 |
| `f02_row_softmax` | 176 / 201 | 232 / 232 instructions, 19 / 16 registers, 4 / 5 waits | 2.94 | 2.94 |
| `f03_l2norm` | 76 / 116 | 168 / 176 instructions, 4 / 5 waits | 2.94 | 2.94 |
| `m02_matmul_rect` | 104 / 163 | same 89 instructions, address math in a different order | 4.82–5.02 | 4.58–4.77 |
| `r02_big_sum` | 1,084 / 2,622 | the same | 6.85 | 6.85 |

`m02_matmul_rect` was 1–6% faster without CSE in all three runs. Its SASS has the same 89 instructions. Only
the address math before the first loads is in a different order. We do not know if that is the cause, or
noise: `../op_fusion` saw this kernel move between 4.70 and 4.86 µs from run to run.

Its max error also moves between two values from run to run, in both builds. The kernel adds its result to
the output with atomic adds, whose order changes.

## Why CSE makes a kernel faster

Mostly not by itself. ptxas does its own CSE, of arithmetic and of read-only loads. In 8 of 15 problems the
SASS does not change at all.

It matters where a choice is made from the size of the PTX, before the copies are gone:

1. **ptxas decides how far to unroll.** In `a01_attention`, a loop body 2.5× larger is no longer unrolled.
   ptxas then reloads `q` on every pass: 2.82× slower.
2. **Our compiler decides how far to unroll.** It counts PTX instructions. In `m01_matmul_512`, the copies
   make a full unroll look too big. Here ptxas unrolls the loop itself, so nothing is lost.

It also keeps the PTX short and easy to read. Where our unroll choices stay the same, the PTX without it is
1.1× to 2.6× longer.

## How our compiler does CSE

Every side-effect-free op goes through one function, `Kernel.pure`: arithmetic, compares, conversions,
constants, address math, and loads of memory nothing writes during the kernel (`ld.global.nc`). An op is
known by its key: the op and its operands, for example `("mad.lo.s32", "%r3", "4", "%r4")`. If a result with
that key is still valid here, `pure` returns its register and emits nothing. Otherwise it emits the op into a
new register and remembers it under the key.

A result stays valid only where it is sure to exist:

- **In its block, and in blocks it dominates.** A block dominates another when every path to the other runs
  through it first. The builder keeps a stack of scopes, one per open block. `region` opens a block that does
  not dominate what follows: an `if` or `else` body, a copy only some threads run (`unless`), a loop body.
  Its results end with it.
- **Until an input is written in place.** Accumulators and loop counters are written in place. `forget`
  drops every result that read one.
- **Not across a loop's back edge.** Before a runtime loop body, results that read a variable the body writes
  are forgotten, since the next trip brings a new value.

Loads of shared memory, and of global memory the kernel writes, never go through `pure`. A store or a
barrier could change them.

The end of `r02_big_sum` shows the scope rule. Three guarded blocks need the same address `%r3 * 4 + %r4`:

```
	@!%p0 bra $L_skip_1;
	mad.lo.s32 %r7, %r3, 4, %r4;
	st.shared.f32 [%r7+0], %f0;
$L_skip_1:
	bar.sync 0;
	setp.lt.s32 %p1, %r3, 2;
	@!%p1 bra $L_skip_2;
	mad.lo.s32 %r8, %r3, 4, %r4;
	...
$L_skip_2:
	...
	@!%p2 bra $L_skip_3;
	mad.lo.s32 %r9, %r3, 4, %r4;
```

Each block computes it again. A thread that skipped the first block never computed `%r7`, so the second block
cannot use it. Had the first `mad.lo` sat before the first branch, both blocks would have reused it.

Two other parts of the compiler use the same cache:

- **Hoisting.** In a runtime loop, arithmetic whose inputs all exist before the loop is emitted in front of
  it, and cached there. If every thread enters the loop, the result stays valid after the loop too.
- **Fusion.** An add checks whether an operand is a still-valid `mul.rn` result, to make one `fma.rn`
  (see `../op_fusion`).

`--no_cse` gives every op a key of its own, so no lookup ever matches. Hoisting and fusion still work; each
copy just hoists its own ops. Simply skipping the lookup would not be safe. Two results would then share one
key, and `forget` would drop only the newer one. Fusion could then use the older multiply after one of its
inputs had changed.

## What ptxas does

ptxas does its own CSE:

- **Of arithmetic.** In `r02_big_sum`, 1,538 extra address instructions disappear. The SASS is the same.
- **Of read-only loads.** `a02_attention_128` has 131 global loads without our CSE and 99 with it. The SASS
  is the same.

What it does not undo is a choice already made from the bigger PTX: its own unrolling in `a01_attention`,
and ours in `m01_matmul_512`.

## Summary

- CSE matters a lot for `a01_attention` (2.82×). Without it, ptxas stops unrolling the inner loops and
  reloads `q` on every pass.
- For most kernels it changes nothing: ptxas does its own CSE. 8 of 15 open problems get the same SASS.
- It changes our own unroll choice for `m01_matmul_512`, with no change in time.
- `m02_matmul_rect` was 1–6% faster without it, with the same instructions in a different order. That may be
  noise.
