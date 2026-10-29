# Multiply-add fusion in our compiler

## Two builds, three outcomes

Each kernel is built two ways:

- **With fusion:** our compiler as it is. It writes one `fma.rn` for a multiply and an add where it can, and
  ptxas fuses what is left.
- **Without fusion:** with `--no_fusion`. Our compiler fuses nothing, but ptxas still fuses a plain `mul` and
  `add` into one fused multiply-add (FFMA) by itself (see [What ptxas does](#what-ptxas-does)).

So the comparison is our fusion against ptxas's. Each folder shows one outcome:

1. **ptxas fuses the same pairs.** `p02_scale_shift` has the same SASS in both builds, byte for byte. So do most
   kernels we timed.
2. **ptxas fuses about as well.** `a01_attention` has the same multiply-adds in both builds and is 1.04× slower
   without our fusion.
3. **Our fusion changes what ptxas does.** In `a03_noscale` our fusion keeps the PTX under a ptxas unroll limit.
   Without it, ptxas fuses the same multiply-adds, but leaves loops rolled and loads more than twice as much:
   2.59× slower.

Median kernel time on the runner, in µs (`tools/bench.py`, 5 rounds, best median). All six builds pass
`submit.py`.

| Kernel | With fusion | Without fusion |
|---|---|---|
| `p02_scale_shift` | 3.07 | 3.07 (1.00×) |
| `a01_attention` | 51.89 | 53.92 (1.04×) |
| `a03_noscale` | 98.99 | 256.74 (2.59×) |

## What is here

Each folder holds one problem's kernel:

| File | What it is |
|---|---|
| `<kernel>.ir.json`, `<kernel>.ir.txt` | The Kernel IR (JSON for the compiler, text to read) |
| `meta.json` | Launch shape and buffers; the compiler needs it beside the IR |
| `<kernel>.fusion.ptx` / `.sass` | With fusion |
| `<kernel>.nofusion.ptx` / `.sass` | Without fusion |
| `a03_noscale/cliff.py` | Un-fuses `a03_noscale`'s loops one at a time and shows what ptxas makes of each |

The SASS is for `sm_120` (the runner's RTX 5090), from `ptxas -arch=sm_120` and `cuobjdump -sass`.

The `a01_attention` fusion build has the same SASS as the `unroll` build in `../loop_unroll`. Its PTX differs only
in that `add`, `sub` and `mul` no longer carry `.rn`; `../loop_unroll` still holds the older PTX.

To regenerate one folder, for example `p02_scale_shift`:

```sh
cd p02_scale_shift
python ../../../src/compile.py k_add_pointwise.ir.json             > k_add_pointwise.fusion.ptx
python ../../../src/compile.py k_add_pointwise.ir.json --no_fusion > k_add_pointwise.nofusion.ptx
ptxas -arch=sm_120 k_add_pointwise.fusion.ptx -o /tmp/k.cubin && cuobjdump -sass /tmp/k.cubin > k_add_pointwise.fusion.sass
```

`ptxas` comes from the `nvidia-cuda-nvcc` wheel and `cuobjdump` from the `triton` wheel, both in `.venv`. To
time a folder's two builds, pass it to `tools/bench.py` with `--variant fusion=ptx:.fusion.ptx` and
`--variant nofusion=ptx:.nofusion.ptx`.

In the SASS, FFMA is one fused multiply-add; FMUL and FADD are a multiply and an add on their own. Instruction
counts leave out NOP padding. A "wait" is a point where a warp stops until earlier loads arrive, counted the way
`tools/sass_waits.py` counts them.

## p02_scale_shift: ptxas fuses the same pairs

4 blocks of 256 threads compute `x * 2.0 + 1.0`, 4 values per thread.

With fusion, our compiler writes 4 `fma.rn`. Without it, our compiler writes 4 `mul` and 4 `add`, and ptxas fuses
them into the same 4 FFMA. Both builds have the same SASS, byte for byte: 4 FFMA, 25 instructions, 18 registers.
Both run in 3.07 µs.

Most kernels we timed look like this; see [Other kernels we timed](#other-kernels-we-timed).

## a01_attention: ptxas fuses about as well

The same kernel as in `../loop_unroll`: 4 blocks of 128 threads. So few warps run, and memory latency is hard
to hide. The main loop runs over `a1` (64 steps, 2 per trip). Each step does four dot products of length 64,
a row of `q` against a column of `kt`: 256 multiply-adds. The online softmax adds 16 more per step.

Per trip of the main loop:

| Build | FFMA | FMUL | FADD | Global loads | Spill loads | Instructions | Waits |
|---|---|---|---|---|---|---|---|
| With fusion | 544 | 72 | 32 | 130 | 138 | 1,010 | 27 |
| Without fusion | 544 | 56 | 32 | 130 | 141 | 997 | 28 |

Without our fusion, ptxas fuses the same 544 multiply-adds per trip. It also keeps the 256 `q` values in
registers in both builds (with some spills; see `../loop_unroll`), so both do 130 global loads per trip.

One difference is left, and in instruction count it favors the build without fusion. The online softmax adds
`acc * alpha + beta * 1.0`, with the `1.0` in a register. Our compiler fuses the `beta * 1.0` product:
`fma.rn(beta, 1.0, acc * alpha)`, with `acc * alpha` as a separate `mul`. ptxas turns that fma back into an add,
`beta + acc * alpha`, but does not fuse the add with the `mul` again. Without our fusion, ptxas drops the
`* 1.0` and fuses `acc * alpha + beta` into one FFMA. That is 16 FMUL per trip, 8 per step.

The build without fusion is still 4% slower, in every run we made. Its loop has 3 more spill loads and 1 more
wait per trip, which may explain it. We did not confirm that.

## a03_noscale: our fusion changes what ptxas does

`softmax(q @ kt) @ v` with 64 × 64 inputs and no scaling, on 2 blocks of 256 threads. Each thread computes up to
6 rows and 2 columns of the output. The main loop runs over `a1` (64 steps, 2 per trip). Each step does six dot
products of length 64, one per row: a row of `q` against a column of `kt`. Then the online softmax updates 12
sums.

The 6 rows of `q` are 384 values per thread, too many to keep in registers across the main loop. So our compiler
re-reads them on every trip (`_rereads` in `src/README.md`, "Loop unrolling"). It emits each dot product as its
own loop, unrolled 8 times, and leaves the rest of the unrolling to ptxas. Both builds have the same loops in
their PTX; only the multiply-adds differ.

Per trip of the main loop. A loop that ptxas leaves rolled counts once per pass.

| Build | Registers | Loops in SASS | FFMA | FMUL | FADD | Global loads | Instructions | Waits |
|---|---|---|---|---|---|---|---|---|
| With fusion | 255 | 1 | 816 | 96 | 48 | 514 | 1,682 | 13 |
| Without fusion | 128 | 7 | 816 | 72 | 48 | 1,218 | 2,598 | 182 |

**With fusion.** ptxas unrolls all 12 dot-product loops of a trip. It then interleaves the 6 rows and shares loads
between the loops: each `kt` value once for all 6 rows, each `q` value once for both steps. 514 global loads and
13 waits per trip. No spills.

**Without fusion: 2.59× slower.** ptxas fuses the same 816 multiply-adds per trip. But it leaves 6 of the 12 loops
rolled (each unrolled 4 times, so 2 passes), gives the kernel 128 registers instead of 255, and no longer shares
the loads. 1,218 global loads and 182 waits per trip. The arithmetic is the same; the time goes to the loads and
waits.

### The cliff

`cliff.py` takes the fusion PTX and un-fuses the multiply-adds of the first n dot-product loops, in PTX order:
each `fma.rn` becomes a plain `mul` and `add`, as `--no_fusion` writes them. Then it shows what ptxas makes of
it. Counts are instructions in the SASS, not executed per trip.

| Un-fused loops | PTX lines | Registers | Loops in SASS | FFMA in SASS | LDG in SASS |
|---|---|---|---|---|---|
| 0 | 966 | 255 | 1 | 847 | 514 |
| 1 | 975 | 255 | 1 | 847 | 514 |
| 2 | 983 | 255 | 1 | 847 | 514 |
| 3 | 991 | 255 | 1 | 847 | 514 |
| 4 | 999 | 255 | 1 | 847 | 514 |
| 5 | 1007 | 255 | 1 | 847 | 514 |
| 6 | 1015 | 255 | 1 | 847 | 514 |
| 7 | 1023 | 128 | 2 | 815 | 898 |
| 8 | 1031 | 128 | 3 | 783 | 898 |
| 9 | 1039 | 128 | 4 | 751 | 898 |
| 10 | 1047 | 128 | 5 | 719 | 898 |
| 11 | 1055 | 128 | 6 | 687 | 898 |
| 12 | 1063 | 128 | 7 | 655 | 834 |

Un-fusing the first 6 loops (the first step's) changes nothing. With the 7th, 8 more PTX lines, ptxas leaves one
loop rolled, and the whole kernel switches: 128 registers, and 898 load instructions instead of 514. From there,
each un-fused loop adds one more rolled loop.

So the fusion build is fast because its PTX stays on the right side of a ptxas limit. We do not know ptxas's
rule; it is not documented. Any other change that grows this kernel's PTX could push the fusion build over too.
`a01_256` (closed) is the same kind of kernel and shows the same, smaller: see below.

We tried having our compiler fully unroll these loops itself. Fusion then stops mattering, but the kernel gets
2.4× slower: the rows run one after another, and ptxas spills. The fix we expect to work is to emit the shape
ptxas finds on its own: one loop that does all 12 dot products of a trip. We have not built it.

## Other kernels we timed

The first group is every open problem whose PTX has a fused `fma.rn`, and the closed reductions with a
multiply. The second group is the closed attention and softmax kernels, where our fusion changes the SASS. Same
method, 5 rounds, all in one run.

| Kernel | With fusion | Without fusion | Same SASS |
|---|---|---|---|
| `a02_attention_128` | 25.28 | 25.28 (1.00×) | no |
| `m02_matmul_rect` | 3.49 | 3.36 (0.96×) | yes |
| `m01_matmul_512` | 15.01 | 14.98 (1.00×) | yes |
| `f01_rmsnorm` | 4.70 | 4.32 (0.92×) | yes |
| `f02_row_softmax` | 3.01 | 3.01 (1.00×) | no |
| `f03_l2norm` | 3.01 | 3.01 (1.00×) | yes |
| `r05_sum_squares` (closed) | 3.02 | 3.04 (1.01×) | yes |
| `f01_rmsnorm_2048` (closed) | 4.80 | 4.80 (1.00×) | yes |
| `f04_l2norm_eps` (closed) | 3.01 | 3.01 (1.00×) | yes |
| `f06_rmsnorm_pow` (closed) | 3.10 | 3.07 (0.99×) | yes |
| `f08_layernorm_affine` (closed) | 4.80 | 4.80 (1.00×) | yes |
| `a00_small` (closed) | 12.99 | 12.99 (1.00×) | no |
| `a01_256` (closed) | 238.21 | 307.71 (1.29×) | no |
| `a02_rect` (closed) | 27.33 | 27.33 (1.00×) | no |
| `a04_batched` (closed) | 21.18 | 21.15 (1.00×) | no |
| `a05_heads` (closed) | 6.85 | 6.85 (1.00×) | no |
| `a06_long_kv` (closed) | 279.23 | 271.01 (0.97×) | no |
| `a07_tall_q` (closed) | 19.10 | 21.18 (1.11×) | no |
| `f00_softmax_1000` (closed) | 3.01 | 3.01 (1.00×) | no |
| `f03_softmax_narrow` (closed) | 2.98 | 2.98 (1.00×) | no |
| `f05_softmax_temp` (closed) | 3.01 | 3.01 (1.00×) | no |
| `f07_softmax_4d` (closed) | 3.01 | 3.01 (1.00×) | no |
| `f09_softmax_wide` (closed) | 4.77 | 4.80 (1.01×) | no |
| `f11_softmax_f16` (closed) | 2.98 | 2.98 (1.00×) | no |

Where the SASS is the same, any gap is noise. `f01_rmsnorm` shows how much: 8% with identical code here, and
4.74 against 4.74 in a re-run. `m02_matmul_rect` is not reliable at all: builds with the same SASS ran anywhere
from 3.2 to 4.7 µs in our runs.

What stands out:

- `a01_256` (1.29×) is the same kind of kernel as `a03_noscale`, with the same cause.
- `a07_tall_q` (1.11×) does not use that loop shape. We have not found its cause.
- `a06_long_kv` is 3% faster without our fusion, in every run.
- The rest run the same, within noise.

## Why fusing in PTX makes a kernel faster

A fused multiply-add computes `x * y + c` in one instruction. On this GPU an FFMA costs the same as one FMUL or
one FADD. So fusion:

1. **Halves the arithmetic** of a multiply followed by an add. A dot product of length 64 is 64 instructions
   instead of 128.
2. **Needs fewer registers.** The product never sits in a register of its own.
3. **Rounds once instead of twice.** The result is a little more exact. It is also a little different, which
   is fine for every problem here.

It does not make a sum of products wait less. In `acc = acc + x * y` the multiply does not depend on `acc`, so
it can run ahead. The adds into `acc` are the chain, one per value, with or without fusion. Fusion shortens the
chain only where the multiply waits on the previous result, as in `(a * b + c) * d + e`.

ptxas gets all of this by fusing on its own. So what our own fusion adds is mostly what it changes in ptxas's
choices of registers, loops and loads.

## How our compiler decides to fuse

When the compiler emits a float add, it looks at both operands. If one is the result of a `mul` of the same
type that it can still reuse, it emits one `fma.rn` from the multiply's operands and the other operand. "Can
still reuse" means the multiply was emitted in this block or in one that always runs before it, and neither of
its inputs has been written in place since.

This covers:

- an elementwise `add`, and `+` in an expression;
- an accumulator `add`, as in `acc <- add(acc, x * y)` in every dot product and sum of squares;
- `f32` and `f64` only.

The multiply stays in the PTX at first. At the end the compiler drops every result nothing reads, so a
multiply that only fed an fma goes away. A multiply that something else reads too stays, and both run.

Not fused by our compiler:

- A subtract, as in `a * b - c`. Only an add is matched. ptxas fuses it: in the closed `p56_scalar_chain`,
  4 `mul` and `sub` pairs become 4 FFMA.
- `f16` and `bf16`.
- Integer address math. It writes `mad.lo` directly; there is nothing to fuse.
- The `fma.rn` inside our `tanh` and `erf` approximations. They are part of the function, so they stay in
  both builds.

A product with a `1.0` operand is fused like any other. That can cost a multiply; see `a01_attention`.

`--no_fusion` makes the check always fail. Nothing else in our compiler changes. Loop unrolling decides the same
way in both builds, since it counts PTX lines before the unused multiplies are dropped. Every problem in
`problems/` and `problems/closed/` gets the same loops and branches in both builds.

## What ptxas does

ptxas fuses a multiply and an add by itself, unless one of them carries an explicit rounding mode such as `.rn`.
Our compiler writes plain `add`, `sub` and `mul` (round to nearest is their default), so ptxas fuses the pairs
our compiler leaves. That is why the build without fusion still has FFMA.

`fma`, `div`, `rcp`, `sqrt` and rounding `cvt` always carry `.rn`, because PTX requires a rounding mode on them.
ptxas does not fuse across our `fma.rn`: it turned `fma.rn(beta, 1.0, t)` into an add, but did not fuse that
add with the `mul` that makes `t`.

Both builds also have some FFMA that come from no multiply and add at all. ptxas writes them itself when it
expands an exact `rcp.rn` or `sqrt.rn` into a few refinement steps.

## Summary

- Without our fusion, ptxas fuses plain `mul` and `add` by itself. In most kernels it fuses the same pairs, and
  the SASS is the same (`p02_scale_shift`).
- Where arithmetic matters, ptxas fuses about as well as our compiler: 1.04× in `a01_attention`.
- Our fusion still pays in a few attention kernels, for an indirect reason: a smaller PTX keeps ptxas on the
  right side of its unroll limit. In `a03_noscale` that is 2.59×, all of it from loads and waits, none from
  arithmetic.
- That makes our fast `a03_noscale` build fragile. A compiler that emits the shape ptxas finds on its own would
  not need luck, and would probably not need our fusion either.
- Our fusion picks the worse product when one operand is `1.0`. That costs 8 FMUL per step in `a01_attention`.
