# Loop unrolling in our compiler

## What is here

Each folder holds one problem's kernel, compiled two ways by `src/compile.py`:

| File | What it is |
|---|---|
| `<kernel>.ir.json`, `<kernel>.ir.txt` | The Kernel IR (JSON for the compiler, text to read) |
| `meta.json` | Launch shape and buffers; the compiler needs it beside the IR |
| `<kernel>.unroll.ptx` / `.sass` | Our compiler as it is: loops unrolled where it pays |
| `<kernel>.nounroll.ptx` / `.sass` | Same, with `--no_loop_unroll`: every loop a runtime loop, one body per trip |

The SASS is for `sm_120` (the runner's RTX 5090), from `ptxas -arch=sm_120` and `cuobjdump -sass`.

To regenerate one folder, for example `r02_big_sum`:

```sh
cd r02_big_sum
python ../../../src/compile.py k_sum_1_reduce.ir.json                  > k_sum_1_reduce.unroll.ptx
python ../../../src/compile.py k_sum_1_reduce.ir.json --no_loop_unroll > k_sum_1_reduce.nounroll.ptx
ptxas -arch=sm_120 k_sum_1_reduce.unroll.ptx -o /tmp/k.cubin && cuobjdump -sass /tmp/k.cubin > k_sum_1_reduce.unroll.sass
```

`ptxas` comes from the `nvidia-cuda-nvcc` wheel and `cuobjdump` from the `triton` wheel, both in `.venv`.

## Results

Median kernel time on the runner, in µs (`tools/bench.py`, 3 rounds, best median). All variants pass
`submit.py`.

| Kernel | Unroll | No unroll | Slowdown | Registers (unroll / no unroll) |
|---|---|---|---|---|
| `r02_big_sum` | 6.85 | 21.15 | 3.09× | 38 / 18 |
| `a01_attention` | 51.87 | 82.62 | 1.59× | 255 (316 B spilled) / 160 |
| `m01_matmul_512` | 15.04 | 15.04 | 1.00× | 46 / 40 |

What the SASS shows (a "wait" is a point where a warp stops until earlier loads arrive; counted with
`tools/sass_waits.py`):

| Kernel | Unroll | No unroll |
|---|---|---|
| `r02_big_sum` | No loop. 512 loads, 21 waits in total | 4 loads per trip, 1 wait per trip, 128 trips: 128 waits |
| `a01_attention` | `q` (256 values) loaded once, before the loop. 65 global loads per step of `a1` | `q` reloaded every step: 321 global loads per step of `a1`, 64 steps |
| `m01_matmul_512` | No loop. 792 shared-memory loads, 134 waits | 8 trips of 99 shared-memory loads, 23 waits each |

### r02_big_sum

One block of 128 threads sums 65,536 floats. Each thread adds 512 values, 128 apart.

With one block there are only 4 warps on one SM. When a warp waits on memory, no other warp can fill the
gap. So the only way to hide memory latency is to have many loads of the *same* thread in flight at once.

Our unrolled PTX is 512 loads in a straight line. ptxas puts about 25 of them in flight before the first add
needs one. The rolled loop gives ptxas a loop whose trip count depends on the thread index. It unrolls that
loop only 4 times. So each thread waits on memory 128 times instead of 21. That is the 3× slowdown.

### a01_attention

4 blocks of 128 threads: again very few warps, so again latency is hard to hide. The main loop runs over `a1`
(64 steps). Inside it, four inner loops of 64 trips each read `q` and `kt`.

The `q` values do not change across `a1`. In the unrolled build, ptxas loads all 256 of them once, before the
loop, and keeps them in registers. That costs 255 registers and some spills to local memory. But each step
of `a1` then does 65 global loads instead of 321. Fewer loads win: 1.59× faster.

In the rolled build ptxas still unrolls the inner loops fully. But it reloads `q` on every step of `a1`. We
do not know why ptxas decides differently in the two cases.

### m01_matmul_512

No difference. The inner loop reads shared memory, which is fast. The `cp.async` copies already overlap global
loads with compute. In the rolled build ptxas unrolls the 64-trip inner loop fully by itself. Our full unroll
gives it nothing more to work with.

## Why unrolling in PTX makes a kernel faster

A GPU hides memory latency in two ways:

1. **Other warps.** While one warp waits, the SM runs another. This needs many warps per SM.
2. **Independent loads in one thread.** A warp only stops when it *uses* a loaded value. If it issues 20 loads
   before using the first, it pays the latency about once, not 20 times.

The small problems here launch very few warps, so (1) does not help. Only (2) is left, and (2) needs the loads
to be visible side by side, which is what unrolling does.

Unrolling in our compiler also helps in two smaller ways:

- **Constant offsets.** Our compiler knows the thread index range, so it knows no index can overflow. Copy *j*
  of a load becomes one base address plus a constant: `ld.global.nc.f32 %f2, [%rd5+512]` in PTX and
  `LDG.E.CONSTANT R14, desc[UR6][R2.64+0x200]` in SASS. One instruction per load.
- **No loop overhead.** No counter, compare or branch per trip.

The cost is code size. On the runner, 6.2K PTX instructions ran fine and 16K or more ran much slower. So our
compiler caps a full unroll at 8K and otherwise falls back to a loop with several body copies per trip.

## How our compiler decides to unroll

The compiler looks at each loop on its own, outer loops first. It takes the first case that fits:

1. **Threads run different trip counts.** For example, the loop starts at the thread index and the counts
   differ by more than one trip, or a barrier sits in the body and the counts differ at all. Then it emits a
   runtime loop with one body per trip.
2. **At most one trip.** It emits the body once, with no loop.
3. **Too many values re-read on every outer trip.** Some loads give the same values on every trip of an
   outer runtime loop. If there are more of them than half a thread's register budget (127 for 128-thread
   blocks), keeping them in registers would spill. So the inner loops that read them stay loops, with 8 body
   copies per trip, and ptxas decides the rest.
4. **Any other loop.** It tries a full unroll, one copy at a time, and stops once it passes 8,192 PTX
   instructions. If every trip fit, the full unroll stays. If not, it throws the attempt away and emits a loop
   instead: 2 copies per trip for a loop that holds other loops, otherwise as many copies as fit in 8,192.

When the result is a loop with several copies per trip, the trips every thread runs go in a loop that counts
from 0, so ptxas sees the trip count. The few trips that only some threads run follow after it as guarded
copies.

`--no_loop_unroll` forces case 1 for every loop.

The numbers 8,192, 8 and 2 were each measured on the runner as the best single choice.

What each kernel here gets:

| Kernel | Case | PTX |
|---|---|---|
| `r02_big_sum` | 4, fits | No loop. 512 loads in a straight line, ~1.1K instructions |
| `m01_matmul_512` | 4, fits | No loop. ~4.2K instructions |
| `a01_attention` | 4, too big, for `a1` / 3 for the inner loops | `a1` loop with 2 copies per trip. Each copy has four inner loops of 8 copies per trip. `q` is 256 values, more than 127 |

## Where ptxas falls short

ptxas unrolls loops too. The question is what it does with a rolled loop that it could unroll itself.

**With a hidden trip count, it barely unrolls.** The `--no_loop_unroll` PTX starts each thread's loop at its
thread index. ptxas cannot tell how many trips that is. For `r02_big_sum` it unrolls 4 times and adds a
remainder loop (`k_sum_1_reduce.nounroll.sass`, the loop at `0xe0`–`0x1f0`).

**With a known trip count, it does better, but not as well as we do.** As a check, we built a third version
of `r02_big_sum` (not in this folder): a rolled loop counting `0..512`, so ptxas sees 512 trips. That PTX
loop body is:

```
$L_loop_1:
	mad.lo.s32 %r7, %r5, 128, %r6;
	mul.wide.u32 %rd4, %r7, 4;
	add.s64 %rd5, %rd1, %rd4;
	ld.global.nc.f32 %f1, [%rd5+0];
	add.rn.f32 %f0, %f0, %f1;
	add.s32 %r5, %r5, 1;
	setp.lt.s32 %p0, %r5, 512;
	@%p0 bra $L_loop_1;
```

ptxas unrolls it 64 times, with 3 waits per 64 loads. But it computes every address from scratch, because a
32-bit index could wrap and it cannot prove it does not:

```
IADD3 R11, PT, PT, R3.reuse, 0x80, RZ ;
IMAD.WIDE.U32 R10, R11, 0x4, R16.reuse ;
LDG.E.CONSTANT R10, desc[UR6][R10.64] ;
```

That is three instructions per load, against one in our unrolled SASS. It runs in 8.90 µs: better than the
rolled 21.15 µs, still 1.3× slower than our 6.85 µs.

The same known-count version gives `a01_attention` 47.81 µs and `m01_matmul_512` 21.15 µs. So for
`a01_attention` our current unrolling is *not* the best choice: a known-count rolled loop beats it by 8%. For
`m01_matmul_512` that version is 1.41× slower than both builds in this folder.

## Summary

- Unrolling matters most where few warps run: `r02_big_sum` (3.1×) and `a01_attention` (1.6×).
- Most of the loss without unrolling comes from ptxas seeing no trip count. Even with one, ptxas leaves about
  30% on the table for `r02_big_sum`, from its address math.
- ptxas does fine on `m01_matmul_512`, where loads come from shared memory.
