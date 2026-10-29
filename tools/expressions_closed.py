"""A GUESSED stand-in for the CLOSED problem set: torch expressions per category.

The real hidden set lives outside the repo. These are our guesses at it, built from
research/hidden_problems.md: same categories as tools/expressions_open.py, different
shapes/sizes/op mixes. Category 1 (pointwise): core ops (add/sub/mul/div/max/min/relu/exp/sqrt)
first, then stretch ops (neg, square, abs, rsqrt, log, sin/cos, tanh, sigmoid, silu, gelu)
that probe gaps in the Assign op table, then the same kinds of ops in bf16, f16 and f64, then
more shapes, constants and LLM-style ops (swiglu, pow, where, clamp). Category 2 (reductions):
full and per-row/column sum/mean/max/min, most pinned to one block of 128 cooperating threads,
some at other block widths and reduce plans (coop-t, r2, grid atomic, grid kernel), some
unpinned. Category 3 (matmul): square, odd, tall, deep-K, batched and `F.linear` shapes.
Category 4 (fused): layer/RMS norm, softmax, log-softmax, logsumexp. Category 5: attention.

Each spec:
  id        problem folder name (category prefix orders them)
  code      the expression handed to `emmy compile -c` (this defines the Kernel IR)
  ref       numpy function computing the expected output(s) from the seeded inputs,
            taking arrays in the order of the traced graph's `inputs` list
  rtol/atol correctness tolerance

Rules when adding expressions:
  - `x.T` does not trace - pre-transpose instead.
  - Constants (e.g. `* 2.0`) become ConstantOp kernel args; resolved automatically.
  - Trace every new expression once via tools/generate.py before committing it.
"""

import math

import numpy as np

_erf = np.vectorize(math.erf, otypes=[np.float64])
f32, f16 = np.float32, np.float16


def _d(x):
    return x.astype(np.float64)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _red(work="t128", reduce="coop"):
    return f"WORK={work},REDUCE={reduce}"


def _attn(q, kt, v, scale):
    return [(_softmax(_d(q) @ _d(kt) * scale) @ _d(v)).astype(f32)]

CLOSED = [
    # ---- category 1: pointwise -------------------------------------------
    # Smaller than open p00 -> simplest IR, with sub instead of add.
    dict(id="p00_small_sub",
         code="torch.randn(64) - torch.randn(64)",
         ref=lambda a, b: [a - b], rtol=0, atol=0),
    # Size not a multiple of the thread count -> bounds guard from the Tile's N.
    dict(id="p01_odd_add",
         code="torch.randn(1000) + torch.randn(1000)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    # Even but not divisible by 4 -> likely v2 loads/stores.
    dict(id="p02_v2_mul",
         code="torch.randn(1538) * torch.randn(1538)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="p03_div",
         code="x = torch.randn(2048); y = torch.randn(2048); x / y",
         ref=lambda x, y: [x / y], rtol=1e-6, atol=1e-6),
    dict(id="p04_max_min",
         code="x = torch.randn(4096); y = torch.randn(4096); "
              "torch.maximum(x, y) - torch.minimum(x, y)",
         ref=lambda x, y: [np.maximum(x, y) - np.minimum(x, y)], rtol=0, atol=0),
    dict(id="p05_add_2d",
         code="torch.randn(64, 128) + torch.randn(64, 128)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="p06_bias_bcast",
         code="x = torch.randn(64, 256); b = torch.randn(256); x + b",
         ref=lambda x, b: [x + b], rtol=0, atol=0),
    dict(id="p07_fma3",
         code="a = torch.randn(8192); b = torch.randn(8192); c = torch.randn(8192); a * b + c",
         ref=lambda a, b, c: [a * b + c], rtol=1e-6, atol=1e-6),
    # Two ConstantOps, an odd size and relu together.
    dict(id="p08_relu_affine_odd",
         code="torch.relu(torch.randn(3000) * 0.5 - 0.25)",
         ref=lambda x: [np.maximum(x * np.float32(0.5) - np.float32(0.25), 0)],
         rtol=1e-6, atol=1e-6),
    dict(id="p09_exp",
         code="torch.exp(torch.randn(4096))",
         ref=lambda x: [np.exp(x)], rtol=1e-5, atol=1e-5),
    dict(id="p10_inv_norm",
         code="x = torch.randn(2048); 1.0 / torch.sqrt(x * x + 1.0)",
         ref=lambda x: [np.float32(1.0) / np.sqrt(x * x + np.float32(1.0))], rtol=1e-5, atol=1e-5),
    # Many blocks.
    dict(id="p11_big_add",
         code="torch.randn(1 << 20) + torch.randn(1 << 20)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),

    # ---- category 1b: pointwise stretch ops ------------------------------
    dict(id="p12_neg",
         code="x = torch.randn(1024); -x",  # a leading "-" would parse as a CLI flag
         ref=lambda x: [-x], rtol=0, atol=0),
    dict(id="p13_square",
         code="torch.square(torch.randn(2048))",
         ref=lambda x: [x * x], rtol=0, atol=0),
    dict(id="p14_abs",
         code="torch.abs(torch.randn(2048))",
         ref=lambda x: [np.abs(x)], rtol=0, atol=0),
    # x * x + 1.0 keeps the argument positive.
    dict(id="p15_rsqrt",
         code="x = torch.randn(2048); torch.rsqrt(x * x + 1.0)",
         ref=lambda x: [np.float32(1.0) / np.sqrt(x * x + np.float32(1.0))], rtol=1e-5, atol=1e-5),
    dict(id="p16_log",
         code="x = torch.randn(2048); torch.log(x * x + 1.0) * 0.5",  # a bare log output shadows log() in .ref.cu
         ref=lambda x: [np.log(x * x + np.float32(1.0)) * np.float32(0.5)], rtol=1e-5, atol=1e-5),
    dict(id="p17_sin_cos",
         code="x = torch.randn(4096); y = torch.randn(4096); torch.sin(x) + torch.cos(y)",
         ref=lambda x, y: [np.sin(x) + np.cos(y)], rtol=1e-5, atol=1e-5),
    dict(id="p18_tanh",
         code="torch.tanh(torch.randn(4096))",
         ref=lambda x: [np.tanh(x)], rtol=1e-5, atol=1e-5),
    dict(id="p19_sigmoid",
         code="torch.sigmoid(torch.randn(4096))",
         ref=lambda x: [np.float32(1.0) / (np.float32(1.0) + np.exp(-x))], rtol=1e-5, atol=1e-5),
    # Odd size again, so the guard path sees a non-trivial op.
    dict(id="p20_silu_odd",
         code="torch.nn.functional.silu(torch.randn(3000))",
         ref=lambda x: [x / (np.float32(1.0) + np.exp(-x))], rtol=1e-5, atol=1e-5),
    dict(id="p21_gelu",
         code="torch.nn.functional.gelu(torch.randn(4096))",
         ref=lambda x: [(0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))).astype(np.float32)],
         rtol=1e-5, atol=1e-5),
    # emmy decomposes the tanh approximation into primitive ops plus tanh.
    dict(id="p22_gelu_tanh",
         code='torch.nn.functional.gelu(torch.randn(4096), approximate="tanh")',
         ref=lambda x: [(0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi)
                                                  * (x + 0.044715 * x.astype(np.float64) ** 3))))
                        .astype(np.float32)],
         rtol=1e-5, atol=1e-5),
    # 2-D, two different stretch ops in one kernel.
    dict(id="p23_sigmoid_gate",
         code="x = torch.randn(64, 256); y = torch.randn(64, 256); torch.sigmoid(x) * torch.tanh(y)",
         ref=lambda x, y: [np.float32(1.0) / (np.float32(1.0) + np.exp(-x)) * np.tanh(y)],
         rtol=1e-5, atol=1e-5),

    # ---- category 1c: other dtypes ---------------------------------------
    # bf16 refs compute in f32; tools/generate.py rounds bf16 inputs and outputs.
    dict(id="p24_add_f64",
         code="torch.randn(4096, dtype=torch.float64) + torch.randn(4096, dtype=torch.float64)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="p25_fma_f64_odd",
         code="a = torch.randn(3000, dtype=torch.float64); b = torch.randn(3000, dtype=torch.float64); "
              "c = torch.randn(3000, dtype=torch.float64); a * b + c",
         ref=lambda a, b, c: [a * b + c], rtol=1e-12, atol=1e-12),
    # PTX has no f64 ex2: needs a software exp.
    dict(id="p26_exp_f64",
         code="torch.exp(torch.randn(4096, dtype=torch.float64))",
         ref=lambda x: [np.exp(x)], rtol=1e-10, atol=1e-10),
    dict(id="p27_add_bf16",
         code="torch.randn(2048, dtype=torch.bfloat16) + torch.randn(2048, dtype=torch.bfloat16)",
         ref=lambda a, b: [a + b], rtol=1e-2, atol=1e-2),
    dict(id="p28_relu_affine_bf16_odd",
         code="torch.relu(torch.randn(3000, dtype=torch.bfloat16) * 0.5 - 0.25)",
         ref=lambda x: [np.maximum(x * np.float32(0.5) - np.float32(0.25), 0)],
         rtol=1e-2, atol=1e-2),
    dict(id="p29_sigmoid_bf16",
         code="torch.sigmoid(torch.randn(4096, dtype=torch.bfloat16))",
         ref=lambda x: [np.float32(1.0) / (np.float32(1.0) + np.exp(-x))], rtol=1e-2, atol=1e-2),
    dict(id="p30_add_f16",
         code="torch.randn(2048, dtype=torch.float16) + torch.randn(2048, dtype=torch.float16)",
         ref=lambda a, b: [a + b], rtol=1e-3, atol=1e-3),
    dict(id="p31_tanh_f16",
         code="torch.tanh(torch.randn(4096, dtype=torch.float16))",
         ref=lambda x: [np.tanh(x.astype(np.float32)).astype(np.float16)], rtol=1e-3, atol=1e-3),
    # Mixed dtypes: a bf16 -> f32 cast feeding an f32 multiply.
    dict(id="p32_bf16_to_f32",
         code="x = torch.randn(4096, dtype=torch.bfloat16); y = torch.randn(4096); x.float() * y",
         ref=lambda x, y: [x * y], rtol=1e-6, atol=1e-6),

    # ---- category 1d: more pointwise -------------------------------------
    dict(id="p33_sub_2d",
         code="x = torch.randn(64, 128); y = torch.randn(64, 128); x - y",
         ref=lambda x, y: [x - y], rtol=0, atol=0),
    dict(id="p34_sigmoid",
         code="torch.sigmoid(torch.randn(2048))",
         ref=lambda x: [(1 / (1 + np.exp(-_d(x)))).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="p35_tanh",
         code="torch.tanh(torch.randn(2048))",
         ref=lambda x: [np.tanh(_d(x)).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="p36_silu",
         code="x = torch.randn(4096); x * torch.sigmoid(x)",
         ref=lambda x: [(_d(x) / (1 + np.exp(-_d(x)))).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="p37_div",
         code="a = torch.randn(1024); b = torch.randn(1024); a / (b.abs() + 1.0)",
         ref=lambda a, b: [a / (np.abs(b) + f32(1.0))], rtol=1e-6, atol=1e-6),
    dict(id="p38_bias_bcast",
         code="x = torch.randn(32, 256); b = torch.randn(256); x + b",
         ref=lambda x, b: [x + b], rtol=0, atol=0),
    dict(id="p39_odd_mul",
         code="torch.randn(1000) * torch.randn(1000)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="p40_odd_add_const",
         code="torch.randn(999) + 1.5",
         ref=lambda x: [x + f32(1.5)], rtol=0, atol=0),
    dict(id="p41_maximum",
         code="torch.maximum(torch.randn(2048), torch.randn(2048))",
         ref=lambda a, b: [np.maximum(a, b)], rtol=0, atol=0),
    dict(id="p42_sqrt_abs",
         code="torch.sqrt(torch.randn(2048).abs())",
         ref=lambda x: [np.sqrt(np.abs(x))], rtol=1e-6, atol=1e-6),
    dict(id="p43_square_shift",
         code="x = torch.randn(4096); (x - 0.5) * (x - 0.5)",
         ref=lambda x: [(x - f32(0.5)) * (x - f32(0.5))], rtol=1e-6, atol=1e-6),
    dict(id="p44_leaky_where",
         code="x = torch.randn(1024); torch.where(x > 0, x, 0.1 * x)",
         ref=lambda x: [np.where(x > 0, x, f32(0.1) * x)], rtol=1e-6, atol=1e-6),
    dict(id="p45_rsqrt",
         code="torch.rsqrt(torch.randn(1024).abs() + 1.0)",
         ref=lambda x: [(1 / np.sqrt(np.abs(_d(x)) + 1.0)).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="p46_log",
         code="torch.log(torch.randn(1024).abs() + 1.0)",
         ref=lambda x: [np.log(np.abs(_d(x)) + 1.0).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="p47_gelu",
         code="torch.nn.functional.gelu(torch.randn(2048))",
         ref=lambda x: [(0.5 * _d(x) * (1 + _erf(_d(x) / np.sqrt(2.0)))).astype(f32)],
         rtol=1e-4, atol=1e-5),
    dict(id="p48_3d_mul",
         code="torch.randn(4, 16, 64) * torch.randn(4, 16, 64)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="p49_fma3",
         code="a = torch.randn(512); b = torch.randn(512); c = torch.randn(512); a * b + c",
         ref=lambda a, b, c: [a * b + c], rtol=1e-6, atol=1e-6),
    dict(id="p50_clamp",
         code="torch.clamp(torch.randn(2048), -0.5, 0.5)",
         ref=lambda x: [np.clip(x, f32(-0.5), f32(0.5))], rtol=0, atol=0),
    dict(id="p51_neg_abs",
         code="x = torch.randn(3000); -x.abs()",
         ref=lambda x: [-np.abs(x)], rtol=0, atol=0),
    dict(id="p52_swiglu",
         code="x = torch.randn(64, 256); y = torch.randn(64, 256); torch.nn.functional.silu(x) * y",
         ref=lambda x, y: [(_d(x) * _sigmoid(_d(x)) * _d(y)).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="p53_sig_tanh",
         code="x = torch.randn(4, 64); torch.sigmoid(x) * torch.tanh(x)",
         ref=lambda x: [(_sigmoid(_d(x)) * np.tanh(_d(x))).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="p54_residual_scale",
         code="x = torch.randn(2, 16, 64); r = torch.randn(2, 16, 64); w = torch.randn(64); (x + r) * w",
         ref=lambda x, r, w: [(x + r) * w], rtol=0, atol=0),
    dict(id="p55_f16_add",
         code="torch.randn(1024, dtype=torch.float16) + torch.randn(1024, dtype=torch.float16)",
         ref=lambda a, b: [(a + b).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="p56_scalar_chain",
         code="x = torch.randn(4096); (x * 0.5 + 0.25) * 3.0 - 1.0",
         ref=lambda x: [(x * f32(0.5) + f32(0.25)) * f32(3.0) - f32(1.0)], rtol=1e-6, atol=1e-6),
    dict(id="p57_pow",
         code="torch.pow(torch.randn(2048).abs() + 0.5, 1.5)",
         ref=lambda x: [((np.abs(_d(x)) + 0.5) ** 1.5).astype(f32)], rtol=1e-4, atol=1e-5),
    dict(id="p58_rsqrt_eps",
         code="1.0 / torch.sqrt(torch.randn(1024).abs() + 1e-3)",
         ref=lambda x: [(1 / np.sqrt(np.abs(_d(x)) + 1e-3)).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="p59_large",
         code="torch.randn(1 << 20) * 2.0",
         ref=lambda x: [x * f32(2.0)], rtol=0, atol=0),
    dict(id="p60_tiny",
         code="torch.randn(7) + torch.randn(7)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="p61_f16_silu",
         code="x = torch.randn(2048, dtype=torch.float16); torch.nn.functional.silu(x)",
         ref=lambda x: [(_d(x) * _sigmoid(_d(x))).astype(f16).astype(f32)], rtol=2e-3, atol=2e-3),

    # ---- category 2: reductions ------------------------------------------
    # Pinned to one block of 128 cooperating threads, as in the open set, unless the id says
    # "unpinned" or the knobs pick another width / reduce plan.
    dict(id="r00_sum_2048",
         code="torch.randn(2048).sum()",
         ref=lambda x: [_d(x).sum().astype(f32)], rtol=1e-4, atol=1e-4, knobs=_red()),
    dict(id="r01_sum_odd",
         code="torch.randn(1000).sum()",
         ref=lambda x: [_d(x).sum().astype(f32)], rtol=1e-4, atol=1e-4, knobs=_red()),
    dict(id="r02_row_mean",
         code="torch.randn(32, 1024).mean(-1)",
         ref=lambda x: [_d(x).mean(-1).astype(f32)], rtol=1e-4, atol=1e-4, knobs=_red()),
    dict(id="r03_row_amax",
         code="torch.randn(128, 256).amax(-1)",
         ref=lambda x: [x.max(-1)], rtol=0, atol=0, knobs=_red()),
    dict(id="r04_row_sum_odd",
         code="torch.randn(16, 300).sum(-1)",
         ref=lambda x: [_d(x).sum(-1).astype(f32)], rtol=1e-4, atol=1e-4, knobs=_red()),
    dict(id="r05_sum_squares",
         code="x = torch.randn(8192); (x * x).sum()",
         ref=lambda x: [(_d(x) ** 2).sum().astype(f32)], rtol=1e-4, atol=1e-3, knobs=_red()),
    dict(id="r06_col_sum",
         code="torch.randn(256, 64).sum(0)",
         ref=lambda x: [_d(x).sum(0).astype(f32)], rtol=1e-4, atol=1e-4, knobs=_red()),
    dict(id="r07_sum_unpinned",
         code="torch.randn(65536).sum()",
         ref=lambda x: [_d(x).sum().astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="r08_row_sum_short",
         code="torch.randn(64, 32).sum(-1)",
         ref=lambda x: [_d(x).sum(-1).astype(f32)], rtol=1e-4, atol=1e-4, knobs=_red()),
    dict(id="r09_row_sum_unpinned",
         code="torch.randn(256, 512).sum(-1)",
         ref=lambda x: [_d(x).sum(-1).astype(f32)], rtol=1e-4, atol=1e-4),
    dict(id="r10_t256",
         code="torch.randn(4096).sum()",
         ref=lambda x: [_d(x).sum().astype(f32)], rtol=1e-4, atol=1e-4, knobs=_red("t256")),
    dict(id="r11_t64_rows",
         code="torch.randn(64, 512).sum(-1)",
         ref=lambda x: [_d(x).sum(-1).astype(f32)], rtol=1e-4, atol=1e-4, knobs=_red("t64")),
    dict(id="r12_t32_rows",
         code="torch.randn(128, 128).sum(-1)",
         ref=lambda x: [_d(x).sum(-1).astype(f32)], rtol=1e-4, atol=1e-4, knobs=_red("t32")),
    dict(id="r13_t512_mean",
         code="torch.randn(65536).mean()",
         ref=lambda x: [_d(x).mean().astype(f32)], rtol=1e-4, atol=1e-4, knobs=_red("t512")),
    dict(id="r14_coop_t",
         code="torch.randn(64, 512).sum(-1)",
         ref=lambda x: [_d(x).sum(-1).astype(f32)], rtol=1e-4, atol=1e-4,
         knobs=_red("t128", "coop-t")),
    dict(id="r15_coop_r2",
         code="torch.randn(8192).sum()",
         ref=lambda x: [_d(x).sum().astype(f32)], rtol=1e-4, atol=1e-4,
         knobs=_red("t128", "coop/r2")),
    dict(id="r16_grid_atomic",
         code="torch.randn(65536).sum()",
         ref=lambda x: [_d(x).sum().astype(f32)], rtol=1e-3, atol=1e-3,
         knobs=_red("t128", "g4a/coop")),
    dict(id="r17_grid_kernel",
         code="torch.randn(32768).sum()",
         ref=lambda x: [_d(x).sum().astype(f32)], rtol=1e-3, atol=1e-3,
         knobs=_red("t128", "g2k/coop")),
    dict(id="r18_amax_4d",
         code="torch.randn(2, 4, 8, 256).amax(-1)",
         ref=lambda x: [x.max(-1)], rtol=0, atol=0, knobs=_red()),
    dict(id="r19_mean_mid_axis",
         code="torch.randn(16, 64, 32).mean(1)",
         ref=lambda x: [_d(x).mean(1).astype(f32)], rtol=1e-4, atol=1e-4, knobs=_red()),
    dict(id="r20_sig_tanh_sum",
         code="(lambda x: (torch.sigmoid(x) * torch.tanh(x)).sum(-1))(torch.randn(4, 64))",
         ref=lambda x: [(_sigmoid(_d(x)) * np.tanh(_d(x))).sum(-1).astype(f32)],
         rtol=1e-4, atol=1e-4),
    dict(id="r21_f16_rowsum",
         code="torch.randn(32, 256, dtype=torch.float16).sum(-1)",
         ref=lambda x: [_d(x).sum(-1).astype(f32)], rtol=2e-3, atol=2e-2, knobs=_red()),

    # ---- category 3: matmul ----------------------------------------------
    dict(id="m00_256x128x256",
         code="torch.randn(256, 128) @ torch.randn(128, 256)",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="m01_64sq",
         code="torch.randn(64, 64) @ torch.randn(64, 64)",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="m02_odd",
         code="torch.randn(96, 80) @ torch.randn(80, 48)",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="m03_tall",
         code="torch.randn(1024, 64) @ torch.randn(64, 32)",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="m04_deep_k",
         code="torch.randn(32, 1024) @ torch.randn(1024, 128)",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="m05_bias",
         code="a = torch.randn(128, 128); b = torch.randn(128, 128); c = torch.randn(128); a @ b + c",
         ref=lambda a, b, c: [(_d(a) @ _d(b) + _d(c)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="m06_1024sq",
         code="torch.randn(1024, 1024) @ torch.randn(1024, 1024)",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="m07_linear_b1",
         code="x = torch.randn(1, 64); w = torch.randn(32, 64); torch.nn.functional.linear(x, w)",
         ref=lambda x, w: [(_d(x) @ _d(w).T).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="m08_linear_b5",
         code="x = torch.randn(5, 96); w = torch.randn(72, 96); torch.nn.functional.linear(x, w)",
         ref=lambda x, w: [(_d(x) @ _d(w).T).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="m09_matvec_wide",
         code="torch.matmul(torch.randn(1, 1024), torch.randn(1024, 3072))",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="m10_bmm",
         code="torch.bmm(torch.randn(4, 64, 32), torch.randn(4, 32, 48))",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="m11_linear_deep",
         code="x = torch.randn(1, 4096); w = torch.randn(512, 4096); torch.nn.functional.linear(x, w)",
         ref=lambda x, w: [(_d(x) @ _d(w).T).astype(f32)], rtol=1e-3, atol=2e-3),
    dict(id="m12_linear_bias",
         code="x = torch.randn(16, 128); w = torch.randn(64, 128); b = torch.randn(64); "
              "torch.nn.functional.linear(x, w, b)",
         ref=lambda x, w, b: [(_d(x) @ _d(w).T + _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="m13_matmul_relu",
         code="torch.relu(torch.randn(128, 64) @ torch.randn(64, 128))",
         ref=lambda a, b: [np.maximum(_d(a) @ _d(b), 0).astype(f32)], rtol=1e-3, atol=1e-3),
    # Not here: an fp16 matmul, which Emmy schedules onto tensor cores (RegFragment / mma.sync),
    # a statement family the task excludes from every problem.

    # ---- category 4: fused -----------------------------------------------
    dict(id="f00_softmax_1000",
         code="torch.softmax(torch.randn(32, 1000), dim=-1)",
         ref=lambda x: [_softmax(_d(x)).astype(f32)], rtol=1e-3, atol=1e-5),
    dict(id="f01_rmsnorm_2048",
         code="x = torch.randn(16, 2048); w = torch.randn(2048); "
              "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w",
         ref=lambda x, w: [(_d(x) / np.sqrt((_d(x) ** 2).mean(-1, keepdims=True) + 1e-6) * w)
                           .astype(f32)],
         rtol=1e-3, atol=1e-4),
    dict(id="f02_center",
         code="x = torch.randn(64, 256); x - x.mean(-1, keepdim=True)",
         ref=lambda x: [(_d(x) - _d(x).mean(-1, keepdims=True)).astype(f32)], rtol=1e-3, atol=1e-5),
    dict(id="f03_softmax_narrow",
         code="torch.softmax(torch.randn(128, 64), dim=-1)",
         ref=lambda x: [_softmax(_d(x)).astype(f32)], rtol=1e-3, atol=1e-5),
    dict(id="f04_l2norm_eps",
         code="x = torch.randn(32, 768); x / (torch.sqrt((x * x).sum(-1, keepdim=True)) + 1e-6)",
         ref=lambda x: [(_d(x) / (np.sqrt((_d(x) ** 2).sum(-1, keepdims=True)) + 1e-6)).astype(f32)],
         rtol=1e-3, atol=1e-4),
    dict(id="f05_softmax_temp",
         code="x = torch.randn(64, 512); torch.softmax(x * 0.5, dim=-1)",
         ref=lambda x: [_softmax(_d(x) * 0.5).astype(f32)], rtol=1e-3, atol=1e-5),
    dict(id="f06_rmsnorm_pow",
         code="x = torch.randn(2,64,256); "
              "torch.mul(x, torch.pow(torch.mean(torch.pow(x,2),-1,keepdim=True)+1e-6, -0.5))",
         ref=lambda x: [(_d(x) / np.sqrt((_d(x) ** 2).mean(-1, keepdims=True) + 1e-6)).astype(f32)],
         rtol=1e-3, atol=1e-4),
    dict(id="f07_softmax_4d",
         code="torch.softmax(torch.randn(2, 4, 16, 128), dim=-1)",
         ref=lambda x: [_softmax(_d(x)).astype(f32)], rtol=1e-3, atol=1e-5),
    # a lambda, not `d = x - m`: the tracer turns every named tensor into a graph input
    dict(id="f08_layernorm_affine",
         code="(lambda x, w, b: (lambda d: d / torch.sqrt((d * d).mean(-1, keepdim=True) + 1e-5) * w + b)"
              "(x - x.mean(-1, keepdim=True)))(torch.randn(32, 256), torch.randn(256), torch.randn(256))",
         ref=lambda x, w, b: [((_d(x) - _d(x).mean(-1, keepdims=True))
                               / np.sqrt(_d(x).var(-1, keepdims=True) + 1e-5) * w + b).astype(f32)],
         rtol=1e-3, atol=1e-4),
    dict(id="f09_softmax_wide",
         code="torch.softmax(torch.randn(8, 4096), dim=-1)",
         ref=lambda x: [_softmax(_d(x)).astype(f32)], rtol=1e-3, atol=1e-6),
    dict(id="f10_normalize_sum",
         code="(lambda x: x / x.sum(-1, keepdim=True))(torch.randn(64, 128).abs())",
         ref=lambda x: [(np.abs(_d(x)) / np.abs(_d(x)).sum(-1, keepdims=True)).astype(f32)],
         rtol=1e-3, atol=1e-5),
    dict(id="f11_softmax_f16",
         code="torch.softmax(torch.randn(64, 256, dtype=torch.float16), dim=-1)",
         ref=lambda x: [_softmax(_d(x)).astype(f16).astype(f32)], rtol=5e-3, atol=1e-4),

    # ---- category 5: attention -------------------------------------------
    dict(id="a00_small",
         code="q = torch.randn(32, 16); kt = torch.randn(16, 32); v = torch.randn(32, 16); "
              "torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         ref=lambda q, kt, v: _attn(q, kt, v, 1 / 4.0), rtol=1e-3, atol=1e-4),
    dict(id="a01_256",
         code="q = torch.randn(256, 64); kt = torch.randn(64, 256); v = torch.randn(256, 64); "
              "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: _attn(q, kt, v, 1 / 8.0), rtol=1e-3, atol=1e-4),
    dict(id="a02_rect",
         code="q = torch.randn(64, 32); kt = torch.randn(32, 128); v = torch.randn(128, 64); "
              "torch.softmax(q @ kt * 0.1767767, dim=-1) @ v",
         ref=lambda q, kt, v: _attn(q, kt, v, 0.1767767), rtol=1e-3, atol=1e-4),
    dict(id="a03_noscale",
         code="q = torch.randn(64, 64); kt = torch.randn(64, 64); v = torch.randn(64, 64); "
              "torch.softmax(q @ kt, dim=-1) @ v",
         ref=lambda q, kt, v: _attn(q, kt, v, 1.0), rtol=1e-3, atol=1e-4),
    dict(id="a04_batched",
         code="q = torch.randn(2, 64, 32); kt = torch.randn(2, 32, 64); v = torch.randn(2, 64, 32); "
              "torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=lambda q, kt, v: _attn(q, kt, v, 1 / 5.656854), rtol=1e-3, atol=1e-4),
    dict(id="a05_heads",
         code="q = torch.randn(4, 32, 16); kt = torch.randn(4, 16, 48); v = torch.randn(4, 48, 16); "
              "torch.softmax(q @ kt * 0.25, dim=-1) @ v",
         ref=lambda q, kt, v: _attn(q, kt, v, 0.25), rtol=1e-3, atol=1e-4),
    dict(id="a06_long_kv",
         code="q = torch.randn(16, 64); kt = torch.randn(64, 512); v = torch.randn(512, 64); "
              "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: _attn(q, kt, v, 1 / 8.0), rtol=1e-3, atol=1e-4),
    dict(id="a07_tall_q",
         code="q = torch.randn(256, 16); kt = torch.randn(16, 64); v = torch.randn(64, 16); "
              "torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         ref=lambda q, kt, v: _attn(q, kt, v, 1 / 4.0), rtol=1e-3, atol=1e-4),
]

OPEN = CLOSED  # tools/generate.py reads `OPEN` from whatever expression file it is given
