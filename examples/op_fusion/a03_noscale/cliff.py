"""ptxas's unroll cliff in a03_noscale: un-fuse our fma in the first n dot-product loops of the
fusion build, for n = 0..12, and show what ptxas makes of each.

    python examples/op_fusion/a03_noscale/cliff.py

Prints one markdown table row per n. A dot-product loop is a loop inside the outer loop; un-fusing
turns its ``fma.rn`` into a plain ``mul`` and ``add``, as ``--no_fusion`` emits them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools"))

from sass_waits import sass  # noqa: E402

FMA = re.compile(r"(\s*)fma\.rn\.f32 (%f\d+), (%f\d+), (%f\d+), (%f\d+);")


def unfuse(ptx: str, n: int) -> str:
    """``ptx`` with the fma of its first ``n`` inner loops split into a mul and an add."""
    out, depth, inner, extra = [], 0, 0, 0
    for line in ptx.splitlines():
        if re.match(r"\$L_loop_\d+:", line):
            depth += 1
            inner += depth == 2
        elif re.match(r"\$L_loop_end_\d+:", line):
            depth -= 1
        m = FMA.fullmatch(line)
        if m and depth == 2 and inner <= n:
            ind, d, a, b, c = m.groups()
            out += [f"{ind}mul.f32 %fx{extra}, {a}, {b};", f"{ind}add.f32 {d}, {c}, %fx{extra};"]
            extra += 1
        else:
            out.append(line)
    text = "\n".join(out) + "\n"
    if not extra:
        return text
    return text.replace(".reg .f32 %f<", f".reg .f32 %fx<{extra}>;\n\t.reg .f32 %f<", 1)


def main() -> None:
    fusion = next(HERE.glob("*.fusion.ptx")).read_text()
    print("| Un-fused loops | PTX lines | Registers | Loops in SASS | FFMA in SASS | LDG in SASS |")
    print("|---|---|---|---|---|---|")
    for n in range(13):
        ptx = unfuse(fusion, n)
        summary, instrs = sass(ptx, "sm_120")
        ops = [i.split()[1] if i.startswith("@") else i.split()[0] for _, i in instrs]
        loops = sum(
            1
            for a, i in instrs
            if (m := re.search(r"\bBRA (0x[0-9a-f]+)", i)) and int(m.group(1), 16) < a
        )
        ffma = sum(op.split(".")[0] == "FFMA" for op in ops)
        ldg = sum(op.startswith("LDG") for op in ops)
        regs = summary.split()[0]
        print(f"| {n} | {ptx.count(chr(10))} | {regs} | {loops} | {ffma} | {ldg} |")


if __name__ == "__main__":
    main()
