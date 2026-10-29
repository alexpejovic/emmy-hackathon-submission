"""Emmy Kernel IR -> PTX.

    python compile.py <kernel>.ir.json [--no_loop_unroll] [--no_fusion] [--no_cse] > out.ptx

Reads the IR (with Emmy's own loader) and the meta.json beside it, which is the contract for
the parameter order, the block size and the buffers' shapes, dtypes and outputs.
``--no_loop_unroll`` turns every loop into a runtime loop with one copy of the body per trip.
``--no_fusion`` never fuses a float multiply and add into one ``fma`` (ptxas still may).
``--no_cse`` turns off common subexpression elimination: every side-effect-free op is emitted again.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel import KernelOp

from ptxgen.compiler import Buffer, compile_module


def to_ptx(
    graph: Graph,
    meta: dict[str, Any],
    no_unroll: bool = False,
    no_fusion: bool = False,
    no_cse: bool = False,
) -> str:
    launches = {k["entry"]: k for k in meta["kernels"]}
    ops = [n.op for n in graph.nodes.values() if isinstance(n.op, KernelOp)]
    op = next((o for o in ops if o.name in launches), None)
    if op is None:
        raise ValueError(f"no KernelOp named like a meta.json kernel ({', '.join(launches)})")
    launch = launches[op.name]
    if launch["block"][1:] != [1, 1] or launch["grid"][1:] != [1, 1]:
        raise NotImplementedError("only 1-D launches")
    if any(a.get("scalar") for a in launch["args"]):
        raise NotImplementedError("by-value scalar arguments")
    buffers = {
        a["name"]: Buffer(
            tuple(a["shape"]), a["dtype"], bool(a.get("output")) or a["name"] in op.outputs
        )
        for a in launch["args"]
    }
    return compile_module(
        op,
        buffers,
        launch["arg_order"],
        launch["block"][0],
        no_unroll=no_unroll,
        no_fusion=no_fusion,
        no_cse=no_cse,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("json_path", type=Path, help="Kernel IR json (<kernel>.ir.json)")
    parser.add_argument("--no_loop_unroll", action="store_true", help="never unroll loops")
    parser.add_argument("--no_fusion", action="store_true", help="never fuse a multiply and an add")
    parser.add_argument("--no_cse", action="store_true", help="never reuse an identical result")
    args = parser.parse_args()
    graph = Graph.from_dict(json.loads(args.json_path.read_text()))
    meta = json.loads(args.json_path.with_name("meta.json").read_text())
    sys.stdout.write(to_ptx(graph, meta, args.no_loop_unroll, args.no_fusion, args.no_cse))


if __name__ == "__main__":
    main()
