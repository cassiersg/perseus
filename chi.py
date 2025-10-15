"""
Top-level file for computing masked chi function probing security.
"""

from gadget import (
    InputSharing,
    IswMult,
    NlognRefresh,
    Xor,
    IswAndNot,
)
from gadget_graph import GadgetGraph


def rot(x, n):
    return x[n:] + x[:n]


if __name__ == "__main__":
    import rp_args

    parser = rp_args.parser("rpchi")
    parser.add_argument("--arity", type=int, default=5, help="number of inputs of chi")
    args = parser.parse_args()
    IswMult.sanity_check(args.d)
    IswAndNot.sanity_check(args.d)
    NlognRefresh.sanity_check(args.d)
    Xor.sanity_check(args.d)
    inputs = [InputSharing(f"a{i}", args.d) for i in range(args.arity)]
    x = [
        args.mul(f"chi_an{i}", a, b)
        # IswAndNot(f"chi_an{i}", a, b)
        for i, (a, b) in enumerate(zip(rot(inputs, 2), rot(inputs, 1)))
    ]
    y = [Xor(f"chi_xor{i}", a, b) for i, (a, b) in enumerate(zip(inputs, x))]
    gg = GadgetGraph(
        args.p, y, gate_leakage=not args.wire_model, backend=args.backend, n_bits=1
    )
    lb, ub = gg.compute_rps_mc(
        n_samples=args.samples,
        delta=args.delta,
        prej_lim=args.prej_lim,
        e_samples=args.e_samples,
    )
    print(f"Opt. RPS bounds: [{lb:e}, {ub:e}]")
