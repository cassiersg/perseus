"""
Top-level file for computing masked AES probing security.
"""

import numpy as np

from gadget_graph import GadgetGraph

import circuit_aes


if __name__ == "__main__":
    import rp_args
    import logging
    import time

    parser = rp_args.parser("rpAES-128")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of rounds (default: 10, full AES)",
    )
    parser.add_argument(
        "--fks",
        action="store_true",
        help="Compute without key schedule (fresh key for each round)",
    )
    parser.add_argument(
        "--mr",
        action="store_true",
        help="More refreshes in S-Boxes",
    )
    args = parser.parse_args()
    opts = circuit_aes.AesOptions(
        d=args.d,
        nrounds=args.rounds,
        fake_ks=args.fks,
        mul=args.mul,
        more_refreshes=args.mr,
    )
    ct = circuit_aes.aes(opts)
    y = [x for col in ct for x in col]
    logging.info("AES circuit built")
    t_start = time.time()
    gg = GadgetGraph(
        args.p,
        y,
        gate_leakage=not args.wire_model,
        backend=args.backend,
        n_bits=8,
    )
    logging.info("GadgetGraph initialized")
    lb, ub = gg.compute_rps_mc(
        n_samples=args.samples,
        delta=args.delta,
        prej_lim=args.prej_lim,
        e_samples=args.e_samples,
    )
    t_end = time.time()
    logging.info(f"Verification time (s): {t_end-t_start}")
    print(f"Opt. RPS bounds: [{lb:e}, {ub:e}]")
    with np.errstate(divide="ignore"):
        print(f"Opt. RPS bounds: [2**{np.log2(lb)}, 2**{np.log2(ub)}]")
