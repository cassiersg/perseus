"""Top-level script for checking gadget descriptions."""

import logging

from gadget import (
    IswMult,
    NlognRefresh,
    Xor,
)

from circuit_aes import BitwiseNop

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("gadget checks")
    parser.add_argument("-d", type=int, required=True)
    args = parser.parse_args()

    gadgets = [NlognRefresh, IswMult, Xor, BitwiseNop]

    for g in gadgets:
        logging.info(f"Sanity checking gadget {str(g)}")
        g.sanity_check(args.d)
