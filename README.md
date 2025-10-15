# PERSEUS

Verification tool for random probing security, for circuits composed of SNI gadgets.

## Dependencies

The tool is partially written in Rust, the toolchain needs to be be [installed](https://rust-lang.org/tools/install).

The remaining part of the tool is implemented in Python, we provide
configuration for automatic management of the dependencies with the
[uv tool](https://docs.astral.sh/uv/getting-started/installation/).

### Backend

PERSEUS provides three backends to evaluate the simulatability of tuples:
- **`favom` (built-in)**: a home-grown backend for fast verification of masking. No extra setup beyond PERSEUS.
- **`maskverif` (custom fork)**:  build the custom branch and make the binary available on your `PATH`:
  - Source: <https://github.com/cassiersg/maskverif/tree/verif_single_tuple> (** branch:** `verif_single_tuple`)
  - Recommended build:
    ```bash
    git clone -b verif_single_tuple https://github.com/cassiersg/maskverif.git
    cd maskverif
    nix-shell        # optional but recommended: provides a consistent OCaml toolchain
    dune build
    # make the freshly built binary available (run this before running PERSEUS, not in the nix-shell)
    export PATH="$PWD:$PATH"
    ```
- **`verifMSI`**: extracted from the verifMSI tool and installed automatically via uv alongside PERSEUS; no manual steps required.

## Getting started

```sh
uv run aes.py --help
```

For example
```sh
uv run aes.py -d 8 -p '2**-10' --samples='2**12' --backend=favom --e-samples='2**14' --prej-lim=3
```

- 10 AES rounds with a AES-128 key schedule
- computes for 8 shares
- uses the gate leakage model with gate leakage probability `2**-10`
- Uses `2**12` Monte Carlo samples with the FAVOM verification backend
- For SNI verification failure, uses the best of `2**14` Monte-Carlo samples and inclusion-exclusion up to size 3.

**Changing the number of cores used** Set the `RAYON_NUM_THREADS` variable to the number of desired threads.
