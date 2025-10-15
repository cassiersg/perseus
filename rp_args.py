"""Command-line parsing."""

import argparse
import logging

import gadget


def get_backend(name):
    match name:
        case "verifmsi":
            import backend_vmsi

            return backend_vmsi.get_backend
        case "maskverif":
            import backend_mv

            return backend_mv.get_backend
        case "favom":
            import backend_favom

            return backend_favom.get_backend
        case "favom-d":
            import backend_favom

            return lambda *args: backend_favom.get_backend(*args, debug_assert=True)
    raise ValueError(f"Unknown backend {name}")


def get_refresh(name: str):
    REFRESH = dict(
        nlogn=gadget.NlognRefresh,
        simple=gadget.SimpleRefresh,
        half=gadget.HalfRefresh,
        half2=gadget.HalfRefresh2,
        circ=gadget.CircRefresh,
    )
    try:
        return REFRESH[name]
    except KeyError:
        raise ValueError(
            f"'{name}' is not a known refresh gadget (known gadgets: {list(REFRESH.keys())}"
        )


def get_mul(arg: str):
    arg = arg.lower()
    if arg == "isw":
        return gadget.IswMult
    elif arg.startswith("full"):
        ref = get_refresh(arg.removeprefix("full"))
        return lambda name, *ops: gadget.FullMatRefMult(name, *ops, refresh=ref)
    else:
        raise ValueError(f"'{arg}' is not a known multiplication gadget.")


def set_log(level):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        force=True,
        format="[%(asctime)s]%(levelname)s:%(message)s",
    )


def parser(name="rpcheck"):
    parser = argparse.ArgumentParser(name)
    parser.add_argument("-d", type=int, default=3, help="number of shares")
    parser.add_argument(
        "-w",
        "--wire-model",
        action="store_true",
        help="wire leakage model instead of gate model",
    )
    parser.add_argument(
        "-p", type=eval, default="0.01", help="random probing probability"
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=eval,
        default="10**4",
        help="number of Monte-Carlo samples (for the main sampling)",
    )
    parser.add_argument(
        "--e-samples",
        type=eval,
        default=None,
        help="number of Monte-Carlo samples for Pr[E] computation",
    )
    parser.add_argument(
        "--delta", type=float, default=1e-3, help="confidence level of the bound"
    )
    parser.add_argument(
        "--prej-lim",
        type=int,
        default=None,
        help="limit for excl.-incl. in P[E] computation",
    )
    parser.add_argument(
        "--backend",
        type=get_backend,
        default="maskverif",
        help="backend (verifmsi, maskverif, favom or favom-d)",
    )
    parser.add_argument(
        "--log", type=set_log, default="info", help="log level (DEBUG/INFO)"
    )
    parser.add_argument(
        "--mul",
        type=get_mul,
        default="isw",
        help="Multiplication gadget (isw/fullnlogn/fullsimple/fullhalf/fullhalf2/fullcirc)",
    )
    return parser
