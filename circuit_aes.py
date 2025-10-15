"""
Implementation of the masked AES circuit.
"""

from dataclasses import dataclass
from typing import Sequence, Callable
import logging

import tqdm

import gates
from gadget import InputSharing, NlognRefresh, Xor, LinearGadget, MulCst


@dataclass
class AesOptions:
    d: int
    nrounds: int
    fake_ks: bool
    mul: Callable
    more_refreshes: bool


class Squaring(LinearGadget):
    N_SQ = 1

    @classmethod
    def arity(cls) -> int:
        return 1

    def linear_op(
        self, share_idx: int, shares: Sequence[gates.Gate]
    ) -> tuple[gates.Gate, list[gates.Gate]]:
        (x,) = shares
        circ = []
        for i in range(self.N_SQ):
            name = f"sq{self.N_SQ}{i}_{share_idx}"
            x = gates.OpGate.new_unique(name, gates.OpKind.SQUARE, [x])
            circ.append(x)
        return x, circ


class Squaring2(Squaring):
    N_SQ = 2


class Squaring4(Squaring):
    N_SQ = 4


class Mul2(LinearGadget):
    @classmethod
    def arity(cls) -> int:
        return 1

    def linear_op(
        self, share_idx: int, shares: Sequence[gates.Gate]
    ) -> tuple[gates.Gate, list[gates.Gate]]:
        x = gates.OpGate.new_unique(f"mul2_{share_idx}", gates.OpKind.MUL2, shares)
        return (x, [x])


class Mul3(LinearGadget):
    @classmethod
    def arity(cls) -> int:
        return 1

    def linear_op(
        self, share_idx: int, shares: Sequence[gates.Gate]
    ) -> tuple[gates.Gate, list[gates.Gate]]:
        x = gates.OpGate.new_unique(f"mul3_{share_idx}", gates.OpKind.MUL3, shares)
        return (x, [x])


class SboxAffine(LinearGadget):
    """We need all gadgets to be SNI, so we put a refresh here."""

    @classmethod
    def arity(cls) -> int:
        return 1

    def linear_op(
        self, share_idx: int, shares: Sequence[gates.Gate]
    ) -> tuple[gates.Gate, list[gates.Gate]]:
        (x,) = shares
        kind = gates.OpKind.AFFINE if share_idx == 0 else gates.OpKind.MULC
        x = gates.OpGate.new_unique(f"sbaff{share_idx}", kind, [x])
        return x, [x]


class AddRc(LinearGadget):
    """We need all gadgets to be SNI, so we put a refresh here."""

    @classmethod
    def arity(cls) -> int:
        return 1

    def linear_op(
        self, share_idx: int, shares: Sequence[gates.Gate]
    ) -> tuple[gates.Gate, list[gates.Gate]]:
        (x,) = shares
        if share_idx == 0:
            x = gates.OpGate.new_unique("addrc", gates.OpKind.ADDPUB, [x])
            return x, [x]
        else:
            return x, []


def field_aes_inv(name, input, opts: AesOptions):
    input2 = NlognRefresh(f"{name}_r1", input)
    in_sq = Squaring(f"{name}_s1", input2)  # .^2
    m1 = opts.mul(f"{name}_m1", input2, in_sq)
    m1_4 = Squaring2(f"{name}_s2", m1)  # .^4
    if opts.more_refreshes:
        m1 = NlognRefresh(f"{name}_rm1", m1)
    m2 = opts.mul(f"{name}_m2", m1_4, m1)
    m2_16 = Squaring4(f"{name}_s3", m2)  # .^16
    m3 = opts.mul(f"{name}_m3", m1_4, m2_16)
    if opts.more_refreshes:
        m3 = NlognRefresh(f"{name}_rm3", m3)
    m4 = opts.mul(f"{name}_m4", m3, in_sq)
    return m4


def aes_sbox(name, input, opts: AesOptions):
    inv = field_aes_inv(name, input, opts)
    # Affine transformation.
    return SboxAffine(f"{name}_aff", inv)


def sub_bytes(name, state, opts):
    return [
        [aes_sbox(f"{name}_{i}_{j}", x, opts) for i, x in enumerate(col)]
        for j, col in enumerate(state)
    ]


def shift_rows(state):
    return [[state[(col + row) % 4][row] for row in range(4)] for col in range(4)]


def sum4(name, x0, x1, x2, x3):
    t0 = Xor(f"{name}_sum4t0", x0, x1)
    t1 = Xor(f"{name}_sum4t1", x2, x3)
    return Xor(f"{name}_sum4r", t0, t1)


def refresh_col(name, col):
    return [NlognRefresh(f"{name}_{i}", x) for i, x in enumerate(col)]


def mix_column(name, col):
    col1 = refresh_col(f"{name}_r1", col)
    col2 = refresh_col(f"{name}_r2", col)

    doubles = [Mul2(f"{name}_double_{i}", x) for i, x in enumerate(col1)]
    triples = [Mul3(f"{name}_triple_{i}", x) for i, x in enumerate(col1)]
    return [
        sum4(f"{name}_0", doubles[0], triples[1], col2[2], col2[3]),
        sum4(f"{name}_1", col2[0], doubles[1], triples[2], col2[3]),
        sum4(f"{name}_2", col2[0], col2[1], doubles[2], triples[3]),
        sum4(f"{name}_3", triples[0], col2[1], col2[2], doubles[3]),
    ]


def mix_columns(name, state):
    return [mix_column(f"{name}_{i}", col) for i, col in enumerate(state)]


def xor_cols(s, col0, col1):
    return [Xor(f"{s}_{i}", a, b) for i, (a, b) in enumerate(zip(col0, col1))]


def add_round_key(name, state, round_key):
    return [
        xor_cols(f"{name}_{i}", col, rk_col)
        for i, (col, rk_col) in enumerate(zip(state, round_key))
    ]


def rot_word(col):
    return col[1:] + [col[0]]


def sub_word(name, col, opts):
    return [aes_sbox(f"{name}_{i}", x, opts) for i, x in enumerate(col)]


def add_rc(name, col):
    return [AddRc(f"{name}_{i}", x) for i, x in enumerate(col)]


def key_schedule(opts):
    nsk = opts.nrounds + 1
    key = [[InputSharing(f"key{i}_{j}", opts.d) for i in range(4)] for j in range(4)]
    w = [refresh_col(f"KS_r_inner{i}", col) for i, col in enumerate(key)]
    out = [refresh_col(f"KS_r_outer{i}", col) for i, col in enumerate(key)]
    for i in tqdm.tqdm(range(4, 4 * nsk), desc="KS cols"):
        k_prev = refresh_col(f"KS_r_prev{i}", w[i - 1])
        if i % 4 == 0:
            t = add_rc(f"RC_{i}", sub_word(f"KS_s{i}", rot_word(k_prev), opts))
        else:
            t = k_prev
        k_new = xor_cols(f"KS_x{i}", w[i - 4], t)
        w.append(refresh_col(f"KS_r_inner{i}", k_new))
        out.append(refresh_col(f"KS_r_outer{i}", k_new))
    return [out[4 * i : 4 * (i + 1)] for i in range(nsk)]


def fake_key_schedule(opts):
    return [
        [[InputSharing(f"key{r}_{i}_{j}", opts.d) for i in range(4)] for j in range(4)]
        for r in range(opts.nrounds + 1)
    ]


def aes(opts: AesOptions):
    logging.info(f"Building AES circuit with {opts.nrounds} rounds and {opts.d} shares")
    state = [[InputSharing(f"pt{i}_{j}", opts.d) for i in range(4)] for j in range(4)]
    if opts.fake_ks:
        rks = fake_key_schedule(opts)
    else:
        rks = key_schedule(opts)
    state = add_round_key("ARK_0", state, rks[0])
    for i, rk in enumerate(tqdm.tqdm(rks[1:], desc="AES rounds")):
        r = i + 1
        if r == 9:
            state = sub_bytes(f"SB_{r}", state, opts)
            state = shift_rows(state)
            state = add_round_key(f"ARK_{r}", state, rks[-1])
        else:
            state = sub_bytes(f"SB_{r}", state, opts)
            state = shift_rows(state)
            state = mix_columns(f"MC_{r}", state)
            state = add_round_key(f"ARK_{r}", state, rk)
    return state
