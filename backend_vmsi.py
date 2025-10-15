"""
VerifMSI backend.
"""

from typing import Iterable

import verif_msi
import tqdm

import gates
import backends


class VerifMsiBackend(backends.Backend):
    def __init__(self, nbits=1):
        self._nbits = nbits
        self._verifmsi_gates = dict()
        self._verifmsi_sharings = dict()

    def name(self):
        return "VerifMSI"

    def _from_gate(self, gate: gates.Gate) -> verif_msi.hw.Gate:
        if gate.name not in self._verifmsi_gates:
            self._verifmsi_gates[gate.name] = self._build_verifmsi(gate)
        return self._verifmsi_gates[gate.name]

    def _build_verifmsi(self, gate: gates.Gate) -> verif_msi.hw.Gate:
        if isinstance(gate, gates.RndGate):
            return verif_msi.utils.symbol(gate.name, "M", self._nbits)
        elif isinstance(gate, gates.CstGate):
            return verif_msi.constant(gate.value, self._nbits)
        elif isinstance(gate, gates.ShareGate):
            if gate.sharing_name not in self._verifmsi_sharings:
                secret = verif_msi.utils.symbol(gate.sharing_name, "S", self._nbits)
                shares = verif_msi.utils.getPseudoShares(secret, gate.nshares)
                self._verifmsi_sharings[gate.sharing_name] = shares
            return self._verifmsi_sharings[gate.sharing_name][gate.share_idx]
        elif isinstance(gate, gates.OpGate):
            ops = [self._from_gate(op) for op in gate.operands]
            match gate.op:
                case gates.OpKind.ADD:
                    return verif_msi.simplify(verif_msi.Node.makeBitwiseNode("^", ops))
                case gates.OpKind.MUL:
                    if self._nbits == 1:
                        return verif_msi.simplify(
                            verif_msi.Node.makeBitwiseNode("&", ops)
                        )
                    elif self._nbits == 8:
                        return verif_msi.simplify(verif_msi.GMul(*ops))
                    else:
                        raise ValueError("Cannot MUL with VerifMSI if not 1 or 8 bits.")
                case gates.OpKind.NEG:
                    one = verif_msi.constant(2**self._nbits - 1, self._nbits)
                    res = verif_msi.Node.makeBitwiseNode("^", [*ops, one])
                    return verif_msi.simplify(res)
                case gates.OpKind.MUL2:
                    assert self._nbits == 8
                    two = verif_msi.constant(2, self._nbits)
                    return verif_msi.simplify(verif_msi.GMul(*ops, two))
                case gates.OpKind.MUL3:
                    assert self._nbits == 8
                    two = verif_msi.constant(2, self._nbits)
                    return verif_msi.simplify(verif_msi.GMul(*ops, two))
                case gates.OpKind.MULC:
                    assert self._nbits == 8
                    c = verif_msi.constant(31, self._nbits)
                    return verif_msi.simplify(verif_msi.GMul(*ops, c))
                case gates.OpKind.AFFINE:
                    assert self._nbits == 8
                    c = verif_msi.constant(31, self._nbits)
                    a = verif_msi.constant(99, self._nbits)
                    return verif_msi.simplify(
                        verif_msi.Node.makeBitwiseNode(
                            "^", [verif_msi.GMul(*ops, c), a]
                        )
                    )
                case gates.OpKind.ADDPUB:
                    return ops[0]  # Doesn't change simulatability
                case gates.OpKind.SQUARE:
                    assert self._nbits == 8
                    return verif_msi.simplify(verif_msi.GMul(*ops, *ops))
                case gates.OpKind.NOP:
                    return ops[0]

    def check_gate_tuple(self, gates: list[gates.Gate]) -> bool:
        """Return True if the tuple of `probes` is independent of any secret."""
        if not gates:
            return True
        probed_wires = [self._from_gate(p) for p in gates]
        res = verif_msi.check_leakage.checkTpsVal(verif_msi.node.Concat(*probed_wires))[
            0
        ]
        return res

    def check_gate_tuples(self, tuples: Iterable[list[gates.Gate]]) -> int:
        """Return the number of failures"""
        return sum(
            not self.check_gate_tuple(p)
            for p in tqdm.tqdm(tuples, desc="VMSI backend check")
        )


_backend = None


def get_backend(gates: list[gates.Gate], n_bits: int):
    global _backend
    if _backend is None:
        _backend = VerifMsiBackend(n_bits)
    return _backend
