"""
A `backend` is an implementation of a secret-independence checking algorithm:
given a tuple of expressions, the backend tells whether they can be proven to
be independent of the secret.
All backends inherit from the `Backend` class.
"""

import abc
from typing import Iterable

import gates

import rbackend


class Backend:
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def check_gate_tuple(self, all_gates: list[gates.Gate]) -> bool:
        pass

    @abc.abstractmethod
    def check_gate_tuples(self, tuples: Iterable[list[gates.Gate]]) -> bool:
        pass


class FastBackend(Backend):
    @abc.abstractmethod
    def sample_and_check(
        self,
        gadgets: list[list[list[gates.Gate]]],
        n_samples: int,
        tuples_sampler: rbackend.TupleSizeSampler,
    ) -> tuple[int, int]:
        pass


class CircuitBuilder:
    """Helper for the circuit-building part of backends:
    from any set of gates, give a sorted list of gates covering all necessary
    sub-expressions.
    """

    def __init__(self, all_gates: list[gates.Gate]):
        self._gate_list = []
        self._gates_by_name = dict()
        self._gates = dict()
        self._sharings = dict()
        self._gate_to_name = dict()
        self._add_gates(all_gates)

    def name_of_gate(self, gate: gates.Gate):
        return self._gate_to_name[gate]

    def _add_gates(self, gs: list[gates.Gate]):
        gates_to_do = list(gs)
        while gates_to_do:
            g = gates_to_do.pop()
            if isinstance(g, gates.OpGate):
                op_g = next((op for op in g.operands if op not in self._gates), None)
                if op_g is not None:
                    gates_to_do.append(g)
                    gates_to_do.append(op_g)
                    continue
            self._add_gate_assume_parents(g)

    def _add_gate_assume_parents(self, gate: gates.Gate):
        if gate not in self._gates:
            self._build_gate(gate)
            self._gates[gate] = gate
            self._gates_by_name[gate.name] = gate
        else:
            assert self._gates_by_name[gate.name] == gate

    def _build_gate(self, gate: gates.Gate):
        if isinstance(gate, gates.RndGate):
            self._gate_list.append(
                dict(
                    name=gate.name,
                    kind="random",
                )
            )
            self._gate_to_name[gate] = gate.name
        elif isinstance(gate, gates.CstGate):
            self._gate_list.append(
                dict(
                    name=gate.name,
                    kind="constant",
                    value=gate.value,
                )
            )
            self._gate_to_name[gate] = gate.name
        elif isinstance(gate, gates.ShareGate):
            if gate.sharing_name not in self._sharings:
                secret = dict(name=gate.name, kind="secret")
                randoms = [
                    dict(name=f"SH:rnd{i}:{gate.name}", kind="random")
                    for i in range(gate.nshares - 1)
                ]
                self._gate_list.append(secret)
                last_share = dict(
                    name=f"SH:rnd{gate.nshares-1}:{gate.name}",
                    kind="operation",
                    operation="add",
                    operands=[secret["name"]] + [r["name"] for r in randoms],
                )
                shares = randoms + [last_share]
                self._gate_list.extend(shares)
                self._sharings[gate.sharing_name] = shares
            self._gate_to_name[gate] = self._sharings[gate.sharing_name][
                gate.share_idx
            ]["name"]
        elif isinstance(gate, gates.OpGate):
            op_names = [self.name_of_gate(op) for op in gate.operands]
            if gate.op == gates.OpKind.NOP:
                self._gate_to_name[gate] = op_names[0]
                return  # We're done, no new gate.
            operation = gate.op.name.lower()
            self._gate_list.append(
                dict(
                    name=gate.name,
                    kind="operation",
                    operation=operation,
                    operands=op_names,
                )
            )
            self._gate_to_name[gate] = gate.name

    def list_gates(self):
        return self._gate_list
