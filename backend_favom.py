"""
New backend: FAVOM.

This uses the same simplification algorithm as maskVerif but uses more efficient datastructures.
"""

from typing import Iterable
import random
import logging
import time

import tqdm

import gates
import rbackend
import backends
import utils


class FavomBackend(backends.FastBackend):
    def __init__(self, all_gates: list[gates.Gate], debug_assert):
        self._da = debug_assert
        assert rbackend.is_release()
        self._circuit = backends.CircuitBuilder(all_gates)
        self._fv = rbackend.FavomExprs()
        self._fv.add_operator("square", True, 1, True, False)
        self._fv.add_operator("mul2", True, 1, True, False)
        self._fv.add_operator("mul3", True, 1, True, False)
        self._fv.add_operator("mulc", True, 1, True, False)
        self._fv.add_operator("affine", True, 1, True, False)
        self._fv.add_operator("addpub", True, 1, True, False)
        self._wires = dict()
        self._secret_names = dict()
        tmp_add_cnt = 0
        for gate in self._circuit.list_gates():
            logging.debug("gate : %s", gate)
            n = gate["name"]
            match gate["kind"]:
                case "secret":
                    e, s = self._fv.add_secret(n)
                    self._wires[n] = e
                    self._secret_names[s] = n
                case "random":
                    self._wires[n] = self._fv.add_random(n)
                case "constant":
                    # FIXME: should have actual constant
                    self._wires[n] = self._fv.add_pub(n)
                case "operation":
                    if gate["operation"] == "add" and len(gate["operands"]) > 2:
                        x = gate["operands"]
                        while len(x) > 2:
                            t = f"__tmp_add_{tmp_add_cnt}"
                            tmp_add_cnt += 1
                            self.add_op_names(t, "add", x[:2])
                            x = [t] + x[2:]
                        self.add_op_names(n, "add", x)
                    else:
                        self.add_op_names(n, gate["operation"], gate["operands"])
                case kind:
                    raise ValueError(f"Unknown gate kind {kind}")
        if self._da:
            import backend_mv

            self._mv_backend = backend_mv.MaskVerifBackend(all_gates, debug=True)
        self._t_state_create = 0.0
        self._t_simplify = 0.0

    def add_op_names(self, dest: str, operation: str, operands: list[str]):
        self._wires[dest] = self._fv.add_op_expr(
            operation, [self._wires[op] for op in operands]
        )

    def name(self):
        return "Favom"

    def check_gate_tuple(self, all_gates: list[gates.Gate]) -> bool:
        probes = [self._wires[self._circuit.name_of_gate(gate)] for gate in all_gates]
        t0 = time.time()
        state = rbackend.FavomState(1, self._fv, probes)
        self._t_state_create += time.time() - t0
        t0 = time.time()
        state.simplify_until(0, self._fv)
        self._t_simplify += time.time() - t0
        secrets = [self._secret_names[s] for s in state.used_secrets()]
        res = len(secrets) == 0
        if self._da:
            mv_res = self._mv_backend.check_gate_tuple(all_gates)
            assert res == mv_res
        return res

    def check_gate_tuples(self, tuples: Iterable[list[gates.Gate]]) -> int:
        """Return the number of failures"""
        logging.info("Enter FV check_gate_tuples")
        probes = [
            [self._wires[self._circuit.name_of_gate(gate)] for gate in probe]
            for probe in tuples
        ]
        with utils.interruptible():
            res = sum(not x for x in rbackend.check_tuples(self._fv, probes))
        logging.info("FAVOM check_gate_tuples done")
        return res

    def sample_and_check(
        self,
        gadgets: list[list[list[gates.Gate]]],
        tuples_sampler: rbackend.TupleSizeSampler,
        n_samples: int,
    ) -> tuple[int, int]:
        logging.info("Enter FV sample_and_check")
        gadgets = [
            [
                [self._wires[self._circuit.name_of_gate(gate)] for gate in eprobe]
                for eprobe in gadget
            ]
            for gadget in gadgets
        ]
        probe_sampler = rbackend.ProbeSampler(gadgets)
        seed = random.randrange(2**64)
        with utils.interruptible():
            res = rbackend.check_tuples_sample_probes(
                self._fv, probe_sampler, tuples_sampler, n_samples, seed
            )
        logging.info("FAVOM sample_and_check done")
        return res


def get_backend(all_gates: list[gates.Gate], n_bits: int, debug_assert: bool = False):
    return FavomBackend(all_gates, debug_assert)
