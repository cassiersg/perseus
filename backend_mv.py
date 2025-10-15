"""
Backend for using MaskVerif.

We use a custom version of maskverif that uses the same verification algorithm
but exposes it through a different interface.
"""

# Protocol for interaction with maskverif based on json lines
#
# Maskverif stdin:
# - 1st line is circuit description
# - each consecutive line is a Tuple
#
# Maskverif stdout:
# - each line is the Result for a Tuple
#
# Circuit description is an object that contains a single entry "gates".
# The "gates" entry is a list, whose element are objects with entries:
# - "name": string with a unique gate name
# - "kind": can be "secret", "random", "operation", "constant"
# - "operation": can be "mul", "add", "neg" (only present when "kind" is "operation")
# - "operands" is a list of gate names (only present when "kind" is "operation")
# - "value" is an integer (only present when "kind" is "constant")
#
# Tuple is an object with a single entry "probes" which is a list of gate names.
#
# Result is an object with a single entry "result" which is true (no secret
# dependency) or false (possible dependency).

import os
import logging
import json
import shutil
import subprocess as sp
from typing import Iterable

import tqdm

import gates
import sp_line_pool
import backends


class MvWorker(sp_line_pool.LineAsyncWorker):
    async def post_init(self, circuit):
        init_res = await self.execute_job(circuit)
        assert init_res["done"] is True

    async def execute_job(self, probes):
        logging.debug(f"Backend {self._idx} job %s", json.dumps(probes))
        await self.write_line(json.dumps(probes))
        MV_PREFIX = "MVRES: "
        res = ""
        while not res.startswith(MV_PREFIX):
            res = await self.read_line()
            if res:
                logging.debug(f"Backend {self._idx} from mv: %s", res)
        return json.loads(res.removeprefix(MV_PREFIX))


class MaskVerifBackendMT(backends.Backend):
    def __init__(
        self,
        all_gates: list[gates.Gate],
        mv_exec,
        num_workers: int = 1,
        debug: bool = False,
    ):
        self._debug = debug
        if self._debug:
            self._f = open("mv_input.txt", "w")
        self._gates = dict()
        self._circuit = backends.CircuitBuilder(all_gates)
        self.timings = dict()
        circuit = dict(gates=self._circuit.list_gates())
        logging.info(f"Initializing maskverif circuit ({len(circuit['gates'])} gates)")
        self._pool = sp_line_pool.SyncWorkPool(
            num_workers, lambda i: MvWorker.create(i, mv_exec, circuit)
        )
        if self._debug:
            self._f.write(json.dumps(circuit))
            self._f.write("\n")
        logging.info("Maskverif circuit initialized")

    def name(self):
        return "Maskverif"

    def tuple2probes(self, all_gates: list[gates.Gate]):
        probes = dict(probes=[self._circuit.name_of_gate(gate) for gate in all_gates])
        if self._debug:
            self._f.write(json.dumps(probes))
            self._f.write("\n")
        return probes

    def process_res(self, tuple_res):
        for k, v in tuple_res.items():
            if k.endswith("_time"):
                self.timings[k] = self.timings.get(k, 0.0) + v
        return tuple_res["result"]

    def check_gate_tuple(self, all_gates: list[gates.Gate]) -> bool:
        probes = self.tuple2probes(all_gates)
        logging.debug("Maskverif backend probes: %s", json.dumps(probes))
        res = self._pool.exec(probes)
        return self.process_res(res)

    def check_gate_tuples(self, tuples: Iterable[list[gates.Gate]]) -> int:
        """Return the number of failures"""
        tuples = tqdm.tqdm(tuples, desc="MV backend check")
        n_fail = sum(
            not self.process_res(res)
            for res in self._pool.map(map(self.tuple2probes, tuples))
        )
        logging.info(f"MV timings (s): {self.timings}, {n_fail=}")
        return n_fail


class MaskVerifBackend(backends.Backend):
    def __init__(
        self,
        all_gates: list[gates.Gate],
        mv_exec,
        idx: int = 0,
        debug: bool = False,
    ):
        self._debug = debug
        if self._debug:
            self._f = open("mv_input.txt", "w")
        self._idx = idx
        self._gates = dict()
        self._circuit = backends.CircuitBuilder(all_gates)
        self.timings = dict()
        circuit = dict(gates=self._circuit.list_gates())
        self._sp = sp.Popen(
            [mv_exec], stdin=sp.PIPE, stdout=sp.PIPE, text=True, bufsize=1
        )
        c_json = json.dumps(circuit)
        # print("MV circuit", c_json)
        logging.info(f"Backend {self._idx} writing MV circuit")
        self._sp.stdin.write(c_json)
        self._sp.stdin.write("\n")
        if self._debug:
            self._f.write(c_json)
            self._f.write("\n")
        logging.info(
            f"Backend {self._idx} Initializing maskverif circuit... ({len(circuit['gates'])} gates)"
        )
        assert self._read_mv_res()["done"] is True

    def name(self):
        return "Maskverif"

    def check_gate_tuple(self, all_gates: list[gates.Gate]) -> bool:
        probes = dict(probes=[self._circuit.name_of_gate(gate) for gate in all_gates])
        probes_json = json.dumps(probes)
        logging.debug(f"Backend {self._idx}: probes_json: %s", probes_json)
        self._sp.stdin.write(probes_json)
        self._sp.stdin.write("\n")
        if self._debug:
            self._f.write(probes_json)
            self._f.write("\n")
        res = self._read_mv_res()
        for k, v in res.items():
            if k.endswith("_time"):
                self.timings[k] = self.timings.get(k, 0.0) + v
        # print("got", res)
        return res["result"]

    def check_gate_tuples(self, tuples: Iterable[list[gates.Gate]]) -> int:
        """Return the number of failures"""
        logging.info("Enter MV check_gate_tuples")
        res = sum(
            not self.check_gate_tuple(p)
            for p in tqdm.tqdm(tuples, desc="MV backend check")
        )
        logging.info(f"MV timings (s): {self.timings}")
        return res

    def _read_mv_res(self):
        MV_PREFIX = "MVRES: "
        res = ""
        while not res.startswith(MV_PREFIX):
            res = self._sp.stdout.readline()[:-1]
            if res:
                logging.debug(f"Backend {self._idx} from mv: %s", res)
        return json.loads(res.removeprefix(MV_PREFIX))


def get_backend(all_gates: list[gates.Gate], n_bits: int):
    mv_exec = shutil.which("maskverif.exe")
    if mv_exec is None:
        mv_exec = shutil.which("maskverif")
    if mv_exec is None:
        raise Exception(
            "Could not find 'maskverif.exe', PATH={}".format(os.environ.get("PATH"))
        )
    return MaskVerifBackendMT(
        all_gates, num_workers=int(os.environ.get("NUM_THREADS", "8")), mv_exec=mv_exec
    )
