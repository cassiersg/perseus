"""
Main file for computing the random probing security of a composition of SNI gadgets.
"""

from dataclasses import dataclass
import itertools as it
import logging
from typing import Sequence
from collections import Counter
import time

import numpy as np
import numpy.typing as npt
import scipy.special
import tqdm

import rbackend

import gates
from gadget import SimGadget, Gadget, gadget_n_probe_distr
import proba_sim_ie
import backends
from graph_inequality import Inequality


# All randomness should be derived from this seed, for the sake of
# exact reproducibility.
SEED = 0


def strict_subsets(iterable):
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)))


def bernouilli_bound(
    n_samples: int, n_failures: int, delta: float
) -> tuple[float, float]:
    """
    Random probing security (1-`delta`)-confidence upper bound from
    Monte-Carlo with `n_samples` samples among which `n_failures`
    simulation failures.
    """
    if n_failures != 0:
        logging.info(
            f"Found {n_failures} failures out of {n_samples} samples! {delta=}"
        )
    if n_failures == n_samples:
        ub = 1.0
    else:
        ub = scipy.special.betainccinv(n_failures + 1, n_samples - n_failures, delta)
    if n_failures == 0:
        lb = 0.0
    else:
        lb = scipy.special.betaincinv(n_failures, n_samples - n_failures + 1, delta)
    return (lb, ub)


@dataclass(frozen=True, eq=True)
class InequalityCharacteristic:
    """Some properties of an inequality from which some of its statistical
    properties can be derived."""

    p: float
    t: int
    # Number of leaking points (wires/gates) in gadgets all involved gadgets.
    # TODO: for perf, we might keep that sorted and permute when needed.
    nleak: tuple[int, ...]
    # Number of times each gadget appears in the inequality.
    # TODO: should we warn when this is > 1?
    multiplicity: tuple[int, ...]

    @classmethod
    def from_ineq(
        cls, ineq: Inequality, p: float, gate_leakage: bool
    ) -> "InequalityCharacteristic":
        n_mul = [(g.n_probes(gate_leakage), m) for g, m in ineq.all_g_cnt.items()]
        n_mul = [(n, m) for n, m in n_mul if n != 0]
        return cls(
            p=p,
            t=ineq.t(),
            nleak=tuple(n for n, _ in n_mul),
            multiplicity=tuple(m for _, m in n_mul),
        )

    def group_distr(
        self,
    ) -> tuple[tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]], float]:
        sample_res_iter = it.product(*(range(n + 1) for n in self.nleak))
        sample_res: npt.NDArray[np.int64] = np.array(list(sample_res_iter))
        sample_res *= np.array(list(self.multiplicity))[np.newaxis, :]
        delta = np.sum(sample_res, axis=1) > self.t
        n_tuples = sample_res[delta, :]
        fail_distr = np.prod(
            [
                gadget_n_probe_distr(nl, self.p)[ni]
                for nl, ni in zip(self.nleak, n_tuples.T)
            ],
            axis=0,
        )
        proba_fail = np.sum(fail_distr)
        fail_distr_norm = fail_distr / proba_fail
        return (n_tuples, fail_distr_norm), proba_fail


class Sampler:
    """All sorts of sampling-related functions."""

    def __init__(
        self,
        rng: np.random.Generator,
        inequalities: list[Inequality],
        p: float,
        gate_leakage: bool,
    ):
        self._rng = rng
        self._gate_leakage = gate_leakage
        all_g = set().union(*(ineq.all_g_cnt.keys() for ineq in inequalities))
        self._leaky_gadgets = [g for g in all_g if g.leaky(gate_leakage)]
        leaky_gidxs = {g: i for i, g in enumerate(self._leaky_gadgets)}
        gadget_n_leak = [g.n_probes(gate_leakage) for g in self._leaky_gadgets]
        ineq2gidx = [
            np.array(
                [leaky_gidxs[g] for g in ineq.all_g_cnt.keys() if g in leaky_gidxs]
            )
            for ineq in inequalities
        ]
        ineq_t = np.array([ineq.t() for ineq in inequalities])
        logging.info(
            f"Ineq size counter: {Counter(len(i.all_g_cnt) for i in inequalities)}"
        )

        ineq2c = {
            ineq: InequalityCharacteristic.from_ineq(ineq, p, gate_leakage)
            for ineq in inequalities
        }
        char2distr = {
            c: c.group_distr()
            for c in tqdm.tqdm(set(ineq2c.values()), desc="Ineq distr")
        }
        char2sampler = {
            c: rbackend.JointSizesSampler(weights, sizes)
            for c, ((sizes, weights), _) in tqdm.tqdm(
                char2distr.items(), desc="Ineq samplers"
            )
        }
        ineq_group_distr_samplers = [
            char2sampler[ineq2c[ineq]] for ineq in inequalities
        ]
        probas_fail = np.array([char2distr[ineq2c[ineq]][1] for ineq in inequalities])
        ineq_weights = probas_fail / np.sum(probas_fail)
        self.tuples_sampler = rbackend.TupleSizeSampler(
            p,
            gadget_n_leak,
            ineq2gidx,
            ineq_t,
            ineq_weights,
            ineq_group_distr_samplers,
        )

    def sample_violated_inequalities(self, n_samples: int) -> int:
        """Take random tuples, count how many such tuples violate at least one inequality."""
        return self.tuples_sampler.sample_ineq_violation(
            n_samples, self._rng.integers(0xFFFFFFFF)
        )

    def sample_sizes_cond(self) -> tuple[list[int], int]:
        """Sample number of probes in a gadget, conditioned on violating at least one inequality."""
        return self.tuples_sampler.sample_sizes_cond(self._rng.integers(2**63 - 1))

    def sizes2dict(self, sizes: npt.NDArray[np.uint32]) -> list[dict[Gadget, int]]:
        return [
            {self._leaky_gadgets[gidx]: s for gidx, s in enumerate(sz) if s != 0}
            for sz in tqdm.tqdm(sizes, desc="sizes2dict")
        ]

    def gadgets_eprobes(self) -> list[list[list[gates.Gate]]]:
        return [g.extended_probes(self._gate_leakage) for g in self._leaky_gadgets]

    def sample_set_sizes_multi(self, n_samples: int) -> npt.NDArray[np.uint32]:
        """Number of probes in each gadget, conditioned on not being SNI-provable."""
        res = []
        n_rej = 0
        for _ in tqdm.tqdm(range(n_samples), desc="rejection sampling E (set sizes)"):
            sizes, nr = self.sample_sizes_cond()
            res.append(sizes)
            n_rej += nr
        logging.info(f"Rejected {n_rej} attempts for E")
        return np.array(res, dtype=np.uint32)

    def sample_probes(self, sizes: dict[Gadget, int]) -> list[gates.Gate]:
        """Given number of probes in each gadget, sample the probes."""
        res = []
        for g, s in sizes.items():
            eprobes = g.extended_probes(self._gate_leakage)
            eprobes_idxs = self._rng.choice(
                len(eprobes), size=s, replace=False, shuffle=False
            )
            res.extend(gate for idx in eprobes_idxs for gate in eprobes[idx])
        return res


class GadgetGraph:
    def __init__(
        self,
        p: float,
        outputs: Sequence[Gadget],
        *,
        gate_leakage: bool = True,
        backend,
        max_gadget_outputs=2,
        err_gadget_outputs=logging.error,
        n_bits: int,
    ):
        logging.info("Creating GadgetGraph")
        self.p = p
        self.gate_leakage = gate_leakage
        # Maps each gadget to all gadgets connected to its output.
        gadget_outputs = self._explore_gadgets(outputs)
        for g, uses in gadget_outputs.items():
            g.set_output_uses(uses)
        gadgets = list(gadget_outputs)
        gadget_stats = Counter(map(type, gadgets))
        logging.info(f"Circuit gadgets: {gadget_stats}")

        gate_kind_count = Counter(g.kind() for g in self._list_gates(outputs))
        logging.info(f"Circuit gates: {gate_kind_count}")
        non_random_gates = sum(
            v for k, v in gate_kind_count.items() if k not in ("random", "share")
        )
        logging.info(f"Non-random gates: {non_random_gates}")

        # gadgets restricted to gadgets leaking (i.e., excludes input
        # sharings).
        l_gadgets = [g for g in gadgets if g.leaky(self.gate_leakage)]
        # gadgets restricted to gadgets what compute and for which we
        # therefore need to run a simulator.
        sim_gadgets = [g for g in gadgets if isinstance(g, SimGadget)]
        for g in sim_gadgets:
            go = gadget_outputs[g]
            if len(go) > max_gadget_outputs:
                err_gadget_outputs(f"Gadget {g} has {len(go)} outputs: {go}")

        self.inequalities = [Inequality(g, gadget_outputs[g]) for g in sim_gadgets]
        logging.debug("GadgetGraph: computing sample weights (proba of each ineq)")
        logging.debug("GadgetGraph: structure done")
        all_gates = [
            gate
            for g in l_gadgets
            for eprobe in g.extended_probes(self.gate_leakage)
            for gate in eprobe
        ]
        logging.debug("Building Sampler")
        self.sampler = Sampler(
            np.random.default_rng(SEED),
            self.inequalities,
            self.p,
            self.gate_leakage,
        )
        logging.debug("Building backend")
        self.backend: backends.Backend = backend(list(set(all_gates)), n_bits)

    @staticmethod
    def _list_gates(outputs: Sequence[Gadget]) -> set[gates.Gate]:
        to_explore = [g for og in outputs for g in og.output_sharing()]
        res = set(to_explore)
        while to_explore:
            g = to_explore.pop()
            if isinstance(g, gates.OpGate):
                for op in g.operands:
                    if op not in res:
                        res.add(op)
                        to_explore.append(op)
        return res

    @staticmethod
    def _explore_gadgets(outputs: Sequence[Gadget]) -> dict[Gadget, list[Gadget]]:
        gadgets = dict()  # Maps each gadget to the gadgets connected to its outputs.
        stack = [x for x in outputs]
        gadgets_seen = set(outputs)
        while stack:
            gadget = stack.pop()
            gadgets.setdefault(gadget, [])
            for inp_gadget in gadget.inputs():
                gadgets.setdefault(inp_gadget, []).append(gadget)
                if inp_gadget not in gadgets_seen:
                    stack.append(inp_gadget)
                    gadgets_seen.add(inp_gadget)
        return gadgets

    def bounds_pre_mc(self, nsamples: int, delta: float) -> tuple[float, float]:
        """Probability of not being provable by SNI composition, computed with
        a Monte-Carlo method."""
        n_fail = self.sampler.sample_violated_inequalities(nsamples)
        return bernouilli_bound(nsamples, n_fail, delta)

    def bounds_pre(
        self, max_size: int | None, nsamples: int | None, delta
    ) -> tuple[float, float]:
        """Probability of not being provable by SNI composition
        (Monte-Carlo and inclusion-exclusion methods).
        """
        if max_size is None:
            lb_ie, ub_ie = 0.0, 1.0
        else:
            lb_ie, ub_ie = proba_sim_ie.bounds_pre_ie(
                self.inequalities, self.p, self.gate_leakage, max_size
            )
            logging.info(f"pre IE bounds: [{lb_ie:e}, {ub_ie:e}]")
        if nsamples is None:
            lb_mc, ub_mc = 0.0, 1.0
        else:
            lb_mc, ub_mc = self.bounds_pre_mc(nsamples, delta)
            logging.info(f"pre MC bounds: [{lb_mc:e}, {ub_mc:e}]")
        return max(lb_ie, lb_mc), min(ub_ie, ub_mc)

    def cond_sim_failures(self, n_samples: int) -> int:
        """Probability of simulation failure, given not provable by SNI composition."""
        t_start = time.time()
        if isinstance(self.backend, backends.FastBackend):
            res, n_rej = self.backend.sample_and_check(
                self.sampler.gadgets_eprobes(), self.sampler.tuples_sampler, n_samples
            )
            logging.info(f"Rejected {n_rej} samples")
        else:
            sizes = self.sampler.sample_set_sizes_multi(n_samples)
            sizes = self.sampler.sizes2dict(sizes)
            probes = [
                self.sampler.sample_probes(s)
                for s in tqdm.tqdm(sizes, desc="sample probes")
            ]
            res = self.backend.check_gate_tuples(probes)
        t_end = time.time()
        dt = t_end - t_start
        logging.info(f"Backend execution time (s): {dt}")
        logging.info(f"Backend speed (sample/s): {n_samples/dt}")
        return res

    def compute_rps_mc(
        self,
        n_samples: int,
        delta: float,
        prej_lim: int | None = 3,
        *,
        e_samples: int | None,
    ) -> tuple[float, float]:
        """Top-level random probing security bounds computation."""
        pre_delta = delta / 2
        main_delta = delta if e_samples is None else delta / 2
        logging.info("Start SNI failure probability computation")
        snifail_lb, snifail_ub = self.bounds_pre(prej_lim, e_samples, pre_delta)
        logging.info(f"SNI failure proba bounds: [{snifail_lb:e}, {snifail_ub:e}]")
        n_failures = self.cond_sim_failures(n_samples)
        lb, ub = bernouilli_bound(n_samples, n_failures, main_delta)
        logging.info(f"raw bounds: [{lb:e}, {ub:e}]")
        comp_lb, comp_ub = snifail_lb * lb, snifail_ub * ub
        logging.info(f"Opt bound: [{comp_lb:e}, {comp_ub:e}]")
        return (comp_lb, comp_ub)
