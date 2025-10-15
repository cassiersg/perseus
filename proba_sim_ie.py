"""
Compute probability of a random tuple of probes not being simulatable by SNI
composition (deterministic algorithm).
"""

from dataclasses import dataclass
import copy
import itertools as it
import logging
import math

import tqdm

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph
from pgmpy.inference import VariableElimination

import gadget
from graph_inequality import Inequality


def strict_subsets(iterable):
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)))


@dataclass(frozen=True, eq=True)
class IneqStruct:
    t: int
    gadget_ids: list[int]


@dataclass(frozen=True, eq=True)
class IneqSetHash:
    gadget_sizes: tuple  # must be sorted
    gadget_occurences: tuple  # must be sorted


@dataclass(frozen=True, eq=True)
class IneqSetStruct:
    ineqs: list[IneqStruct]
    gadget_sizes: list[int]

    @classmethod
    def from_inequalities(
        cls, ineqs: list["Inequality"], gate_leakage: bool
    ) -> "IneqSetStruct":
        g2id = dict()
        res_ineqs = [
            IneqStruct(
                t=ineq.t(),
                gadget_ids=[
                    g2id.setdefault(g, len(g2id)) for g in [ineq.g] + ineq.outputs
                ],
            )
            for ineq in ineqs
        ]
        gadget_sizes = [0] * len(g2id)
        for g, id in g2id.items():
            gadget_sizes[id] = g.n_probes(gate_leakage)
        return cls(res_ineqs, gadget_sizes)

    def struct_hash(self) -> IneqSetHash:
        occurences = [0 for _ in self.gadget_sizes]
        for ineq in self.ineqs:
            for i in ineq.gadget_ids:
                occurences[i] += 1
        return IneqSetHash(tuple(sorted(self.gadget_sizes)), tuple(sorted(occurences)))

    # Should not be used
    def matches(self, other: "IneqSetStruct") -> bool:
        if len(self.ineqs) != len(other.ineqs):
            return False
        other_id_map: list[None | int] = [None for _ in other.gadget_sizes]
        for si, oi in zip(self.ineqs, other.ineqs):
            if si.t != oi.t or len(si.gadget_ids) != len(oi.gadget_ids):
                return False
            for s_id, o_id in zip(si.gadget_ids, oi.gadget_ids):
                if other_id_map[o_id] is None:
                    other_id_map[o_id] = s_id
                elif other_id_map[o_id] != s_id:
                    return False
        return True

    def match_ineq(
        self,
        other: "IneqSetStruct",
        other_id_map: list[None | int],
        si: IneqStruct,
        oi: IneqStruct,
    ) -> None | list[None | int]:
        if si.t != oi.t or len(si.gadget_ids) != len(oi.gadget_ids):
            return None
        other_id_map = copy.copy(other_id_map)
        for s_id, o_id in zip(si.gadget_ids, oi.gadget_ids):
            if other_id_map[o_id] is None:
                if self.gadget_sizes[s_id] != other.gadget_sizes[o_id]:
                    return None
                other_id_map[o_id] = s_id
            elif other_id_map[o_id] != s_id:
                return None
        return other_id_map

    def continue_match(
        self,
        other: "IneqSetStruct",
        other_id_map: list[None | int],
        sis: list[IneqStruct],
        ois: list[IneqStruct],
    ) -> bool:
        if sis == ois == []:
            return True
        for i, oi in enumerate(ois):
            new_map = self.match_ineq(other, other_id_map, sis[0], oi)
            if new_map is not None:
                if self.continue_match(other, new_map, sis[1:], ois[:i] + ois[i + 1 :]):
                    return True
        return False

    def matches_any_order(self, other: "IneqSetStruct") -> bool:
        if len(self.ineqs) != len(other.ineqs):
            return False
        other_id_map: list[None | int] = [None for _ in other.gadget_sizes]
        res = self.continue_match(other, other_id_map, self.ineqs, other.ineqs)
        return res


def _compute_adjacent_ineqs(
    inequalities: list[Inequality],
) -> dict[Inequality, set[Inequality]]:
    ineq_map = {ineq.g: ineq for ineq in inequalities}
    adjacent_ineqs = {ineq: set() for ineq in inequalities}
    for ineq in inequalities:
        inputs = [
            ineq_map[g] for g in ineq.g.inputs() if isinstance(g, gadget.SimGadget)
        ]
        for in_ineq in inputs:
            adjacent_ineqs[in_ineq].add(ineq)
            adjacent_ineqs[ineq].add(in_ineq)
        for ineq1, ineq2 in it.combinations(inputs, 2):
            adjacent_ineqs[ineq1].add(ineq2)
            adjacent_ineqs[ineq2].add(ineq1)
    return adjacent_ineqs


def build_clusters(
    inequalities: list[Inequality], max_size: int
) -> list[frozenset[Inequality]]:
    adjacent_ineqs = _compute_adjacent_ineqs(inequalities)
    clusters = [{frozenset([ineq]) for ineq in inequalities}]
    for _ in range(2, max_size + 1):
        clusters_new = set()
        for cluster in clusters[-1]:
            adj = set().union(*(adjacent_ineqs[ineq] for ineq in cluster)) - cluster
            for ineq in adj:
                clusters_new.add(frozenset(cluster | {ineq}))
        clusters.append(clusters_new)
    return [c for clusterl in clusters for c in clusterl]


def pr_ineqs(ineqs: list[Inequality], p: float, gate_leakage: bool) -> float:
    """Pr[\bigwedge_i E_i]"""
    # Algorithm:
    # We use variable elimination, on a factor graph made of
    # 1 variable node for each N_i, and a factor node for each E_i.

    # All gadgets at outputs of `gadgets`
    all_g = list(
        set(ineq.g for ineq in ineqs).union(*(set(ineq.outputs) for ineq in ineqs))
    )
    assert all(isinstance(g, gadget.SimGadget) for g in all_g)
    max_t = max(ineq.t() for ineq in ineqs)
    tsuff_g = {g: min(max_t + 1, g.n_probes(gate_leakage)) for g in all_g}
    graph = FactorGraph()
    # graph.check_model = lambda: None
    graph.add_nodes_from([g.name() for g in all_g])
    for ineq in ineqs:
        factor_value = [
            int(sum(n) > ineq.t())
            for n in it.product(
                *[
                    [cnt * x for x in range(tsuff_g[go] + 1)]
                    for go, cnt in ineq.all_g_cnt.items()
                ]
            )
        ]
        factor = DiscreteFactor(
            [g.name() for g in ineq.all_g_cnt],
            [tsuff_g[g] + 1 for g in ineq.all_g_cnt],
            factor_value,
        )
        graph.add_factors(factor)
        graph.add_edges_from([(g.name(), factor) for g in ineq.all_g_cnt])
    # graph.check_model()
    for g in all_g:
        factor = DiscreteFactor(
            [g.name()],
            [tsuff_g[g] + 1],
            g.n_probe_distr_group_t(p, tsuff_g[g], gate_leakage),
        )
        graph.add_factors(factor)
        graph.add_edge(g.name(), factor)
    return VariableElimination(graph).query([]).get_value()


def inclusion_exclusion_cluster(
    inequalities: list[Inequality],
    p: float,
    gate_leakage: bool,
    max_size: int,
) -> list[float]:
    clusters = build_clusters(inequalities, max_size)  # sorted by size
    clusters_set = set(clusters)

    # Probability of joint Ei for each cluster.
    logging.info("start cluster aggregation")
    cluster_struct = [
        IneqSetStruct.from_inequalities(list(c), gate_leakage) for c in clusters
    ]
    logging.info("cluster hashing done")
    cluster_structs_by_hash = dict()
    ref_clusters = set()
    cluster2ref = dict()
    for i, cs in enumerate(cluster_struct):
        for j in cluster_structs_by_hash.get(cs.struct_hash(), set()):
            if cs.matches_any_order(cluster_struct[j]):
                # if True:
                cluster2ref[i] = j
                break
        if i not in cluster2ref:
            ref_clusters.add(i)
            cluster2ref[i] = i
            cluster_structs_by_hash.setdefault(cs.struct_hash(), set()).add(i)
    logging.info("end cluster aggregation")
    p_cluster_ref: dict[int, float] = dict()
    for c in tqdm.tqdm(ref_clusters, desc="P_C"):
        p_cluster_ref[c] = pr_ineqs(list(clusters[c]), p, gate_leakage)
        assert isinstance(p_cluster_ref[c], float)
    p_cluster = {c: p_cluster_ref[cluster2ref[i]] for i, c in enumerate(clusters)}

    # computing T_C
    p_prime: dict[frozenset[Inequality], float] = dict()
    t_cluster: dict[frozenset[Inequality], float] = dict()

    def split(s):
        s = list(s)
        i = s.pop()
        srem = frozenset(s)
        return srem, i

    def get_p_prime(s: frozenset[Inequality]) -> float:
        srem, i = split(s)
        if s in p_prime:
            pass
        elif s in clusters_set:
            p_prime[s] = p_cluster[s]
        else:
            res = 0.0
            for sub in strict_subsets(srem):
                c_sub = frozenset(list(sub) + [i])
                if c_sub in clusters_set:
                    res += t_cluster[c_sub] * get_p_prime(srem - c_sub)
            p_prime[s] = res
        return p_prime[s]

    for c in tqdm.tqdm(clusters, desc="T_C"):
        res = get_p_prime(c)
        crem, i = split(c)
        for s in strict_subsets(crem):
            c_sub = frozenset(list(s) + [i])
            if c_sub in clusters_set:
                res -= t_cluster[c_sub] * get_p_prime(crem - c_sub)
        t_cluster[c] = res

    # Induction formula for B'_s.
    s_ab = [1.0] + [0.0 for _ in range(max_size)]
    for c in tqdm.tqdm(clusters, desc="Aggreg T_C"):
        tc = t_cluster[c]
        s_ab = [
            s + (tc * s_ab[i - len(c)] if i >= len(c) else 0.0)
            for i, s in enumerate(s_ab)
        ]
    ie = [0.0] + [
        (-1) ** (size + 1) * b_prime
        for b_prime, size in zip(s_ab[1:], range(1, max_size + 1))
    ]
    return ie


def inclusion_exclusion_naive(
    inequalities: list[Inequality], p: float, gate_leakage: bool, max_size: int
) -> list[float]:
    bar_len = sum(math.comb(len(inequalities), s) for s in range(1, max_size + 1))
    inclusion_exclusion = [0.0]
    with tqdm.tqdm(total=bar_len, desc="Pr[E]") as pbar:
        for size in range(1, max_size + 1):
            p = 0.0
            for ineqs in it.combinations(inequalities, size):
                p += pr_ineqs(list(ineqs), p, gate_leakage)
                pbar.update(1)
            inclusion_exclusion.append((-1) ** (size + 1) * p)
    return inclusion_exclusion


def bounds_pre_ie(
    inequalities: list[Inequality], p: float, gate_leakage: bool, max_size: int
) -> tuple[float, float]:
    if max_size <= 3:
        ie = inclusion_exclusion_cluster(inequalities, p, gate_leakage, max_size)
    else:
        logging.warning("Use inefficient algorithm for IE bound")
        ie = inclusion_exclusion_naive(inequalities, p, gate_leakage, max_size)
    bounds = [sum(ie[:-1]), sum(ie)]
    return (min(bounds), max(bounds))
