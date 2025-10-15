from collections import Counter

from gadget import SimGadget, Gadget


class Inequality:
    """Inequality related to the output of a gadget in a gadget graph.

    For a t-SNI gadget, the total number of probes (sum over the gadget and all
    gadgets connected to the output, taking multiplicity into account) must be
    at most t.
    """

    def __init__(self, g: SimGadget, outputs: list[Gadget]):
        self.g = g
        # All gadgets connected to the output of g.
        self.outputs = outputs
        self.outputs_cnt = Counter(outputs)
        self.all_g_cnt = self.outputs_cnt.copy()
        self.all_g_cnt.update([g])

    def __repr__(self):
        return repr(self.g)

    def t(self) -> int:
        return self.g.t_sni()

    def all_g(self) -> list[Gadget]:
        return [self.g] + self.outputs

    def sim_outputs(self) -> list[SimGadget]:
        return [o for o in self.outputs if isinstance(o, SimGadget)]
