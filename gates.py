"""
Description of circuits as gates (i.e. expressions).
"""

import abc
import enum


class Gate:
    all_gates: dict[str, "Gate"] = dict()
    prefixes: dict[str, int] = {}

    def __init__(self, name: str):
        self.name = name
        self.all_gates[name] = self

    @classmethod
    def get_or_new(cls, name: str, *args, **kwargs) -> "Gate":
        if name in cls.all_gates:
            return cls.all_gates[name]
        else:
            return cls(name, *args, **kwargs)

    @classmethod
    def new_unique(cls, prefix: str, *args, **kwargs):
        """Create gate, ensuring fresh name."""
        if prefix not in cls.prefixes:
            cls.prefixes[prefix] = 0
        while True:
            name = prefix + "_" + str(cls.prefixes[prefix])
            cls.prefixes[prefix] += 1
            if name not in cls.all_gates:
                return cls.get_or_new(name, *args, **kwargs)

    @abc.abstractmethod
    def _operands(self) -> list["Gate"]:
        pass

    @abc.abstractmethod
    def kind(self) -> str:
        pass

    def leak_gate(self, gate_leakage: bool) -> list["Gate"]:
        if gate_leakage:
            return self._operands()
        else:
            return [self]

    def set_of_probes(self) -> set["Gate"]:
        x: set["Gate"] = set()
        new_x: set["Gate"] = set([self])
        while new_x != x:
            x = new_x
            new_x = x.union(*(op._operands() for op in x))
        return x


class RndGate(Gate):
    def __init__(self, name: str):
        super().__init__(name)

    def _operands(self):
        return []

    def kind(self) -> str:
        return "random"

    def __repr__(self):
        return f"RndGate({self.name})"


class CstGate(Gate):
    def __init__(self, name: str, value: int):
        super().__init__(name)
        self._value = value

    @property
    def value(self) -> int:
        return self._value

    def _operands(self):
        return []

    def kind(self) -> str:
        return "constant"

    def __repr__(self):
        return f"CstGate({self.name})"


class ShareGate(Gate):
    def __init__(self, name: str, sharing_name: str, share_idx: int, nshares: int):
        super().__init__(name)
        self.sharing_name = sharing_name
        self.share_idx = share_idx
        self.nshares = nshares

    def _operands(self):
        return []

    def kind(self) -> str:
        return "share"

    def __repr__(self):
        return f"ShareGate({self.sharing_name}[{self.share_idx}])"


class OpKind(enum.Enum):
    # We work in a characteristic-2 field
    ADD = enum.auto()
    MUL = enum.auto()
    NEG = enum.auto()
    NOP = enum.auto()
    MUL2 = enum.auto()
    MUL3 = enum.auto()
    MULC = enum.auto()
    AFFINE = enum.auto()
    ADDPUB = enum.auto()
    SQUARE = enum.auto()


class OpGate(Gate):
    def __init__(self, name: str, op: OpKind, operands: list[Gate]):
        super().__init__(name)
        self.op = op
        self.operands = operands

    def _operands(self):
        return self.operands

    def kind(self) -> str:
        return self.op.name

    def __repr__(self):
        return f"OpGate({self.name}, {self.op})"


def xor_sharing(name, d) -> list[Gate]:
    return [ShareGate(f"{name}[{i}]", name, i, d) for i in range(d)]
