# Copyright (c) 2022, NVIDIA CORPORATION.

from typing import Any, Set, Tuple, TypeVar

# Note: It may be possible to define a narrower bound here eventually.
BinaryOperandType = TypeVar("BinaryOperandType", bound="Any")

class BinaryOperand:
    _SUPPORTED_BINARY_OPERATIONS: Set

    def _binaryop(self, other: BinaryOperandType, op: str):
        ...

    def __add__(self, other):
        ...

    def __sub__(self, other):
        ...

    def __mul__(self, other):
        ...

    def __truediv__(self, other):
        ...

    def __floordiv__(self, other):
        ...

    def __mod__(self, other):
        ...

    def __pow__(self, other):
        ...

    def __and__(self, other):
        ...

    def __xor__(self, other):
        ...

    def __or__(self, other):
        ...

    def __radd__(self, other):
        ...

    def __rsub__(self, other):
        ...

    def __rmul__(self, other):
        ...

    def __rtruediv__(self, other):
        ...

    def __rfloordiv__(self, other):
        ...

    def __rmod__(self, other):
        ...

    def __rpow__(self, other):
        ...

    def __rand__(self, other):
        ...

    def __rxor__(self, other):
        ...

    def __ror__(self, other):
        ...

    def __lt__(self, other):
        ...

    def __le__(self, other):
        ...

    def __eq__(self, other):
        ...

    def __ne__(self, other):
        ...

    def __gt__(self, other):
        ...

    def __ge__(self, other):
        ...

    @staticmethod
    def _check_reflected_op(op) -> Tuple[bool, str]:
        ...
