"""Buffers a kernel reads or writes, global or shared: each knows its shape and dtype and how
to address, load and store element i."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from ptxgen import index, ops
from ptxgen.builder import PTXKernel
from ptxgen.dtypes import BYTES, mem_type
from ptxgen.index import Lin
from ptxgen.ops import Val


@dataclass(frozen=True)
class Memory(ABC):
    """A buffer the kernel reads or writes: its shape, its dtype, and how to reach element i."""

    shape: tuple[int, ...]
    dtype: str

    @abstractmethod
    def address(self, k: PTXKernel, flat: Lin, dtype: str) -> tuple[str, int, str]:
        """(address register, byte offset, state space) of element ``flat``; the constant part
        of the index becomes the offset."""

    def load(self, k: PTXKernel, flat: Lin, dtype: str, n: int = 1) -> list[Val]:
        addr, off, space = self.address(k, flat, dtype)
        regs = ops.load(k, space, addr, off, mem_type(dtype), n, cached=space == "global.nc")
        return [Val(r, dtype) for r in regs]

    def store(self, k: PTXKernel, flat: Lin, dtype: str, values: Sequence[Val]) -> None:
        addr, off, space = self.address(k, flat, dtype)
        regs = [ops.convert(k, v, dtype).reg for v in values]
        ops.store(k, space, addr, off, mem_type(dtype), regs)


@dataclass(frozen=True)
class Global(Memory):
    ptr: str  # u64 register holding the buffer's global address
    is_output: bool

    def address(self, k: PTXKernel, flat: Lin, dtype: str) -> tuple[str, int, str]:
        size, var, addr = BYTES[mem_type(dtype)], flat.var_part(), self.ptr
        if not var.is_const:
            wide = k.pure("mul.wide.u32", "u64", index.reg_of(k, var), str(size))
            addr = k.pure("add.s64", "u64", addr, wide)
        return addr, flat.const * size, "global" if self.is_output else "global.nc"


@dataclass(frozen=True)
class Shared(Memory):
    base: str  # u32 register holding the address of the .shared symbol
    offset: int  # byte offset of the buffer from it

    def address(self, k: PTXKernel, flat: Lin, dtype: str) -> tuple[str, int, str]:
        size, var, addr = BYTES[mem_type(dtype)], flat.var_part(), self.base
        if not var.is_const:
            addr = k.pure("mad.lo.s32", "u32", index.reg_of(k, var), str(size), self.base)
        return addr, self.offset + flat.const * size, "shared"
