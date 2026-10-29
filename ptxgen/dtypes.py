"""Data types: Emmy dtype names, their PTX types, register classes, sizes and literals."""

from __future__ import annotations

import struct

# Emmy dtype -> (PTX type of its registers and arithmetic, PTX type of its ld / st). 16-bit
# floats move as raw .b16 (ld / st have no .f16 form); u8 widens into a 16-bit register.
PTX_TYPE = {
    "f32": ("f32", "f32"),
    "f64": ("f64", "f64"),
    "f16": ("f16", "b16"),
    "bf16": ("bf16", "b16"),
    "u8": ("u16", "u8"),
    "i16": ("s16", "s16"),
    "u16": ("u16", "u16"),
    "i32": ("s32", "s32"),
    "u32": ("u32", "u32"),
    "i64": ("s64", "s64"),
    "u64": ("u64", "u64"),
    "bool": ("pred", "pred"),
}
FLOATS = ("f32", "f64", "f16", "bf16")  # PTX float types

# PTX type -> (register prefix, .reg declaration type). Bit-size-compatible types share a class,
# as ptxas expects (shfl.sync.b32 takes an .f32 register).
REG_CLASS = {
    "pred": ("%p", "pred"),
    **dict.fromkeys(("b16", "s16", "u16", "u8", "f16", "bf16"), ("%rs", "b16")),
    **dict.fromkeys(("b32", "s32", "u32"), ("%r", "b32")),
    **dict.fromkeys(("b64", "s64", "u64"), ("%rd", "b64")),
    "f32": ("%f", "f32"),
    "f64": ("%fd", "f64"),
}

BYTES = {"u8": 1, **dict.fromkeys(("b16", "s16", "u16", "f16", "bf16"), 2)}
BYTES |= dict.fromkeys(("b32", "s32", "u32", "f32"), 4) | dict.fromkeys(
    ("b64", "s64", "u64", "f64"), 8
)


def ptx_type(dtype: str) -> str:
    """The register / arithmetic PTX type of an Emmy dtype."""
    if dtype not in PTX_TYPE:
        raise NotImplementedError(f"dtype {dtype}")
    return PTX_TYPE[dtype][0]


def mem_type(dtype: str) -> str:
    """The ld / st PTX type of an Emmy dtype."""
    ptx_type(dtype)
    return PTX_TYPE[dtype][1]


def mov_type(typ: str) -> str:
    """The suffix ``mov`` and ``selp`` take for a register of PTX type ``typ``."""
    return "b16" if typ in ("f16", "bf16") else typ


def lit(value: float, typ: str) -> str:
    """``value`` as an exact PTX immediate of type ``typ`` (hex floats survive bit for bit)."""
    if typ == "f32":
        return f"0f{struct.unpack('<I', struct.pack('<f', value))[0]:08X}"
    if typ == "f64":
        return f"0d{struct.unpack('<Q', struct.pack('<d', value))[0]:016X}"
    if typ == "f16":  # raw bits for mov.b16; past the half range rounds to inf
        try:
            return f"0x{struct.unpack('<H', struct.pack('<e', value))[0]:04X}"
        except OverflowError:
            return "0xFC00" if value < 0 else "0x7C00"
    if typ == "bf16":  # the f32 bits rounded to the top half, to nearest even
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        return f"0x{(bits + 0x7FFF + (bits >> 16 & 1)) >> 16:04X}"
    if not float(value).is_integer():
        raise ValueError(f"{typ} literal {value!r} is not an integer")
    return str(int(value))


def is_int(dtype: str) -> bool:
    """Whether Emmy dtype ``dtype`` is an integer (bool is not)."""
    return dtype != "bool" and ptx_type(dtype) not in FLOATS
