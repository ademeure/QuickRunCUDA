#!/usr/bin/env python3
"""SASS binary patcher for four_six_fp4 kernel: test LOP3 redundancy.

Usage:
  python3 patch_lop3.py <original.cubin> <output.cubin> <mode>

Modes:
  identity    - Change LOP3 & 0xff to LOP3 & 0xffffffff (proven correct)
  passthrough - Change LOP3 LUT from 0xc0 (A&B) to 0xf0 (A) (proven correct)
  nop_self    - NOP the 3 self-copy LOP3s (R16=R16 etc.) (proven correct)
"""
import sys

KERNEL_BASE = 0x6300  # .text.kernel offset (GPT=1 NC=2 build)
NOP_HI = bytes.fromhex('1879000000000000')

# All LOP3 & 0xff byte-extract instructions (GPT=1 NC=2 COLS_PARAM_CONST=2048)
LOP3_TARGETS = [
    (0x0430, 14, 27), (0x0440, 15, 20),
    (0x0510, 16, 16), (0x0520, 17, 17),  # self-copies
    (0x0610, 23, 16), (0x0620, 22, 17),
    (0x0710, 16, 16),                      # self-copy
    (0x0720, 25, 17),
    (0x0820, 19, 24), (0x0830, 18, 27),
    (0x0900, 19, 16), (0x0910, 18, 17),
    (0x0a00, 17,  4), (0x0a10, 16,  5),
    (0x0b40,  3, 12), (0x0b60,  0, 13),
]

def patch(src, dst, mode):
    with open(src, 'rb') as f:
        data = bytearray(f.read())

    for sass_off, rdst, rsrc in LOP3_TARGETS:
        cubin_off = KERNEL_BASE + sass_off
        if mode == 'identity':
            # Change immediate from 0xff to 0xffffffff
            data[cubin_off+5] = 0xff
            data[cubin_off+6] = 0xff
            data[cubin_off+7] = 0xff
        elif mode == 'passthrough':
            data[cubin_off+9] = 0xf0  # LUT: A passthrough
        elif mode == 'nop_self':
            if rdst == rsrc:
                data[cubin_off:cubin_off+8] = NOP_HI
                data[cubin_off+8] = 0x00
                data[cubin_off+9] = 0x00

    with open(dst, 'wb') as f:
        f.write(data)

if __name__ == '__main__':
    patch(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"Patched {sys.argv[1]} → {sys.argv[2]} (mode={sys.argv[3]})")
