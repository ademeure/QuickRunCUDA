#!/usr/bin/env python3
"""Automatically NOP redundant LOP3 & 0xff byte-extract instructions
by redirecting their UNPACK_B consumers to read from the source register directly.

Proven safe: F2FP.PACK_AB_MERGE_C with C=RZ always zeros upper bytes,
making the 0xff mask redundant. The LOP3 was really just a register copy.

Usage: python3 rename_lop3.py <input.cubin> <output.cubin>
"""
import re, sys, subprocess

def find_kernel_base(cubin_path):
    r = subprocess.run(['cuobjdump', '--dump-elf', cubin_path], capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if '.text.kernel' in line and 'PROGBITS' in line:
            return int(line.split()[1], 16)
    raise RuntimeError("Could not find .text.kernel")

def parse_sass(cubin_path):
    r = subprocess.run(['cuobjdump', '--dump-sass', cubin_path], capture_output=True, text=True)
    instrs = []
    in_kernel = False
    for line in r.stdout.split('\n'):
        if 'Function : kernel' in line: in_kernel = True; continue
        if in_kernel and 'Function :' in line and 'kernel' not in line: break
        if not in_kernel: continue
        m = re.search(r'/\*([0-9a-f]+)\*/', line)
        if m:
            instrs.append((int(m.group(1), 16), line.split('*/')[1].split('/*')[0].strip()))
    return instrs

def build_renames(instrs):
    renames = []
    for i, (off, text) in enumerate(instrs):
        m = re.match(r'LOP3\.LUT R(\d+), R(\d+), 0xff,', text)
        if not m: continue
        dst, src = int(m.group(1)), int(m.group(2))
        consumer = None; src_overwritten = False
        for j in range(i+1, len(instrs)):
            joff, jtext = instrs[j]
            dm = re.match(r'\S+\s+R(\d+)', jtext)
            if dm and int(dm.group(1)) == src and not src_overwritten:
                src_overwritten = True
            um = re.search(r'UNPACK_B R(\d+), R(\d+)', jtext)
            if um and int(um.group(2)) == dst:
                consumer = joff; break
        if not src_overwritten and consumer:
            renames.append((off, dst, src, consumer))
    return renames

def patch(cubin_in, cubin_out):
    kbase = find_kernel_base(cubin_in)
    instrs = parse_sass(cubin_in)
    renames = build_renames(instrs)

    with open(cubin_in, 'rb') as f: data = bytearray(f.read())

    NOP_HI = bytes.fromhex('1879000000000000')
    for lop3_off, dst, src, unpack_off in renames:
        cl = kbase + lop3_off
        data[cl:cl+8] = NOP_HI; data[cl+8] = 0; data[cl+9] = 0
        if dst != src:
            data[kbase + unpack_off + 4] = src

    with open(cubin_out, 'wb') as f: f.write(data)
    print(f"Patched {len(renames)} LOP3 → NOP+rename (kernel base 0x{kbase:x})")

if __name__ == '__main__':
    patch(sys.argv[1], sys.argv[2])
