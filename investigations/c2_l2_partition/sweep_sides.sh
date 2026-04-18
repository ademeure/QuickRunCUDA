#!/bin/bash
# Sweep address offsets to find near vs far partition latencies
# Run with single block on one SM; offsets in bytes
cd /root/github/QuickRunCUDA
echo "offset_bytes,cycles_per_atomic"
for off in 0 4096 8192 12288 16384 20480 24576 28672 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864; do
  ./QuickRunCUDA tests/bench_atom_lat_sides.cu -t 1 -b 1 -A 67108864 -0 1000 -1 $off --dump-c raw -T 1 2>/dev/null > /dev/null
  cyc=$(python3 -c "import struct; print(struct.unpack('<Q',open('raw','rb').read(8))[0])")
  cpr=$(python3 -c "print($cyc/1000)")
  echo "$off,$cpr"
done
