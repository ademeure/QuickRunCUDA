#!/bin/bash
# V8 D3: sparse vs dense mma.sync power
sudo nvidia-smi -i 0 -lgc 1500,1500 > /dev/null 2>&1
sleep 1

for M in 0 1; do
    BASE_W=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | head -1)
    sleep 0.3

    # Persistent to keep GPU busy during sampling
    timeout 30 ./QuickRunCUDA -f tests/bench_v8_d1_sparse_mma.cu -t 32 -p \
        -H "#define MODE $M" -0 100000 -A 1024 -B 1024 -C 1024 > /dev/null 2>&1 &
    KERN_PID=$!
    sleep 1.5

    PSUM=0
    for i in 1 2 3 4 5 6; do
        P=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | head -1)
        PSUM=$(echo "$PSUM + $P" | bc); sleep 0.2
    done
    AVG=$(echo "scale=2; $PSUM / 6" | bc)
    ACT=$(echo "scale=2; $AVG - $BASE_W" | bc)
    pkill -9 QuickRunCUDA 2>/dev/null
    wait $KERN_PID 2>/dev/null
    sleep 1

    if [ $M -eq 0 ]; then
        echo "Dense HMMA:  Pavg=$AVG W Pact=$ACT W"
    else
        echo "Sparse HMMA.SP: Pavg=$AVG W Pact=$ACT W"
    fi
done
