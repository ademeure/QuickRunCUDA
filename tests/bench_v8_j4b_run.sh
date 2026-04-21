#!/bin/bash
# V8 J4b: power for each persistent kernel variant
sudo nvidia-smi -i 0 -lgc 1500,1500 > /dev/null 2>&1
sleep 1

NAMES=("1 SM x 1 thr SPIN" "148 SMs x 1 thr SPIN" "148 SMs x 256 thr SPIN" "148 SMs x 256 thr NANOSLEEP")
BLOCKS=(1 148 148 148)
THREADS=(1 32 256 256)

# We pass threads=N via -t; the kernel filters internally for MODEs 0/1
# For MODE 0 we want b=1 t=32 (kernel filters to 1 thread)
# For MODE 1 we want b=148 t=32 (kernel filters to thread 0 of each block)
# For MODE 2,3 we want b=148 t=256

CONFIGS=("0|1|32" "1|148|32" "2|148|256" "3|148|256")

BASE_W=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | head -1)
echo "Baseline (idle GPU): $BASE_W W"
echo ""
printf "%-32s %-9s %-9s\n" "Variant" "Pavg(W)" "Pact(W)"
echo "----------------------------------------------------"

for cfg in "${CONFIGS[@]}"; do
    M=$(echo $cfg | cut -d'|' -f1)
    B=$(echo $cfg | cut -d'|' -f2)
    T=$(echo $cfg | cut -d'|' -f3)
    NAME=${NAMES[$M]}

    # Launch kernel that spins forever-ish (10 sec)
    timeout 30 ./QuickRunCUDA -f tests/bench_v8_j4b_persistent_all_sms.cu -t $T -b $B \
        -H "#define MODE $M" -0 999999 -A 1024 -B 1024 -C 1024 > /dev/null 2>&1 &
    KERN_PID=$!
    sleep 1.5

    PSUM=0
    for i in 1 2 3 4 5 6; do
        PSAMP=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | head -1)
        PSUM=$(echo "$PSUM + $PSAMP" | bc); sleep 0.2
    done
    AVG_P=$(echo "scale=2; $PSUM / 6" | bc)
    ACT_P=$(echo "scale=2; $AVG_P - $BASE_W" | bc)

    pkill -9 QuickRunCUDA 2>/dev/null
    wait $KERN_PID 2>/dev/null
    sleep 1

    printf "%-32s %-9s %-9s\n" "$NAME" "$AVG_P" "$ACT_P"
done
