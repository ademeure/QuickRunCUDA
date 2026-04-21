#!/bin/bash
# V7 J4: persistent kernel power profile
# Sample power during persistent kernel idle (waiting) vs busy (processing)
sudo nvidia-smi -i 0 -lgc 1500,1500 > /dev/null 2>&1
sleep 1

BASE_W=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0)
echo "Baseline (no GPU work): $BASE_W W"

# Launch persistent kernel that just spins waiting
cat > /tmp/persist_idle.cu << 'EOF'
extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    volatile unsigned int* sig = (volatile unsigned int*)A;
    for (int i = 0; i < ITERS; i++) {
        while (*sig != (unsigned)i) {}
    }
}
EOF

# 1 SM persistent + spinning
timeout 30 ./QuickRunCUDA -f /tmp/persist_idle.cu -t 32 -b 1 -0 1000000000 -A 1024 -B 1024 -C 1024 > /dev/null 2>&1 &
KERN_PID=$!
sleep 2

PSUM=0
for i in 1 2 3 4 5 6; do
    PSAMP=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0)
    PSUM=$(echo "$PSUM + $PSAMP" | bc); sleep 0.2
done
AVG_P=$(echo "scale=2; $PSUM / 6" | bc)
ACT_P=$(echo "scale=2; $AVG_P - $BASE_W" | bc)
echo "Persistent kernel SPINNING (1 SM, 1 thread): Pavg=$AVG_P W Pact=$ACT_P W"

pkill -9 QuickRunCUDA 2>/dev/null; sleep 2
