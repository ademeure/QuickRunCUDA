#!/usr/bin/env bash
# Drive bench_pipe_matrix.cu across solo baselines and pairs.
# Usage: ./run_pipe_matrix.sh [pair|solo|all]
set -uo pipefail

cd /root/github/QuickRunCUDA
export CUDA_VISIBLE_DEVICES=0

SMS=148
CLOCK_HZ=1920000000
THREADS=512
BLOCKS=592      # 4x SM oversubscription
ITERS=2048
UNROLL=16
TIMED=20

# Peak per-SM-per-cycle for each op (theoretical, thread-level PTX insts/SM/clock)
declare -A PEAK
PEAK[FFMA]=128
PEAK[FMUL]=128
PEAK[IMAD]=64
PEAK[IADD3]=128
PEAK[LOP3]=128
PEAK[SHL]=64
PEAK[HFMA2]=64
PEAK[HADD2]=64
PEAK[UNPACK]=32
PEAK[PACK]=32
PEAK[EX2]=32
PEAK[RSQ]=16

# SASS pattern for counting the canonical opcode of each op (inside inner loop)
declare -A SASS_PAT
SASS_PAT[FFMA]=' FFMA R'
SASS_PAT[FMUL]=' FMUL R'
SASS_PAT[IMAD]=' IMAD R'
SASS_PAT[IADD3]=' IADD3 R'
SASS_PAT[LOP3]=' LOP3\.LUT '
SASS_PAT[SHL]=' SHF\.L\.U32 '
SASS_PAT[HFMA2]=' HFMA2 '
SASS_PAT[HADD2]=' HADD2 '
SASS_PAT[UNPACK]='F2FP\.F16\.E4M3\.UNPACK_B '
SASS_PAT[PACK]='F2FP\.SATFINITE\.E4M3\.F16\.UNPACK_B_MERGE_C'
SASS_PAT[EX2]='MUFU\.EX2'
SASS_PAT[RSQ]='MUFU\.RSQ'

run_one() {
    # $1 = header defines, $2 = total_ops_per_thread (per kernel call)
    local header="$1"
    local N="$2"
    local mult
    mult=$(python3 -c "print($N*1e-9)")
    local out
    out=$(./QuickRunCUDA tests/bench_pipe_matrix.cu -H "$header" -t $THREADS -b $BLOCKS -0 $ITERS -T $TIMED -N $mult -U GOps/s 2>&1)
    local gops
    gops=$(echo "$out" | tail -1 | sed -E 's/.*==> ([0-9.]+) GOps.*/\1/')
    echo "$gops"
}

latest_sass() {
    ls -t /root/github/QuickRunCUDA/sass/bench_pipe_matrix_*.sass 2>/dev/null | head -1
}

count_sass() {
    local pat="$1"
    local f
    f=$(latest_sass)
    grep -cE "$pat" "$f" 2>/dev/null
}

rate_per_sm_per_cycle() {
    # $1 = GOps/s total (float), $2 = dyn_ops_per_thread, ${3}=threads, ${4}=blocks
    # r = GOps/s * 1e9 / (SMS * CLOCK_HZ)  — already instruction-rate per /SM/cycle at threads*blocks coverage.
    # Actually GOps/s already is total thread-level ops rate, so:
    # per_SM_per_cycle = gops_s * 1e9 / (SMS * CLOCK_HZ)
    python3 -c "print(f'{$1 * 1e9 / ($SMS * $CLOCK_HZ):.3f}')"
}

solo() {
    local op="$1"
    local N="$2"
    local header="#define N_${op} ${N}
#define UNROLL ${UNROLL}"
    local dyn=$(( N * ITERS ))
    local g
    g=$(run_one "$header" "$dyn")
    local r
    r=$(rate_per_sm_per_cycle "$g" "$dyn")
    local peak=${PEAK[$op]}
    local pct
    pct=$(python3 -c "print(f'{$r/$peak*100:.1f}')")
    # SASS verification
    local pat=${SASS_PAT[$op]}
    local cnt
    cnt=$(count_sass "$pat")
    local expected=$(( N * UNROLL ))
    printf "  %-8s N=%2d  GOps/s=%8.1f  /SM/clk=%6.2f  peak=%3d  util=%5.1f%%  sass=%d/exp=%d\n" \
        "$op" "$N" "$g" "$r" "$peak" "$pct" "$cnt" "$expected"
}

pair() {
    local opA="$1" NA="$2"
    local opB="$3" NB="$4"
    local header="#define N_${opA} ${NA}
#define N_${opB} ${NB}
#define UNROLL ${UNROLL}"
    local dyn=$(( (NA + NB) * ITERS ))
    local g
    g=$(run_one "$header" "$dyn")
    # Approximate: assume both ops ran at equal fraction of the total time.
    # Fair split: rate_A = (NA/(NA+NB)) * total_rate; rate_B = NB-fraction.
    local r_total
    r_total=$(rate_per_sm_per_cycle "$g" "$dyn")
    local r_A
    r_A=$(python3 -c "print(f'{$r_total * $NA / ($NA + $NB):.3f}')")
    local r_B
    r_B=$(python3 -c "print(f'{$r_total * $NB / ($NA + $NB):.3f}')")
    local peakA=${PEAK[$opA]}
    local peakB=${PEAK[$opB]}
    local u
    u=$(python3 -c "print(f'{$r_A/$peakA + $r_B/$peakB:.3f}')")
    # SASS verify
    local patA=${SASS_PAT[$opA]}
    local patB=${SASS_PAT[$opB]}
    local cntA cntB
    cntA=$(count_sass "$patA")
    cntB=$(count_sass "$patB")
    local expA=$(( NA * UNROLL ))
    local expB=$(( NB * UNROLL ))
    printf "  %-8s(%2d)+%-8s(%2d)  GOps/s=%8.1f  rA=%6.2f/pk%3d  rB=%6.2f/pk%3d  u=%5.3f  sassA=%d/%d sassB=%d/%d\n" \
        "$opA" "$NA" "$opB" "$NB" "$g" "$r_A" "$peakA" "$r_B" "$peakB" "$u" "$cntA" "$expA" "$cntB" "$expB"
}

# ------- SOLO -------
if [[ "${1:-all}" == "solo" || "${1:-all}" == "all" ]]; then
    echo "=== SOLO BASELINES ==="
    solo FFMA 16
    solo FMUL 16
    solo IMAD 16
    solo IADD3 16
    solo LOP3 16
    solo SHL 24
    solo HFMA2 48
    solo HADD2 48
    solo UNPACK 24
    solo PACK 16
    solo EX2 16
    solo RSQ 16
fi

# ------- PAIRS -------
if [[ "${1:-all}" == "pair" || "${1:-all}" == "all" ]]; then
    echo
    echo "=== PAIRS (balanced ILP) ==="
    # Balanced-ILP pairs. For each pair, pick N_A, N_B so that:
    #   demand_A = N_A / peak_A  and  demand_B = N_B / peak_B
    # are (approximately) equal, AND each N is large enough to hide latency
    # (4cy × issue_width; for peak=128 need N ≥ 16; for peak=32 need N ≥ 4; peak=16 need N ≥ 2).

    # Integer pipe suspects — LOP3(128), IADD3(128), SHL(64), IMAD(64)
    pair LOP3 16  IADD3 16                # both 128
    pair LOP3 16  SHL 8                   # 128 vs 64  → demand ratio 1:1
    pair IADD3 16 SHL 8                   # 128 vs 64
    pair LOP3 16  IMAD 8                  # 128 vs 64
    pair IADD3 16 IMAD 8                  # 128 vs 64
    pair SHL 8    IMAD 8                  # 64 vs 64

    # IMAD vs FMA pipe
    pair IMAD 8  FFMA 16                  # 64 vs 128, balanced demand
    pair IMAD 8  FMUL 16

    # FFMA vs other pipes
    pair FFMA 16 LOP3 16
    pair FFMA 16 IADD3 16
    pair FFMA 16 SHL 8                    # balanced
    pair FFMA 16 FMUL 16
    pair FFMA 16 HFMA2 8                  # 128 vs 64
    pair FFMA 16 HADD2 8

    # Half pipe
    pair HFMA2 8 HADD2 8                  # both 64
    pair HFMA2 8 LOP3 16                  # 64 vs 128
    pair HFMA2 8 IMAD 8                   # 64 vs 64
    pair HADD2 8 LOP3 16

    # F2FP vs FMA pipe / INT pipe / each other
    pair PACK 4   FFMA 16                 # 32 vs 128
    pair PACK 4   IMAD 8                  # 32 vs 64
    pair PACK 4   LOP3 16
    pair PACK 4   IADD3 16
    pair PACK 4   UNPACK 4                # both 32 (obs solo)
    pair UNPACK 4 FFMA 16
    pair UNPACK 4 IMAD 8
    pair UNPACK 4 LOP3 16

    # SFU internal
    pair EX2 4 RSQ 2                      # 32 vs 16
    pair EX2 4 PACK 4                     # both 32
    pair EX2 4 UNPACK 4                   # both 32
    pair RSQ 2 PACK 4                     # 16 vs 32
    pair EX2 4 FFMA 16                    # 32 vs 128
    pair EX2 4 IMAD 8                     # 32 vs 64
    pair RSQ 2 FFMA 16                    # 16 vs 128
    pair RSQ 2 LOP3 16

    # F2FP.PACK + IMAD specifically (question 6)
    pair PACK 4 IMAD 8

    # F2FP.PACK + FFMA specifically (question 5)
    pair PACK 4 FFMA 16
fi
