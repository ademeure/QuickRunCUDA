# V9: nanosleep cross-thread behavior — WARNING: divergent values get MIN, not MAX

## Measurement (looped 100×, then divided)

Single warp, all 32 threads, __syncwarp at iteration boundaries:

| Mode                                     | per-iter ns | Expected       | Verdict     |
|------------------------------------------|--------------|----------------|-------------|
| All threads sleep 1000 ns                | 1015.3       | ~1000 (correct)| ✓ matches solo |
| **Half (lanes 0-15) sleep 1000, half (16-31) sleep 100** | **127.3** | ~1000 (max)? | **MIN, not MAX** |
| Only lane 0 sleeps 1000 (others no-op)   | 1018.8       | ~1000 (sync waits) | ✓ matches solo |
| Lane N sleeps (100 + N*100) ns           | 351.0        | varies         | NOT max      |

## Critical finding

**When lanes have DIFFERENT nanosleep values, the warp sleeps the SHORTEST amount.**
This contradicts the intuitive expectation that __syncwarp would force the
warp to wait for the longest sleeper.

Measured behavior:
- Half-half (1000 / 100) → ~127 ns (close to MIN of 100)
- Only-one-lane sleeps long → ~1018 ns (the lone-sleeper amount)
- Diverse values 100..3200 → ~351 ns (somewhere in between)

## Implication for persistent kernels

V7 J4 found persistent spin with nanosleep saves 62% power vs spin.
**This works ONLY if ALL THREADS use the same nanosleep value.**

WRONG (divergent values lose effect):
```cuda
if (tid == 0) __nanosleep(1000);  // ← only delays during a warp converge,
else __nanosleep(100);             //   but if all 32 lanes diverge: sleep ~MIN
```

CORRECT (warp-uniform):
```cuda
__nanosleep(1000);  // every lane the same → reliable 1000 ns sleep
```

## 10-rule rigor

1. **Theoretical**: nanosleep is "approximate" per PTX spec; divergent behavior
   not specified.
2. **Measured**: mode 1 = 127 ns ≪ expected MAX of 1000.
3. **Rule 3 N/A**: not throughput.
4. Empirical exploration → MIN-like behavior, not MAX.
5-7. Loop average, multiple modes, syncwarp barrier.
8. **Conclusive**: divergent nanosleep with mixed N values DOES NOT enforce MAX.
9. Initial single-shot test gave noisy 288 ns; loop averaging gave clean 127.
10. **Confidence: HIGH** for "divergent != MAX"; **MEDIUM** on exact MIN-vs-other behavior.

## Persistent kernel power pattern (V7 J4 reaffirmation)

For minimum power in idle persistent kernels:
```cuda
// All warps participate, all lanes use same N
while (true) {
    if (work_available) do_work();
    else __nanosleep(1000);  // warp-uniform = reliable 1 µs sleep
}
```

V7 J4 finding: nanosleep(1000) saves 62% power vs busy-spin. Confirmed only
when all threads in warp use the same value.

## Related: warp scheduler hints

This may be related to the SMSP warp scheduler treating nanosleep as a
priority/quantum hint. With divergent values, scheduler may use
the smallest as the "ready" time for the warp.

## Confidence: HIGH for the warning, MEDIUM for exact mechanism