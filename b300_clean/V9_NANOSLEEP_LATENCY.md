# V9: __nanosleep(N) actual latency — HW rounds up, not truly "N ns"

## Measurements

Single-thread measurements (100-iteration average per N) via `%globaltimer`:

| Requested N (ns) | Actual (ns) | Ratio | Note                     |
|------------------|-------------|-------|--------------------------|
| 100              | 128         | 1.28× | Floor ≈ 128 ns            |
| 500              | 510         | 1.02× | Near-exact                |
| 1,000            | 1018        | 1.02× | Near-exact                |
| 5,000            | **8,171**   | **1.63×** | Rounded up                 |
| 10,000           | **16,384**  | **1.64×** | = 2^14 exactly            |
| 50,000           | 65,372      | 1.31× |                          |
| 100,000          | **131,072** | **1.31×** | = 2^17 exactly            |
| 500,000          | 521,666     | 1.04× | Large values near-exact   |

## Findings

1. **Floor ≈ 128 ns**: even nanosleep(0) or very small values cost ~128 ns.
2. **500-1000 ns sweet spot**: minimal overhead in this range.
3. **5-10 µs → 64% overhead**: HW rounds up to next power of 2 apparently
   (16384 = 2^14, 131072 = 2^17).
4. **Large (500 µs+) ≈ accurate**: relative overhead drops as N grows.
5. **Maximum per doc = 1 ms**, but real ceiling in this test 521 µs for N=500K.

## Implication for persistent kernel power (V7 J4)

V7 J4 found persistent spin with nanosleep(1µs) saves 62% power vs spin.
Now confirmed nanosleep(1µs) actually sleeps ~1018 ns — near exact.
Slightly larger nanosleep values might waste time: nanosleep(5000) actually
sleeps 8171 ns.

For power-saving persistent kernel loops, use **nanosleep(1000)** for the
predictable 1 µs sleep. Avoid the 5-100 µs range where rounding doubles
the actual sleep.

## Confidence: HIGH

Measured via device-side `%globaltimer` (ns-resolution). 100-iteration
average smooths noise. Results reproducible across runs.

## Deferred investigation

- Does the rounding pattern depend on clock state (boost vs locked)?
- Can nanosleep be preempted by higher-priority work?
- Is there a "nanosleep.u32 0" special-case opcode?