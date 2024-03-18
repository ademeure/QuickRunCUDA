# Run the following command: ./QuickRunCUDA -b 32 -n 132 -H "constexpr int FMA_LOOP_ALWAYS = 1, FMA_LOOP_SKIP = 27, innerloop = 14;
import os
import sys
import re
import subprocess

NUM_BLOCKS = 132
THREADS_PER_BLOCK = 32

# Set the path to the QuickRunCUDA executable
output_filename = "icache_sweep_output.txt"
output_file = open(output_filename, "wt")

'''
innerloops = [i for i in range(0, 17, 1)] + [i for i in range(18, 81, 2)] + [i for i in range(88, 321, 8)]
for innerloop in innerloops:
    header = "constexpr int FMA_LOOP_ALWAYS = 1, FMA_LOOP_SKIP = 27, innerloop = " + str(innerloop) + ";"
    out_process = subprocess.run(["./QuickRunCUDA",
                    "-f", "icache.cu",
                    "-b", str(NUM_BLOCKS), "-t", str(THREADS_PER_BLOCK),
                    "-0", "0", "-1", "0", "-2", "0",
                    "-H", header],
                    capture_output=True)
    print(out_process.stdout.decode().strip())
    output_file.write(out_process.stdout.decode())
'''

loop_skips = [i for i in range(1, 1, 1)] + [i for i in range(1000, 2000, 1)]
innerloops = [i for i in range(16, 17, 8)]
for innerloop in innerloops:
    for loop_skip in loop_skips:
        header = "constexpr int FMA_LOOP_ALWAYS = 1, FMA_LOOP_SKIP = " + str(loop_skip) + ", innerloop = " + str(innerloop) + ";"
        out_process = subprocess.run(["./QuickRunCUDA",
                        "-f", "icache.cu",
                        "-b", str(NUM_BLOCKS), "-t", str(THREADS_PER_BLOCK),
                        "-0", "0", "-1", "0", "-2", "0",
                        "-H", header],
                        capture_output=True)
        print(innerloop, loop_skip)
        print(out_process.stdout.decode().strip())
        output_file.write(out_process.stdout.decode())

output_file.close()
