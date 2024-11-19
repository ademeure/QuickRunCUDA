from cuda_controller import CUDAController
import sys

def main():
    NUM_BLOCKS = 114
    THREADS_PER_BLOCK = 32

    cuda = CUDAController()

    with open("icache_sweep_output.txt", "wt") as output_file:
        loop_skips = [i for i in range(1, 1, 1)] + [i for i in range(64, 65, 1)]
        innerloops = [i for i in range(100, 101, 50)]

        # Dictionary to track which GPC each SM belongs to
        sm_to_gpc = {}
        gpcs = [] # List of lists, each inner list represents SMs in one GPC

        for innerloop in innerloops:
            for loop_skip in loop_skips:
                for arg0 in range(NUM_BLOCKS):
                    if arg0 in sm_to_gpc:
                        continue

                    for arg1 in range(arg0 + 1, NUM_BLOCKS, 1):
                        if arg1 in sm_to_gpc:
                            continue

                        header = f"constexpr int FMA_LOOP_ALWAYS = 1, FMA_LOOP_SKIP = {loop_skip}, innerloop = {innerloop};"
                        # Wrap header in quotes to make it a single argument
                        header = f"'{header}'"

                        args = [
                            "-f", "icache.cu",
                            "-b", str(NUM_BLOCKS),
                            "-t", str(THREADS_PER_BLOCK),
                            "--kernel-int-arg0", str(arg0),
                            "--kernel-int-arg1", str(arg1),
                            "-H", header
                        ]

                        if not (arg0 == 0 and arg1 == 1):
                            args.append("--reuse-cubin")
                        else:
                            args.append("--dummy")

                        response = cuda.send_command(args)
                        out_raw = response.strip().replace('\n', ',')
                        out = out_raw.split(',')

                        if any(float(num) > 200.0 for num in out):
                            print(out)
                            if arg0 in sm_to_gpc:
                                gpc_idx = sm_to_gpc[arg0]
                                gpcs[gpc_idx].append(arg1)
                                sm_to_gpc[arg1] = gpc_idx
                            else:
                                gpcs.append([arg0, arg1])
                                sm_to_gpc[arg0] = len(gpcs)-1
                                sm_to_gpc[arg1] = len(gpcs)-1

                        output_file.write(out_raw + '\n')

        # Write "gpcs" to CSV, one row per GPC
        with open("GPCs.csv", "wt") as csv_file:
            for i, gpc in enumerate(gpcs):
                csv_file.write(",".join(str(sm) for sm in gpc) + "\n")
                print(f"GPC {i}: {gpc} ({len(gpc)}x)")

if __name__ == "__main__":
    main()
