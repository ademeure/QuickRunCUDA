# QuickRunCUDA

This is the microbenchmarking framework I used to build the project that won the SemiAnalysis GPU Hackathon ("Optimizing NVIDIA Blackwell’s Split L2"): https://semianalysis.com/2025-hackathon-eol/

The finished & polished project code is available here: https://github.com/ademeure/QuickRunCUDA/blob/main/tests/side_aware.cu

Example command to run the L2 Side Aware reduction that calculates the FP32 absmax of an input array (on H100/GH200/GB200):
> make
>
> ./QuickRunCUDA -i -p -t 1024 -A 1000000000 -0 1000000000 -T 100 -P 4.0 -U GB/s tests/side_aware.cu

You can uncomment "FORCE_RANDOM_SIDE" to prevent the optimization (but keeping some of the overhead). This shows that performance doesn't significantly improve, but it reduces power consumption by up to ~9% on GH200 with random data ('-r' flag)!

It is possible to extend this to any elementwise operation or memcpy, but it requires very complicated manual memory management to make it work on both the input and output sides simultaneously. So it can't really be done as part of this kind of microbenchmarking framework. It *might* be possible to do it in PyTorch using a custom allocator and mempool but I'm not 100% sure at this point.

Let me know if you have any questions about the L2 Side Aware project or the QuickRunCUDA framework in general!
