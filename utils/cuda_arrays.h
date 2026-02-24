#pragma once

#include "cuda_compile.h"
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iomanip>

struct GPUArrays {
	CUdeviceptr base = 0;
	size_t totalBytes = 0;
	std::vector<CUdeviceptr> d;
	std::vector<size_t> dwords;
	std::vector<uint32_t> h;

	static constexpr size_t ALIGN = 2 * 1024 * 1024;
	static size_t alignUp(size_t n) { return (n + ALIGN - 1) & ~(ALIGN - 1); }
	size_t bytes(size_t i) const { return dwords[i] * sizeof(uint32_t); }
	size_t count() const { return d.size(); }

	void check(size_t i, const char* ctx) const {
		if (i >= d.size()) {
			fprintf(stderr, "%s: array index %zu out of range (%zu arrays)\n", ctx, i, d.size());
			exit(EXIT_FAILURE);
		}
	}

	std::vector<uint32_t> readBinaryFile(const std::string& filename, size_t expectedBytes, const char* ctx) {
		std::ifstream f(filename, std::ios::binary | std::ios::ate);
		if (!f) { fprintf(stderr, "Failed to open %s\n", filename.c_str()); exit(EXIT_FAILURE); }
		size_t sz = f.tellg();
		if (sz != expectedBytes) {
			fprintf(stderr, "%s: file size mismatch (got %zu, expected %zu)\n", ctx, sz, expectedBytes);
			exit(EXIT_FAILURE);
		}
		std::vector<uint32_t> data(expectedBytes / sizeof(uint32_t));
		f.seekg(0);
		f.read(reinterpret_cast<char*>(data.data()), expectedBytes);
		return data;
	}

	void allocate(const std::vector<size_t>& sizes) {
		dwords = sizes;
		d.resize(sizes.size());
		totalBytes = 0;
		for (auto s : sizes) totalBytes += alignUp(s * sizeof(uint32_t));
		checkCudaErrors(cuMemAlloc(&base, totalBytes));
		checkCudaErrors(cuMemsetD8(base, 0, totalBytes));
		size_t offset = 0;
		for (size_t i = 0; i < sizes.size(); i++) {
			d[i] = base + offset;
			offset += alignUp(sizes[i] * sizeof(uint32_t));
		}
	}

	void free() {
		if (base) { checkCudaErrors(cuMemFree(base)); base = 0; }
		d.clear();
	}

	void initRandom(const std::vector<size_t>& which, uint32_t seed, uint32_t mask) {
		static const char* src = R"(
			extern "C" __global__ void rng_fill(unsigned* out, unsigned long long n,
			                                    unsigned seed, unsigned mask) {
				unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
				if (i >= n) return;
				unsigned long long x = seed + i;
				x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
				x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
				out[i] = (unsigned)(x ^ (x >> 31)) & mask;
			}
		)";
		static CUfunction func = nullptr;
		if (!func) {
			static CUmodule mod = compileSource(src, "rng.cu", false);
			checkCudaErrors(cuModuleGetFunction(&func, mod, "rng_fill"));
		}
		for (size_t idx : which) {
			check(idx, "--random");
			unsigned long long n = dwords[idx];
			uint32_t s = seed + (uint32_t)(idx * n);
			void* args[] = { &d[idx], &n, &s, &mask };
			int threads = 256;
			int blocks = (int)((n + threads - 1) / threads);
			checkCudaErrors(cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
		}
		checkCudaErrors(cuCtxSynchronize());
	}

	void load(size_t index, const std::string& filename) {
		check(index, "--load");
		h = readBinaryFile(filename, bytes(index), "--load");
		checkCudaErrors(cuMemcpyHtoD(d[index], h.data(), bytes(index)));
	}

	void readback(size_t index) {
		check(index, "readback");
		h.resize(dwords[index]);
		checkCudaErrors(cuMemcpyDtoH(h.data(), d[index], bytes(index)));
	}

	void dump(size_t index, const std::string& filename, const std::string& format) {
		check(index, "--dump");
		readback(index);
		if (format == "raw") {
			std::ofstream f(filename, std::ios::binary);
			f.write(reinterpret_cast<char*>(h.data()), bytes(index));
		} else {
			std::ofstream f(filename);
			for (size_t i = 0; i < dwords[index]; i++) {
				if (format == "int_csv") f << h[i];
				else if (format == "float_csv")
					f << std::fixed << std::setprecision(2) << *reinterpret_cast<float*>(&h[i]);
				if (i < dwords[index] - 1) f << ",";
			}
		}
	}

	void compare(size_t index, const std::string& refFilename, float atol, float rtol) {
		check(index, "--reference");
		auto ref = readBinaryFile(refFilename, bytes(index), "--reference");
		readback(index);
		for (size_t i = 0; i < dwords[index]; i++) {
			if (atol > 0.0f || rtol > 0.0f) {
				float a = *reinterpret_cast<float*>(&h[i]);
				float b = *reinterpret_cast<float*>(&ref[i]);
				float diff = std::abs(a - b);
				if (diff > atol && diff > rtol * std::abs(b)) {
					printf("First difference at %zu: %.8f vs %.8f (abs=%.8f, rel=%.8f)\n",
					       i, a, b, diff, diff / std::max(std::abs(b), 1e-30f));
					break;
				}
			} else if (h[i] != ref[i]) {
				printf("First difference at %zu: %u vs %u (hex: %08x vs %08x, fp32: %.4f vs %.4f)\n",
				       i, h[i], ref[i], h[i], ref[i],
				       *reinterpret_cast<float*>(&h[i]), *reinterpret_cast<float*>(&ref[i]));
				break;
			}
		}
	}
};
