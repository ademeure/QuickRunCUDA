#pragma once

#include "cuda_helper.h"
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <omp.h>

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
		for (size_t idx : which) {
			check(idx, "--random");
			size_t n = dwords[idx];
			std::vector<uint32_t> buf(n);
			constexpr size_t chunk = 1024 * 1024;
			size_t nchunks = (n + chunk - 1) / chunk;
			#pragma omp parallel for schedule(static)
			for (size_t c = 0; c < nchunks; c++) {
				std::mt19937_64 rng(seed + c);
				std::uniform_int_distribution<uint32_t> dist;
				for (size_t i = c * chunk, end = std::min(i + chunk, n); i < end; ++i)
					buf[i] = dist(rng) & mask;
			}
			checkCudaErrors(cuMemcpyHtoD(d[idx], buf.data(), bytes(idx)));
		}
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

	void compare(size_t index, const std::string& refFilename, float tolerance) {
		check(index, "--reference");
		auto ref = readBinaryFile(refFilename, bytes(index), "--reference");
		readback(index);
		for (size_t i = 0; i < dwords[index]; i++) {
			if (tolerance > 0.0f) {
				float a = *reinterpret_cast<float*>(&h[i]);
				float b = *reinterpret_cast<float*>(&ref[i]);
				if (std::abs(a - b) > tolerance) {
					printf("First difference at %zu: %.8f vs %.8f (diff %.8f)\n", i, a, b, std::abs(a - b));
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
