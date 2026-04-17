// CUDA driver/runtime version info and PCI info
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvml.h>
#include <cstdio>

int main() {
    int rt_ver, drv_ver;
    cudaRuntimeGetVersion(&rt_ver);
    cudaDriverGetVersion(&drv_ver);

    printf("# B300 system version info\n\n");
    printf("CUDA Runtime version: %d (%d.%d)\n", rt_ver, rt_ver/1000, (rt_ver%1000)/10);
    printf("CUDA Driver version:  %d (%d.%d)\n", drv_ver, drv_ver/1000, (drv_ver%1000)/10);

    // NVML driver version
    nvmlInit_v2();
    char drv[80];
    nvmlSystemGetDriverVersion(drv, sizeof(drv));
    printf("NVIDIA driver version: %s\n\n", drv);

    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex_v2(0, &dev);

    // PCIe info
    nvmlPciInfo_t pci;
    nvmlDeviceGetPciInfo_v3(dev, &pci);
    printf("PCIe bus ID: %s\n", pci.busId);
    printf("PCIe domain: %u\n", pci.domain);
    printf("PCIe device: %u\n", pci.device);
    printf("PCIe pciDeviceId: 0x%x\n", pci.pciDeviceId);

    unsigned int link_gen, link_width;
    nvmlDeviceGetCurrPcieLinkGeneration(dev, &link_gen);
    nvmlDeviceGetCurrPcieLinkWidth(dev, &link_width);
    printf("PCIe gen: %u, width: x%u\n", link_gen, link_width);

    unsigned int max_gen, max_width;
    nvmlDeviceGetMaxPcieLinkGeneration(dev, &max_gen);
    nvmlDeviceGetMaxPcieLinkWidth(dev, &max_width);
    printf("PCIe max gen: %u, max width: x%u\n", max_gen, max_width);

    // Power limits
    unsigned int min_lim, max_lim;
    nvmlDeviceGetPowerManagementLimitConstraints(dev, &min_lim, &max_lim);
    printf("\nPower limits: min=%u W, max=%u W\n", min_lim/1000, max_lim/1000);

    unsigned int default_lim, current_lim;
    nvmlDeviceGetPowerManagementDefaultLimit(dev, &default_lim);
    nvmlDeviceGetPowerManagementLimit(dev, &current_lim);
    printf("Power: default=%u W, current limit=%u W\n", default_lim/1000, current_lim/1000);

    // Memory info
    nvmlMemory_v2_t mem = {.version = nvmlMemory_v2};
    if (nvmlDeviceGetMemoryInfo_v2(dev, &mem) == NVML_SUCCESS) {
        printf("\nMemory: %.1f GB total, %.1f GB free, %.1f GB used, %.1f GB reserved\n",
               mem.total/1e9, mem.free/1e9, mem.used/1e9, mem.reserved/1e9);
    }

    // Compute mode
    nvmlComputeMode_t mode;
    nvmlDeviceGetComputeMode(dev, &mode);
    const char *modes[] = {"Default", "Exclusive", "Prohibited", "Exclusive_Process"};
    printf("Compute mode: %s\n", mode <= 3 ? modes[mode] : "?");

    // Persistence mode
    nvmlEnableState_t persist;
    nvmlDeviceGetPersistenceMode(dev, &persist);
    printf("Persistence mode: %s\n", persist == NVML_FEATURE_ENABLED ? "enabled" : "disabled");

    // ECC
    nvmlEnableState_t ecc_curr, ecc_pending;
    nvmlDeviceGetEccMode(dev, &ecc_curr, &ecc_pending);
    printf("ECC: current=%s, pending=%s\n",
           ecc_curr == NVML_FEATURE_ENABLED ? "on" : "off",
           ecc_pending == NVML_FEATURE_ENABLED ? "on" : "off");

    nvmlShutdown();
    return 0;
}
