#pragma once

#include <torch/script.h>
#include "torch/custom_class.h"
#include "c10/util/intrusive_ptr.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstddef>
#include <vector>

#define _MB (1 << 20)

// CacheDevicePtr class, to warp CUdeviceptr, used to kv-cache tensor
// record the reserved virtual address size and allocated physical memory size.
class CacheDevicePtr : public torch::CustomClassHolder {
 public:
  CUdeviceptr dptr;
  int64_t reservedPageNum;
  int64_t allocatedPageNum;
  // page size, the minimum unit of kvcache memory allocation
  int64_t pageSize = 2 * _MB;

  CacheDevicePtr();

  ~CacheDevicePtr();

  // set the page size, the page size must be a multiple of the granularity
  void setPageSize(int64_t num = 1);

  // get CUdeviceptr dptr
  CUdeviceptr get_dptr();

  // get void * type pointer
  void* get_void_ptr();
};

// CacheAllocator class, used for memory allocation of kv-cachemanager, memory
// allocation is based on page granularity,
class CacheAllocator : public torch::CustomClassHolder {
 private:
  CUmemAllocationProp prop;
  CUmemAccessDesc accessDescr;
  // memory allocation granularity supported by the gpu, 2MB for nvidia gpu
  size_t granularity = 2 * _MB;
  // page size, the minimum unit of kvcache memory allocation, default 2MB
  size_t pageSize = 2 * _MB;

 public:
  CacheAllocator();

  ~CacheAllocator() = default;

  // get the granularity of the memory allocation
  int64_t getGranularity();

  // set the page size, the page size must be a multiple of the granularity
  void setPageSize(int64_t num = 1);

  // reserve function, reserve virtual address space
  int64_t reserveCachePtr(const c10::intrusive_ptr<CacheDevicePtr>& ptr,
                          int64_t pageNum = 1);

  // alloc function, allocate physical memory, map to the reserved virtual
  // address space of dptr, and set access permission
  int64_t allocCachePtr(const c10::intrusive_ptr<CacheDevicePtr>& ptr,
                        int64_t pageNum = 1, int64_t offset = 0);

  // free function, unmap the virtual address space，release physical memory
  // handles and free virtual address space
  int64_t freeCachePtr(const c10::intrusive_ptr<CacheDevicePtr>& ptr);

  // releaseCachePtrPages function, unmap the virtual address space，release
  // physical memory handles but not free virtual address space
  int64_t releaseCachePtr(const c10::intrusive_ptr<CacheDevicePtr>& ptr,
                          int64_t pageNum = 0, int64_t offset = 0);
};

// warp CUdeviceptr to torch tensor
torch::Tensor wrap_dptr_to_tensor(CUdeviceptr d_ptr, const std::string dtype,
                                  at::ArrayRef<int64_t> shape);

torch::Tensor wrap_cache_ptr_to_tensor(
    const c10::intrusive_ptr<CacheDevicePtr>& ptr, const std::string dtype,
    at::ArrayRef<int64_t> shape);
