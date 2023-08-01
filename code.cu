/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
*/

/*

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <limits.h>
/* DPCT_ORIG #include <cuda.h>*/
#include <assert.h>
#include <math.h>
#include <unistd.h>

#include "gridfuncs.h"
#include "utils.h"

/* DPCT_ORIG #define VERIF(x, ou) if ((x) != cudaSuccess)  { CheckErr((ou)); }*/
#define VERIF(x, ou) if ((x) != 0) { CheckErr((ou)); }

void
/* DPCT_ORIG SetBlockDims(long nelmts, long NTHREADS, dim3 & block, dim3 & grid)
   {*/
SetBlockDims(long nelmts, long NTHREADS, sycl::range<3> &block,
             sycl::range<3> &grid) {

  // fill a 2D grid if necessary
  long totalblocks = (nelmts + (NTHREADS) - 1) / (NTHREADS);
  long blocksx = totalblocks;
  long blocksy = 1;
  while (blocksx > 65534) {
    blocksx /= 2;
    blocksy *= 2;
  }
  if ((blocksx * blocksy * NTHREADS) < nelmts)
    blocksx++;
/* DPCT_ORIG   grid.x = blocksx;*/
  grid[2] = blocksx;
/* DPCT_ORIG   grid.y = blocksy;*/
  grid[1] = blocksy;
/* DPCT_ORIG   block.x = NTHREADS;*/
  block[2] = NTHREADS;
/* DPCT_ORIG   block.y = 1;*/
  block[1] = 1;

//     if (verbosity > 1) {
//         fprintf(stderr, "N=%d: bx=%d by=%d gx=%d gY=%d\n", nelmts, block->x,
//                 block->y, grid->x, grid->y);

//     }
}

void
initDevice(long myCard) {
/* DPCT_ORIG   cudaSetDevice(myCard);*/
  /*
  DPCT1093:42: The "myCard" device may be not the one intended for use. Adjust
  the selected device if needed.
  */
  dpct::select_device(myCard);
}

void
releaseDevice(long myCard) {
/* DPCT_ORIG   cudaThreadExit();*/
  dpct::get_current_device().reset();
  CheckErr("releaseDevice");
}

void
CheckErr(const char *where) {
/* DPCT_ORIG   cudaError_t cerror;*/
  dpct::err0 cerror;
/* DPCT_ORIG   cerror = cudaGetLastError();*/
  /*
  DPCT1010:45: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  cerror = 0;
/* DPCT_ORIG   if (cerror != cudaSuccess) {*/
  /*
  DPCT1000:44: Error handling if-stmt was detected but could not be rewritten.
  */
  if (cerror != 0) {
    char host[256];
    char message[1024];
    /*
    DPCT1001:43: The statement could not be removed.
    */
    gethostname(host, 256);
/* DPCT_ORIG     sprintf(message, "CudaError: %s (%s) on %s\n",
 * cudaGetErrorString(cerror), where, host);*/
    /*
    DPCT1009:46: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    sprintf(
        message, "CudaError: %s (%s) on %s\n",
        "cudaGetErrorString is not supported" /*cudaGetErrorString(cerror)*/,
        where, host);
    fputs(message, stderr);
    exit(1);
  }
}

long getDeviceCapability(int *nDevice, long *maxMemOnDevice,
                         long *maxThreads) try {
  int deviceCount;
  long dev;
  long memorySeen = LONG_MAX;
  long maxth = INT_MAX;
/* DPCT_ORIG   cudaError_t status;*/
  dpct::err0 status;
/* DPCT_ORIG   status = cudaGetDeviceCount(&deviceCount);*/
  status =
      DPCT_CHECK_ERROR(deviceCount = dpct::dev_mgr::instance().device_count());
/* DPCT_ORIG   if (status != cudaSuccess) {*/
  /*
  DPCT1000:48: Error handling if-stmt was detected but could not be rewritten.
  */
  if (status != 0) {
    /*
    DPCT1001:47: The statement could not be removed.
    */
    CheckErr("cudaGetDeviceCount");
    return 1;
  }
  if (deviceCount == 0) {
    printf("There is no device supporting CUDA\n");
    return 1;
  }
  for (dev = 0; dev < deviceCount; ++dev) {
/* DPCT_ORIG     cudaDeviceProp deviceProp;*/
    dpct::device_info deviceProp;
/* DPCT_ORIG     status = cudaGetDeviceProperties(&deviceProp, dev);*/
    status = DPCT_CHECK_ERROR(
        dpct::dev_mgr::instance().get_device(dev).get_device_info(deviceProp));
/* DPCT_ORIG     if (status != cudaSuccess) {*/
    /*
    DPCT1000:50: Error handling if-stmt was detected but could not be rewritten.
    */
    if (status != 0) {
      /*
      DPCT1001:49: The statement could not be removed.
      */
      CheckErr("cudaGetDeviceProperties");
      return 1;
    }
    if (dev == 0) {
/* DPCT_ORIG       if (deviceProp.major == 9999 && deviceProp.minor == 9999) {*/
      /*
      DPCT1005:51: The SYCL device version is different from CUDA Compute
      Compatibility. You may need to rewrite this code.
      */
      if (deviceProp.get_major_version() == 9999 &&
          deviceProp.get_minor_version() == 9999) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return 1;
      }
    }
/* DPCT_ORIG     if (deviceProp.totalGlobalMem < memorySeen)*/
    if (deviceProp.get_global_mem_size() < memorySeen)
/* DPCT_ORIG       memorySeen = deviceProp.totalGlobalMem;*/
      memorySeen = deviceProp.get_global_mem_size();
/* DPCT_ORIG     if (maxth > deviceProp.maxThreadsPerBlock)*/
    if (maxth > deviceProp.get_max_work_group_size())
/* DPCT_ORIG       maxth = deviceProp.maxThreadsPerBlock;*/
      maxth = deviceProp.get_max_work_group_size();
  }
  *nDevice = deviceCount;
  *maxMemOnDevice = memorySeen;
  *maxThreads = maxth;
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/* DPCT_ORIG __global__ void
LoopKredMaxDble(double *src, double *res, const long nb) {*/
void LoopKredMaxDble(double *src, double *res, const long nb,
                     const sycl::nd_item<3> &item_ct1, double *sdata) {
/* DPCT_ORIG   __shared__ double sdata[512];*/

/* DPCT_ORIG   int blockSize = blockDim.x * blockDim.y * blockDim.z;*/
  int blockSize = item_ct1.get_local_range(2) * item_ct1.get_local_range(1) *
                  item_ct1.get_local_range(0);
/* DPCT_ORIG   int tidL = threadIdx.x;*/
  int tidL = item_ct1.get_local_id(2);
/* DPCT_ORIG   int myblock = blockIdx.x + blockIdx.y * blockDim.x + blockIdx.z *
 * blockDim.x * blockDim.y;*/
  int myblock = item_ct1.get_group(2) +
                item_ct1.get_group(1) * item_ct1.get_local_range(2) +
                item_ct1.get_group(0) * item_ct1.get_local_range(2) *
                    item_ct1.get_local_range(1);
  int i = idx1d(item_ct1);

  // protection pour les cas ou on n'est pas multiple du block
  // par defaut le max est le premier element
/* DPCT_ORIG   sdata[threadIdx.x] = src[0];*/
  sdata[item_ct1.get_local_id(2)] = src[0];
  if (i < nb) {
/* DPCT_ORIG     sdata[threadIdx.x] = src[i];*/
    sdata[item_ct1.get_local_id(2)] = src[i];
  }
/* DPCT_ORIG   __syncthreads();*/
  /*
  DPCT1065:27: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // do the reduction in parallel
  if (tidL < 32) {
    if (blockSize >= 64) {
      sdata[tidL] = MAX(sdata[tidL], sdata[tidL + 32]);
/* DPCT_ORIG       __syncthreads();*/
      /*
      DPCT1065:28: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }
    if (blockSize >= 32) {
      sdata[tidL] = MAX(sdata[tidL], sdata[tidL + 16]);
/* DPCT_ORIG       __syncthreads();*/
      /*
      DPCT1065:29: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }
    if (blockSize >= 16) {
      sdata[tidL] = MAX(sdata[tidL], sdata[tidL + 8]);
/* DPCT_ORIG       __syncthreads();*/
      /*
      DPCT1065:30: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }
    if (blockSize >= 8) {
      sdata[tidL] = MAX(sdata[tidL], sdata[tidL + 4]);
/* DPCT_ORIG       __syncthreads();*/
      /*
      DPCT1065:31: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }
    if (blockSize >= 4) {
      sdata[tidL] = MAX(sdata[tidL], sdata[tidL + 2]);
/* DPCT_ORIG       __syncthreads();*/
      /*
      DPCT1065:32: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }
    if (blockSize >= 2) {
      sdata[tidL] = MAX(sdata[tidL], sdata[tidL + 1]);
/* DPCT_ORIG       __syncthreads();*/
      /*
      DPCT1065:33: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }
  }
  // get the partial result from this block
  if (tidL == 0) {
    res[myblock] = sdata[0];
    // printf("%d %lf\n", blockSize, sdata[0]);
  }
}

double reduceMax(double *array, long nb) try {
  int bs = 32;
/* DPCT_ORIG   dim3 grid, block;*/
  sycl::range<3> grid(1, 1, 1), block(1, 1, 1);
  int nbb = nb / bs;
  double resultat = 0;
/* DPCT_ORIG   cudaError_t status;*/
  dpct::err0 status;
  double *temp1, *temp2;

  nbb = (nb + bs - 1) / bs;

/* DPCT_ORIG   status = cudaMalloc((void **) &temp1, nbb * sizeof(double));*/
  status = DPCT_CHECK_ERROR(
      temp1 = sycl::malloc_device<double>(nbb, dpct::get_default_queue()));
  VERIF(status, "cudaMalloc temp1");
/* DPCT_ORIG   status = cudaMalloc((void **) &temp2, nbb * sizeof(double));*/
  status = DPCT_CHECK_ERROR(
      temp2 = sycl::malloc_device<double>(nbb, dpct::get_default_queue()));
  VERIF(status, "cudaMalloc temp2");
  double *tmp;

  // on traite d'abord le tableau d'origine
  SetBlockDims(nb, bs, block, grid);
/* DPCT_ORIG   LoopKredMaxDble <<< grid, block >>> (array, temp1, nb);*/
  /*
  DPCT1049:34: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(dpct::get_default_queue().get_device(),
                                 {sycl::aspect::fp64});
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::local_accessor<double, 1> sdata_acc_ct1(sycl::range<1>(512), cgh);

      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                         LoopKredMaxDble(array, temp1, nb, item_ct1,
                                         sdata_acc_ct1.get_pointer());
                       });
    });
  }
  CheckErr("KredMaxDble");
/* DPCT_ORIG   cudaThreadSynchronize();*/
  dpct::get_current_device().queues_wait_and_throw();
  CheckErr("reducMax");

  // ici on a nbb maxima locaux

  while (nbb > 1) {
    SetBlockDims(nbb, bs, block, grid);
/* DPCT_ORIG     LoopKredMaxDble <<< grid, block >>> (temp1, temp2, nbb);*/
    /*
    DPCT1049:35: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(dpct::get_default_queue().get_device(),
                                   {sycl::aspect::fp64});
      dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 1> sdata_acc_ct1(sycl::range<1>(512), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                           LoopKredMaxDble(temp1, temp2, nbb, item_ct1,
                                           sdata_acc_ct1.get_pointer());
                         });
      });
    }
    CheckErr("KredMaxDble 2");
/* DPCT_ORIG     cudaThreadSynchronize();*/
    dpct::get_current_device().queues_wait_and_throw();
    CheckErr("reducMax 2");
    // on permute les tableaux pour une eventuelle iteration suivante,
    tmp = temp1;
    temp1 = temp2;
    temp2 = tmp;
    // on rediminue la taille du probleme
    nbb = (nbb + bs - 1) / bs;
    // fprintf(stderr, "n=%d b=%d\n", nbb, bs);
  }

/* DPCT_ORIG   cudaMemcpy(&resultat, temp1, sizeof(double),
 * cudaMemcpyDeviceToHost);*/
  dpct::get_default_queue().memcpy(&resultat, temp1, sizeof(double)).wait();
  // printf("R=%lf\n", resultat);
/* DPCT_ORIG   cudaFree(temp1);*/
  sycl::free(temp1, dpct::get_default_queue());
/* DPCT_ORIG   cudaFree(temp2);*/
  sycl::free(temp2, dpct::get_default_queue());
  return resultat;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//EOF