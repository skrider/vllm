#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillPagedWrapper(nv_half, 1, 128, true, false, QKVLayout::kNHD, PosEncodingMode::kALiBi)
