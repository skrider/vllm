#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillRaggedWrapper(nv_bfloat16, 8, 64, true, false, QKVLayout::kNHD, PosEncodingMode::kALiBi)
