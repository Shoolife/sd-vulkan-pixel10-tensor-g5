// LiteRT-Next раннер: AOT-скомпилированная под Tensor модель на NPU (edgetpu).
// Замер forward + опционально реальные входы (in_<count>.bin) и дамп выхода (для corr).
// Сборка: NDK clang arm64, линк libLiteRt.so. Только C-API (без absl).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"

#define CK(x)                                                 \
  do {                                                        \
    LiteRtStatus _s = (x);                                    \
    if (_s != kLiteRtStatusOk) {                              \
      fprintf(stderr, "FAIL %s -> status %d\n", #x, (int)_s); \
      return 2;                                               \
    }                                                         \
  } while (0)

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static long elem_count(const LiteRtRankedTensorType* tt) {
  long c = 1;
  for (unsigned d = 0; d < tt->layout.rank; ++d) c *= tt->layout.dimensions[d];
  return c;
}

int main(int argc, char** argv) {
  const char* model_path = argc > 1 ? argv[1] : "/data/local/tmp/unet_g5.tflite";
  const char* dispatch_dir = argc > 2 ? argv[2] : "/data/local/tmp";
  int runs = argc > 3 ? atoi(argv[3]) : 10;
  const char* in_dir = argc > 4 ? argv[4] : NULL;   // каталог с in_<count>.bin
  const char* out_file = argc > 5 ? argv[5] : NULL;  // куда писать выход
  int warmup = 3;

  LiteRtEnvOption opt;
  opt.tag = kLiteRtEnvOptionTagDispatchLibraryDir;
  opt.value.type = kLiteRtAnyTypeString;
  opt.value.str_value = dispatch_dir;
  LiteRtEnvironment env;
  CK(LiteRtCreateEnvironment(1, &opt, &env));

  LiteRtModel model;
  CK(LiteRtCreateModelFromFile(env, model_path, &model));
  LiteRtParamIndex sg_idx;
  CK(LiteRtGetMainModelSubgraphIndex(model, &sg_idx));
  LiteRtSubgraph sg;
  CK(LiteRtGetModelSubgraph(model, sg_idx, &sg));
  LiteRtParamIndex n_in, n_out;
  CK(LiteRtGetNumSubgraphInputs(sg, &n_in));
  CK(LiteRtGetNumSubgraphOutputs(sg, &n_out));
  fprintf(stderr, "model: %s | inputs=%d outputs=%d\n", model_path, (int)n_in,
          (int)n_out);

  LiteRtOptions opts;
  CK(LiteRtCreateOptions(&opts));
  CK(LiteRtSetOptionsHardwareAccelerators(opts, kLiteRtHwAcceleratorNpu));
  double t0 = now_ms();
  LiteRtCompiledModel cm;
  CK(LiteRtCreateCompiledModel(env, model, opts, &cm));
  fprintf(stderr, "CompiledModel(NPU) готов за %.0f ms\n", now_ms() - t0);

  LiteRtTensorBuffer in_buf[32], out_buf[32];
  for (LiteRtParamIndex i = 0; i < n_in; ++i) {
    LiteRtTensor t;
    CK(LiteRtGetSubgraphInput(sg, i, &t));
    LiteRtRankedTensorType tt;
    CK(LiteRtGetRankedTensorType(t, &tt));
    long cnt = elem_count(&tt);
    LiteRtTensorBufferRequirements req;
    CK(LiteRtGetCompiledModelInputBufferRequirements(cm, sg_idx, i, &req));
    CK(LiteRtCreateManagedTensorBufferFromRequirements(env, &tt, req, &in_buf[i]));
    void* host = NULL;
    CK(LiteRtLockTensorBuffer(in_buf[i], &host, kLiteRtTensorBufferLockModeWrite));
    size_t sz = 0;
    CK(LiteRtGetTensorBufferSize(in_buf[i], &sz));
    memset(host, 0, sz);
    if (in_dir) {
      // int8-модель: буфер cnt байт, данные уже квантованы -> in_<cnt>_i8.bin
      char path[512];
      int is_i8 = (sz == (size_t)cnt);
      snprintf(path, sizeof path, "%s/in_%ld%s.bin", in_dir, cnt, is_i8 ? "_i8" : "");
      FILE* f = fopen(path, "rb");
      if (f) {
        size_t rd = fread(host, 1, sz, f);  // ровно размер буфера, любой dtype
        fclose(f);
        fprintf(stderr, "  in[%d] cnt=%ld <- %s (%zu/%zu байт)\n", (int)i, cnt, path, rd, sz);
      } else {
        fprintf(stderr, "  in[%d] cnt=%ld: НЕТ %s -> нули\n", (int)i, cnt, path);
      }
    }
    CK(LiteRtUnlockTensorBuffer(in_buf[i]));
  }
  long out_cnt[32];
  for (LiteRtParamIndex i = 0; i < n_out; ++i) {
    LiteRtTensor t;
    CK(LiteRtGetSubgraphOutput(sg, i, &t));
    LiteRtRankedTensorType tt;
    CK(LiteRtGetRankedTensorType(t, &tt));
    out_cnt[i] = elem_count(&tt);
    LiteRtTensorBufferRequirements req;
    CK(LiteRtGetCompiledModelOutputBufferRequirements(cm, sg_idx, i, &req));
    CK(LiteRtCreateManagedTensorBufferFromRequirements(env, &tt, req, &out_buf[i]));
  }

  for (int w = 0; w < warmup; ++w)
    CK(LiteRtRunCompiledModel(cm, sg_idx, n_in, in_buf, n_out, out_buf));

  double best = 1e18, sum = 0;
  for (int r = 0; r < runs; ++r) {
    double s = now_ms();
    CK(LiteRtRunCompiledModel(cm, sg_idx, n_in, in_buf, n_out, out_buf));
    double dt = now_ms() - s;
    if (dt < best) best = dt;
    sum += dt;
    fprintf(stderr, "  run %d: %.1f ms\n", r, dt);
  }
  printf("TPU forward: best=%.1f ms  avg=%.1f ms  (n=%d)\n", best, sum / runs, runs);

  if (out_file) {
    void* host = NULL;
    CK(LiteRtLockTensorBuffer(out_buf[0], &host, kLiteRtTensorBufferLockModeRead));
    size_t osz = 0;
    CK(LiteRtGetTensorBufferSize(out_buf[0], &osz));
    FILE* f = fopen(out_file, "wb");
    if (f) {
      fwrite(host, 1, osz, f);  // сырые байты: float32 или int8, деквант на ПК
      fclose(f);
      fprintf(stderr, "output %ld элем / %zu байт -> %s\n", out_cnt[0], osz, out_file);
    }
    CK(LiteRtUnlockTensorBuffer(out_buf[0]));
  }
  return 0;
}
