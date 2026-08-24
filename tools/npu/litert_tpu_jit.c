// JIT-компиляция на устройстве: подаём ОБЫЧНЫЙ .tflite (не AOT), рантайм сам компилирует
// его под Tensor TPU через libLiteRtCompilerPlugin_google_tensor.so.
// Смысл: AOT-компилятор для x86 выдаётся Google по заявке (ACL), а arm64-плагин лежит
// в публичном релизе LiteRT — если JIT работает, закрытый SDK для сборки моделей не нужен.
// Аргументы: model plugin_dir dispatch_dir [runs=3]
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
  const char* model_path = argv[1];
  const char* plugin_dir = argc > 2 ? argv[2] : "/data/local/tmp";
  const char* dispatch_dir = argc > 3 ? argv[3] : "/data/local/tmp";
  int runs = argc > 4 ? atoi(argv[4]) : 3;

  const char* cache_dir = argc > 5 ? argv[5] : NULL;
  LiteRtEnvOption opts_env[3];
  opts_env[0].tag = kLiteRtEnvOptionTagCompilerPluginLibraryDir;
  opts_env[0].value.type = kLiteRtAnyTypeString;
  opts_env[0].value.str_value = plugin_dir;
  opts_env[1].tag = kLiteRtEnvOptionTagDispatchLibraryDir;
  opts_env[1].value.type = kLiteRtAnyTypeString;
  opts_env[1].value.str_value = dispatch_dir;

  int n_opt = 2;
  if (cache_dir) {  // кэш компиляции: второй запуск не платит за JIT заново
    opts_env[2].tag = kLiteRtEnvOptionTagCompilerCacheDir;
    opts_env[2].value.type = kLiteRtAnyTypeString;
    opts_env[2].value.str_value = cache_dir;
    n_opt = 3;
  }
  LiteRtEnvironment env;
  CK(LiteRtCreateEnvironment(n_opt, opts_env, &env));
  fprintf(stderr, "plugin_dir=%s dispatch_dir=%s\n", plugin_dir, dispatch_dir);

  LiteRtModel model;
  double t0 = now_ms();
  CK(LiteRtCreateModelFromFile(env, model_path, &model));
  fprintf(stderr, "модель загружена за %.0f ms: %s\n", now_ms() - t0, model_path);

  LiteRtParamIndex sg_idx;
  CK(LiteRtGetMainModelSubgraphIndex(model, &sg_idx));
  LiteRtSubgraph sg;
  CK(LiteRtGetModelSubgraph(model, sg_idx, &sg));
  LiteRtParamIndex n_in, n_out;
  CK(LiteRtGetNumSubgraphInputs(sg, &n_in));
  CK(LiteRtGetNumSubgraphOutputs(sg, &n_out));

  LiteRtOptions opts;
  CK(LiteRtCreateOptions(&opts));
  CK(LiteRtSetOptionsHardwareAccelerators(opts, kLiteRtHwAcceleratorNpu));

  // здесь и происходит JIT: рантайм зовёт плагин-компилятор
  t0 = now_ms();
  LiteRtCompiledModel cm;
  CK(LiteRtCreateCompiledModel(env, model, opts, &cm));
  fprintf(stderr, "JIT-КОМПИЛЯЦИЯ + загрузка: %.1f с\n", (now_ms() - t0) / 1000.0);

  LiteRtTensorBuffer in_buf[8], out_buf[8];
  for (LiteRtParamIndex i = 0; i < n_in; ++i) {
    LiteRtTensor t;
    CK(LiteRtGetSubgraphInput(sg, i, &t));
    LiteRtRankedTensorType tt;
    CK(LiteRtGetRankedTensorType(t, &tt));
    LiteRtTensorBufferRequirements r;
    CK(LiteRtGetCompiledModelInputBufferRequirements(cm, sg_idx, i, &r));
    CK(LiteRtCreateManagedTensorBufferFromRequirements(env, &tt, r, &in_buf[i]));
    void* host = NULL;
    size_t sz = 0;
    CK(LiteRtLockTensorBuffer(in_buf[i], &host, kLiteRtTensorBufferLockModeWrite));
    CK(LiteRtGetTensorBufferSize(in_buf[i], &sz));
    memset(host, 0, sz);
    CK(LiteRtUnlockTensorBuffer(in_buf[i]));
    fprintf(stderr, "  вход[%d] %ld элем\n", (int)i, elem_count(&tt));
  }
  for (LiteRtParamIndex i = 0; i < n_out; ++i) {
    LiteRtTensor t;
    CK(LiteRtGetSubgraphOutput(sg, i, &t));
    LiteRtRankedTensorType tt;
    CK(LiteRtGetRankedTensorType(t, &tt));
    LiteRtTensorBufferRequirements r;
    CK(LiteRtGetCompiledModelOutputBufferRequirements(cm, sg_idx, i, &r));
    CK(LiteRtCreateManagedTensorBufferFromRequirements(env, &tt, r, &out_buf[i]));
  }

  double best = 1e18;
  for (int r = 0; r < runs; ++r) {
    double s = now_ms();
    CK(LiteRtRunCompiledModel(cm, sg_idx, n_in, in_buf, n_out, out_buf));
    double dt = now_ms() - s;
    if (dt < best) best = dt;
    fprintf(stderr, "  run %d: %.1f ms\n", r, dt);
  }
  printf("JIT forward: best=%.1f ms (n=%d)\n", best, runs);
  return 0;
}
