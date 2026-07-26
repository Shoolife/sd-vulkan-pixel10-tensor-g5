// TPU-демон: грузит AOT-модель под Tensor ОДИН раз, обслуживает forward'ы по TCP (127.0.0.1).
// Работает под uid shell (adb) → EdgeTPU пускает. TCP-loopback обходит файловые права sdcardfs.
// Протокол: клиент шлёт lat(16384f)+ts(1f)+ctx(59136f) LE-float, демон отвечает noise(16384f).
// Аргументы: model dispatch_dir [port=8763]
#include <arpa/inet.h>
#include <netinet/in.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"

#define CK(x)                                          \
  do {                                                 \
    LiteRtStatus _s = (x);                             \
    if (_s != kLiteRtStatusOk) {                       \
      fprintf(stderr, "FAIL %s -> %d\n", #x, (int)_s); \
      return 2;                                        \
    }                                                  \
  } while (0)

static long elem_count(const LiteRtRankedTensorType* tt) {
  long c = 1;
  for (unsigned d = 0; d < tt->layout.rank; ++d) c *= tt->layout.dimensions[d];
  return c;
}
static int recv_all(int fd, void* buf, size_t n) {
  char* p = (char*)buf;
  while (n) {
    ssize_t k = recv(fd, p, n, 0);
    if (k <= 0) return -1;
    p += k; n -= (size_t)k;
  }
  return 0;
}
static int send_all(int fd, const void* buf, size_t n) {
  const char* p = (const char*)buf;
  while (n) {
    ssize_t k = send(fd, p, n, 0);
    if (k <= 0) return -1;
    p += k; n -= (size_t)k;
  }
  return 0;
}

int main(int argc, char** argv) {
  const char* model_path = argv[1];
  const char* dispatch_dir = argv[2];
  int port = argc > 3 ? atoi(argv[3]) : 8763;

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
  LiteRtOptions opts;
  CK(LiteRtCreateOptions(&opts));
  CK(LiteRtSetOptionsHardwareAccelerators(opts, kLiteRtHwAcceleratorNpu));
  LiteRtCompiledModel cm;
  CK(LiteRtCreateCompiledModel(env, model, opts, &cm));

  LiteRtTensorBuffer in_buf[8], out_buf[8];
  long in_cnt[8], out_cnt[8];
  // int8-модели (mixMS и др.): демон сам квантует вход и деквантует выход,
  // чтобы протокол с приложением остался float32.
  int in_i8[8], out_i8[8];
  float in_s[8], out_s[8];
  int in_z[8], out_z[8];
  for (LiteRtParamIndex i = 0; i < n_in; ++i) {
    LiteRtTensor t; CK(LiteRtGetSubgraphInput(sg, i, &t));
    LiteRtRankedTensorType tt; CK(LiteRtGetRankedTensorType(t, &tt));
    in_cnt[i] = elem_count(&tt);
    in_i8[i] = (tt.element_type == kLiteRtElementTypeInt8);
    in_s[i] = 1.0f; in_z[i] = 0;
    if (in_i8[i]) {
      LiteRtQuantizationPerTensor q;
      if (LiteRtGetPerTensorQuantization(t, &q) == kLiteRtStatusOk) {
        in_s[i] = (float)q.scale; in_z[i] = (int)q.zero_point;
      }
      fprintf(stderr, "  in[%d] int8 scale=%g zero=%d\n", (int)i, in_s[i], in_z[i]);
    }
    LiteRtTensorBufferRequirements r;
    CK(LiteRtGetCompiledModelInputBufferRequirements(cm, sg_idx, i, &r));
    CK(LiteRtCreateManagedTensorBufferFromRequirements(env, &tt, r, &in_buf[i]));
  }
  for (LiteRtParamIndex i = 0; i < n_out; ++i) {
    LiteRtTensor t; CK(LiteRtGetSubgraphOutput(sg, i, &t));
    LiteRtRankedTensorType tt; CK(LiteRtGetRankedTensorType(t, &tt));
    out_cnt[i] = elem_count(&tt);
    out_i8[i] = (tt.element_type == kLiteRtElementTypeInt8);
    out_s[i] = 1.0f; out_z[i] = 0;
    if (out_i8[i]) {
      LiteRtQuantizationPerTensor q;
      if (LiteRtGetPerTensorQuantization(t, &q) == kLiteRtStatusOk) {
        out_s[i] = (float)q.scale; out_z[i] = (int)q.zero_point;
      }
      fprintf(stderr, "  out[%d] int8 scale=%g zero=%d\n", (int)i, out_s[i], out_z[i]);
    }
    LiteRtTensorBufferRequirements r;
    CK(LiteRtGetCompiledModelOutputBufferRequirements(cm, sg_idx, i, &r));
    CK(LiteRtCreateManagedTensorBufferFromRequirements(env, &tt, r, &out_buf[i]));
  }

  // Размеры входов из модели: ts=1, lat=меньший из остальных, ctx=больший (работает для B=1 и B=2)
  long lat_cnt = 0, ctx_cnt = 0;
  for (LiteRtParamIndex i = 0; i < n_in; ++i) {
    if (in_cnt[i] == 1) continue;
    if (lat_cnt == 0) lat_cnt = in_cnt[i];
    else if (in_cnt[i] < lat_cnt) { ctx_cnt = lat_cnt; lat_cnt = in_cnt[i]; }
    else ctx_cnt = in_cnt[i];
  }
  fprintf(stderr, "lat_cnt=%ld ctx_cnt=%ld out_cnt=%ld\n", lat_cnt, ctx_cnt, out_cnt[0]);
  float* lat = malloc((size_t)lat_cnt * 4);
  float ts[1];
  float* ctx = malloc((size_t)ctx_cnt * 4);
  float* outv = malloc((size_t)out_cnt[0] * 4);

  int srv = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in a; memset(&a, 0, sizeof a);
  a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(port);
  if (bind(srv, (struct sockaddr*)&a, sizeof a) < 0) { perror("bind"); return 3; }
  listen(srv, 4);
  fprintf(stderr, "DAEMON READY on 127.0.0.1:%d (inputs=%d)\n", port, (int)n_in);
  fflush(stderr);

  for (;;) {
    int c = accept(srv, NULL, NULL);
    if (c < 0) continue;
    if (recv_all(c, lat, (size_t)lat_cnt * 4) || recv_all(c, ts, 4) ||
        recv_all(c, ctx, (size_t)ctx_cnt * 4)) {
      close(c); continue;
    }
    for (LiteRtParamIndex i = 0; i < n_in; ++i) {
      const float* src = in_cnt[i] == 1 ? ts : in_cnt[i] == lat_cnt ? lat : ctx;
      void* host = NULL; size_t sz = 0;
      LiteRtLockTensorBuffer(in_buf[i], &host, kLiteRtTensorBufferLockModeWrite);
      LiteRtGetTensorBufferSize(in_buf[i], &sz);
      memset(host, 0, sz);
      if (in_i8[i]) {
        signed char* d = (signed char*)host;
        for (long k = 0; k < in_cnt[i]; ++k) {
          long q = lrintf(src[k] / in_s[i]) + in_z[i];
          d[k] = (signed char)(q < -128 ? -128 : q > 127 ? 127 : q);
        }
      } else {
        memcpy(host, src, (size_t)in_cnt[i] * 4);
      }
      LiteRtUnlockTensorBuffer(in_buf[i]);
    }
    if (LiteRtRunCompiledModel(cm, sg_idx, n_in, in_buf, n_out, out_buf) != kLiteRtStatusOk)
      fprintf(stderr, "RUN FAILED\n");
    void* host = NULL;
    LiteRtLockTensorBuffer(out_buf[0], &host, kLiteRtTensorBufferLockModeRead);
    if (out_i8[0]) {
      const signed char* s = (const signed char*)host;
      for (long k = 0; k < out_cnt[0]; ++k) outv[k] = ((float)s[k] - out_z[0]) * out_s[0];
    } else {
      memcpy(outv, host, (size_t)out_cnt[0] * 4);
    }
    LiteRtUnlockTensorBuffer(out_buf[0]);
    send_all(c, outv, (size_t)out_cnt[0] * 4);
    close(c);
    fprintf(stderr, "served forward\n"); fflush(stderr);
  }
  return 0;
}
