#!/usr/bin/env python3
"""AOT-компиляция под Tensor G5 внутри контейнера + печать ДЕТАЛЬНОЙ ошибки.

Запуск (в образе tensor-aot):
    python3 /work/docker_compile.py /work/out/unet.tflite /work/out/compiled
"""

import glob
import os
import re
import sys

from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.vendors.google_tensor import target as gt_target


def main():
    model = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/work/out/compiled"
    trunc = sys.argv[3] if len(sys.argv) > 3 else "bfloat16"
    os.makedirs(out_dir, exist_ok=True)

    t = gt_target.Target(gt_target.SocModel.TENSOR_G5)
    print(f">>> компилирую {model} под TENSOR_G5 (truncation={trunc}) ...", flush=True)
    compiled = aot_lib.aot_compile(
        model, target=[t], keep_going=True, google_tensor_truncation_type=trunc
    )
    report = compiled.compilation_report()
    print("=== COMPILATION REPORT ===")
    print(report)

    # если есть ссылки на .error — печатаем их полностью (Colab это прячет)
    for err_path in set(re.findall(r"(/tmp/\S+\.error)", report)):
        print(f"\n=== ДЕТАЛЬ ОШИБКИ {err_path} ===")
        try:
            with open(err_path) as f:
                lines = [
                    ln
                    for ln in f
                    if not re.search(
                        r"npu_registry|gpu_registry|accelerator_registry|cpu_registry|outstream",
                        ln,
                    )
                ]
            print("".join(lines[-60:]))
        except OSError as e:
            print("не прочитать:", e)

    if "FAILURES" not in report:
        # имя по исходнику, чтобы int8/lcm/base не перезатирали друг друга
        stem = os.path.splitext(os.path.basename(model))[0]
        name = f"{stem}_g5_{trunc}"
        compiled.export(out_dir, model_name=name)
        print("экспорт:", glob.glob(os.path.join(out_dir, name + "*")))


if __name__ == "__main__":
    main()
