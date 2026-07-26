#!/usr/bin/env python3
"""
Десктоп-baseline: время одного forward .tflite на CPU-интерпретаторе LiteRT.

Не TPU и не телефон — это опорная точка «сколько стоит граф на CPU ПК», чтобы было
с чем сравнивать device-замер. На устройстве реальные числа даёт bench_device.sh.

    python tools/bench_cpu.py tools/out/unet.tflite --runs 10 --threads 8
"""

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0, help="0 = по умолчанию")
    args = ap.parse_args()

    from ai_edge_litert.interpreter import Interpreter

    kw = {"num_threads": args.threads} if args.threads else {}
    interp = Interpreter(model_path=args.model, **kw)
    interp.allocate_tensors()
    ins = interp.get_input_details()
    for d in ins:
        interp.set_tensor(d["index"], np.random.randn(*d["shape"]).astype(d["dtype"]))

    for _ in range(args.warmup):
        interp.invoke()
    t = []
    for _ in range(args.runs):
        s = time.perf_counter()
        interp.invoke()
        t.append((time.perf_counter() - s) * 1e3)
    t.sort()
    print(f"{args.model}")
    print(
        f"  forward CPU: med={t[len(t) // 2]:.1f} ms  min={t[0]:.1f}  max={t[-1]:.1f}  (n={args.runs})"
    )
    print(
        "  NB: десктоп-CPU baseline. Реальные TPU/GPU-числа — bench_device.sh на Pixel 10."
    )


if __name__ == "__main__":
    main()
