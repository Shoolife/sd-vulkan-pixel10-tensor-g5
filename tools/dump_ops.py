#!/usr/bin/env python3
"""
op-coverage на уровне ГРАФА: выгрузить все типы операций из .tflite + счётчики.

Это не «что лёг на TPU» (для этого нужен Tensor-компилятор), а «какие op-типы вообще
есть в графе» — то, что компилятору Tensor придётся поддержать. Помечаем известных
кандидатов на CPU-фолбэк, чтобы сразу видеть риск.

    python tools/dump_ops.py tools/out/resnet_block.tflite tools/out/transformer_block.tflite
"""

import sys

# Op-типы LiteRT, исторически проблемные для NPU/TPU-делегатов (частый CPU-фолбэк).
RISKY = {
    "SOFTMAX": "attention softmax",
    "GELU": "GEGLU/активация",
    "LOGISTIC": "SiLU=x*sigmoid (часто через LOGISTIC+MUL)",
    "RSQRT": "norm (group/layer)",
    "MEAN": "norm-редукция",
    "SQUARED_DIFFERENCE": "var в norm",
    "L2_NORMALIZATION": "norm",
    "BATCH_MATMUL": "attention QK/PV (динам. формы)",
    "TRANSPOSE": "перестановки голов",
    "GATHER": "индексирование",
}


def dump(path):
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    counts = {}
    for op in interp._get_ops_details():
        name = op["op_name"]
        counts[name] = counts.get(name, 0) + 1

    total = sum(counts.values())
    print(f"\n=== {path} ===")
    print(f"всего операций: {total}, уникальных типов: {len(counts)}")
    print(f"{'op':<22}{'кол-во':>7}  риск-фолбэка")
    for name, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        flag = f"  ⚠ {RISKY[name]}" if name in RISKY else ""
        print(f"{name:<22}{c:>7}{flag}")
    risky_here = [n for n in counts if n in RISKY]
    if risky_here:
        print(f"кандидаты на CPU-фолбэк ({len(risky_here)}): {', '.join(risky_here)}")
    else:
        print("явных кандидатов на фолбэк среди известных нет")
    return counts


def main():
    if len(sys.argv) < 2:
        print("usage: python dump_ops.py model1.tflite [model2.tflite ...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        dump(p)


if __name__ == "__main__":
    main()
