#!/usr/bin/env python3
"""
Разведка TPU, этап 2: экспорт ПОЛНОГО UNet SD1.5 в .tflite + численная сверка.

Тот же приём, что в export_blocks_tflite.py, но на всей модели и с НАСТОЯЩИМИ весами:
- грузим UNet2DConditionModel (diffusers, HF) — fp32 или fp16;
- конвертируем в .tflite через litert-torch;
- СВЕРЯЕМ выход tflite с PyTorch-эталоном (relErr + Pearson corr) — оракул качества,
  как у нашего Vulkan-движка (там relErr был 0.0003 fp32 / 0.0024 fp16);
- дальше op-coverage гоняется тем же dump_ops.py, делегация — Tensor ML SDK Beta.

Тяжело: UNet ~860M параметров. Конвертация занимает минуты и много RAM; HF-загрузка
весов ~3.2 ГБ (fp32) / ~1.6 ГБ (fp16). Запускать на ПК, не на телефоне.

    pip install -r tools/requirements-export.txt
    # быстрая проверка без экспорта (только загрузка + corr torch-self):
    python tools/export_unet_tflite.py --check --no-export
    # полный экспорт fp16 + сверка с эталоном:
    python tools/export_unet_tflite.py --dtype fp16 --check --out tools/out
    # batch-CFG (cond+uncond одним графом):
    python tools/export_unet_tflite.py --batch 2 --out tools/out

Дальше:
    python tools/dump_ops.py tools/out/unet.tflite
"""

import argparse
import os

import torch
import torch.nn as nn

# SD1.5, 512×512: латент 4×64×64, контекст 77×768.
LAT_C, LAT_HW = 4, 64
CTX_N, CTX_D = 77, 768
TIMESTEP = 999.0


class UNetWrap(nn.Module):
    """UNet2DConditionModel -> позиционный forward(latent, timestep, ctx) -> noise_pred."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, latent, timestep, ctx):
        return self.m(latent, timestep, ctx, return_dict=False)[0]


def sample_inputs(batch):
    # Всегда fp32: CPU-forward для эталона надёжен; fp16 даёт квантизация при конвертации.
    latent = torch.randn(batch, LAT_C, LAT_HW, LAT_HW)
    timestep = torch.full((batch,), TIMESTEP)
    ctx = torch.randn(batch, CTX_N, CTX_D)
    return (latent, timestep, ctx)


def load_unet(model_id):
    from diffusers import UNet2DConditionModel

    print(f"[1/4] Загружаю UNet {model_id} (fp32)...")
    unet = (
        UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").float().eval()
    )
    return UNetWrap(unet).eval()


def fp16_quant_config():
    """QuantConfig для fp16-хранения весов (ровно 2× меньше, как наш боевой fp16-движок)."""
    from litert_torch.generative.quantize import quant_recipes

    return quant_recipes.full_fp16_recipe()


def metrics(a, b):
    """relErr (как у Vulkan-self-test) + Pearson corr."""
    a = a.detach().flatten().double()
    b = b.detach().flatten().double()
    relerr = ((a - b).norm() / b.norm()).item()
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return relerr, corr


def aet():
    try:
        import litert_torch as m
    except ImportError:
        import ai_edge_torch as m
    return m


def tflite_infer(path, inputs):
    """Прогон .tflite на CPU-интерпретаторе; входы матчим по числу измерений/форме."""
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    details = interp.get_input_details()
    # сопоставляем входы по форме (latent 4D, ctx 3D, timestep 1D)
    for arr in inputs:
        np_arr = arr.detach().cpu().numpy()
        for d in details:
            if tuple(d["shape"]) == tuple(np_arr.shape):
                interp.set_tensor(d["index"], np_arr.astype(d["dtype"]))
                break
    interp.invoke()
    out = interp.get_output_details()[0]
    return torch.from_numpy(interp.get_tensor(out["index"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--out", default="tools/out")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--batch", type=int, default=1, help="1 или 2 (batch-CFG)")
    ap.add_argument(
        "--check", action="store_true", help="сверить tflite с PyTorch-эталоном"
    )
    ap.add_argument(
        "--no-export", action="store_true", help="только загрузка/torch-прогон"
    )
    args = ap.parse_args()

    torch.manual_seed(0)
    unet = load_unet(args.model_id)
    inputs = sample_inputs(args.batch)

    print("[2/4] PyTorch-эталон (forward, fp32)...")
    with torch.no_grad():
        ref = unet(*inputs)
    print(
        f"      out {tuple(ref.shape)} (ждём ({args.batch},{LAT_C},{LAT_HW},{LAT_HW}))"
    )

    if args.no_export:
        print("--no-export: стоп после torch-прогона.")
        return

    os.makedirs(args.out, exist_ok=True)
    fp16 = args.dtype == "fp16"
    path = os.path.join(args.out, f"unet{'_fp16' if fp16 else ''}.tflite")
    qcfg = fp16_quant_config() if fp16 else None
    print(
        f"[3/4] Конвертация UNet -> tflite ({'fp16-веса' if fp16 else 'fp32'}, долго, ~860M)..."
    )
    edge = aet().convert(unet, inputs, quant_config=qcfg)
    edge.export(path)
    print(f"      -> {path}  ({os.path.getsize(path) / 1e6:.0f} МБ)")

    if args.check:
        print("[4/4] Сверка tflite vs PyTorch-эталон (fp32)...")
        got = tflite_infer(path, inputs).float()
        relerr, corr = metrics(got, ref)
        bar = "OK" if relerr < 0.01 else "ПРОВЕРИТЬ"
        print(f"      relErr={relerr:.4f}  corr={corr:.6f}  [{bar}]")
        print("      (ориентир Vulkan: fp32 0.0003 / fp16 0.0024)")

    print("\nДальше: python tools/dump_ops.py " + path)
    print("Затем — делегация под Tensor (SDK Beta) + замер vs 32.9 с warm (GPU).")


if __name__ == "__main__":
    main()
