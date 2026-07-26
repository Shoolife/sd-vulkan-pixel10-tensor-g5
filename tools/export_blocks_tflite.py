#!/usr/bin/env python3
"""
Разведка TPU, этап 1: экспорт ОДНОГО resnet-блока и ОДНОГО transformer-блока
SD1.5 UNet в .tflite (через ai-edge-torch) для проверки op-coverage на Tensor G5.

Зачем именно эти два блока: вместе они содержат ПОЧТИ ВСЕ типы операций UNet —
conv2d, GroupNorm, SiLU, Linear/matmul, self/cross-attention + softmax, GELU/GEGLU,
LayerNorm, residual. Если они целиком ложатся на TPU (а не в CPU-фолбэк) — есть смысл
экспортировать весь UNet (этап 2, тот же приём на полной модели). Это в разы дешевле,
чем сразу гнать 1.6 ГБ.

Маршрут — канонические модули diffusers с настоящими весами SD1.5: граф и набор
операций идентичны реальному UNet, без реверс-инжиниринга нашего bin-layout.

Запуск (на машине с интернетом и питон-окружением, НЕ на телефоне):
    pip install -r tools/requirements-export.txt
    python tools/export_blocks_tflite.py --out tools/out

Выход: tools/out/resnet_block.tflite, tools/out/transformer_block.tflite
Дальше — этап 1.2 (op-coverage), см. tools/README.md.
"""

import argparse
import os

import torch
import torch.nn as nn


# Блок 0 down-блока SD1.5: 320 каналов, латент 64×64, контекст 77×768, 8 голов.
# Те же размерности, что в наших Vulkan-тестах (VulkanBench.resnetCheck/transformerCheck).
RES_C = 320
LAT_HW = 64
TEMB = 1280
CTX_N = 77
CTX_D = 768


class ResnetWrap(nn.Module):
    """ResnetBlock2D.forward(input_tensor, temb) -> tensor."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, temb):
        return self.m(x, temb)


class TransformerWrap(nn.Module):
    """Transformer2DModel (groupnorm+proj_in + attn1/attn2 + GEGLU-FF + proj_out)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, ctx):
        return self.m(x, encoder_hidden_states=ctx, return_dict=False)[0]


def build_blocks_random():
    """Блоки down_blocks[0] SD1.5 со случайной инициализацией — БЕЗ загрузки весов.

    Для op-coverage значения весов не важны: граф и набор операций идентичны реальному
    UNet. Работает офлайн (только diffusers, без HuggingFace-скачивания ~3 ГБ).
    """
    from diffusers.models.resnet import ResnetBlock2D
    from diffusers.models.transformers.transformer_2d import Transformer2DModel

    print("[1/4] Строю блоки down_blocks[0] (random-init, офлайн)...")
    resnet = ResnetBlock2D(
        in_channels=RES_C, out_channels=RES_C, temb_channels=TEMB, groups=32
    )
    transformer = Transformer2DModel(
        num_attention_heads=8,
        attention_head_dim=RES_C // 8,  # 40
        in_channels=RES_C,
        num_layers=1,
        cross_attention_dim=CTX_D,
        norm_num_groups=32,
    )
    return ResnetWrap(resnet).eval(), TransformerWrap(transformer).eval()


def load_blocks_pretrained(model_id: str):
    """Те же блоки с НАСТОЯЩИМИ весами SD1.5 (для численной сверки, этап 2). Качает UNet с HF."""
    from diffusers import UNet2DConditionModel

    print(f"[1/4] Загружаю UNet из {model_id} (subfolder=unet)...")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").eval()
    db0 = unet.down_blocks[0]
    return ResnetWrap(db0.resnets[0]).eval(), TransformerWrap(db0.attentions[0]).eval()


def sanity_forward(resnet, transformer):
    """Прогон случайных входов — убеждаемся, что модули исполняются и формы верны."""
    print("[2/4] Sanity-forward (torch, случайные входы)...")
    x = torch.randn(1, RES_C, LAT_HW, LAT_HW)
    temb = torch.randn(1, TEMB)
    ctx = torch.randn(1, CTX_N, CTX_D)
    with torch.no_grad():
        ry = resnet(x, temb)
        ty = transformer(x, ctx)
    print(f"      resnet out {tuple(ry.shape)} (ждём (1,{RES_C},{LAT_HW},{LAT_HW}))")
    print(
        f"      transformer out {tuple(ty.shape)} (ждём (1,{RES_C},{LAT_HW},{LAT_HW}))"
    )
    return (x, temb), (x, ctx)


def _aet():
    """ai-edge-torch переименован в litert-torch; берём что есть."""
    try:
        import litert_torch as aet
    except ImportError:
        import ai_edge_torch as aet
    return aet


def export(resnet, transformer, res_inputs, tf_inputs, out_dir):
    aet = _aet()

    os.makedirs(out_dir, exist_ok=True)
    print("[3/4] Конвертация resnet-блока -> tflite...")
    res_edge = aet.convert(resnet, res_inputs)
    res_path = os.path.join(out_dir, "resnet_block.tflite")
    res_edge.export(res_path)
    print(f"      -> {res_path}")

    print("[4/4] Конвертация transformer-блока -> tflite...")
    tf_edge = aet.convert(transformer, tf_inputs)
    tf_path = os.path.join(out_dir, "transformer_block.tflite")
    tf_edge.export(tf_path)
    print(f"      -> {tf_path}")
    return res_path, tf_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-id",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="HF repo с SD1.5 (если этот недоступен — Lykon/dreamshaper-8 или локальный путь)",
    )
    ap.add_argument("--out", default="tools/out", help="каталог для .tflite")
    ap.add_argument(
        "--pretrained",
        action="store_true",
        help="загрузить настоящие веса SD1.5 с HF (по умолчанию random-init, офлайн — для op-coverage достаточно)",
    )
    args = ap.parse_args()

    torch.manual_seed(0)
    if args.pretrained:
        resnet, transformer = load_blocks_pretrained(args.model_id)
    else:
        resnet, transformer = build_blocks_random()
    res_inputs, tf_inputs = sanity_forward(resnet, transformer)
    res_path, tf_path = export(resnet, transformer, res_inputs, tf_inputs, args.out)

    print("\nГотово. Дальше — op-coverage (этап 1.2):")
    print(
        "  1) Скомпилировать под Tensor через AOT-компилятор LiteRT (нужен Tensor ML SDK Beta)."
    )
    print(
        "  2) Прочитать отчёт делегации: какие подграфы на TPU, какие в CPU/GPU-фолбэк."
    )
    print(f"  Файлы: {res_path}, {tf_path}")
    print("  Подробности и команды — tools/README.md")


if __name__ == "__main__":
    main()
