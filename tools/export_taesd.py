#!/usr/bin/env python3
"""TAESD-декодер в tflite вместо полного VAE (тот же контракт: [1,4,64,64] -> [1,3,512,512], выход [-1,1]).

Полный VAE-декодер занимает ~4 с из 34 с генерации. TAESD — крошечный аналог (~5 МБ против 95 МБ),
качество для 4-шагового LCM практически то же.

Сверка идёт на НАСТОЯЩЕМ латенте с траектории LCM, а не на шуме.
"""

import sys

import numpy as np
import torch
import torch.nn as nn

OUT = sys.argv[1] if len(sys.argv) > 1 else "out/taesd_decoder.tflite"

print(
    ">>> гружу SD1.5 + LCM-LoRA (для эталонного латента и полного VAE)...", flush=True
)
from diffusers import AutoencoderTiny, LCMScheduler, StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None
)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# финальный латент реальной генерации
lat = {}


def grab(_pipe, i, t, kw):
    lat["z"] = kw["latents"]
    return kw


print(">>> генерирую эталонный латент (4 шага LCM)...", flush=True)
pipe(
    "a photograph of an astronaut riding a horse",
    num_inference_steps=4,
    guidance_scale=1.5,
    generator=torch.manual_seed(42),
    callback_on_step_end=grab,
    output_type="latent",
)
z = lat["z"].float()
print(f"    латент {tuple(z.shape)} [{z.min():.2f}, {z.max():.2f}]")

print(">>> полный VAE (эталон)...", flush=True)
with torch.no_grad():
    ref = pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample.float()

print(">>> TAESD...", flush=True)
taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd").float().eval()
with torch.no_grad():
    tae = taesd.decode(z).sample.float()

r, t_ = ref.numpy().ravel(), tae.numpy().ravel()
print(
    f"    полный VAE [{r.min():.2f}, {r.max():.2f}]  TAESD [{t_.min():.2f}, {t_.max():.2f}]"
)
print(f"    corr={np.corrcoef(r, t_)[0, 1]:.4f}  MAE={np.abs(r - t_).mean():.4f}")


def save_png(x, path):
    from PIL import Image

    a = ((x[0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


save_png(ref, "out/vae_full.png")
save_png(tae, "out/vae_taesd.png")
print("    картинки: out/vae_full.png, out/vae_taesd.png")


class Dec(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, z):
        return self.m.decode(z).sample


print(">>> конвертирую в tflite...", flush=True)
import litert_torch as L

L.convert(Dec(taesd).eval(), (torch.randn(1, 4, 64, 64),)).export(OUT)
import os

print("OK:", os.path.getsize(OUT), "bytes ->", OUT)
