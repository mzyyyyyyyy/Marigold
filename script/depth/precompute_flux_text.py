"""
Precompute the fixed empty-prompt text conditioning for the FLUX refiner
(depthfm/flux_refiner.py) once, so training never loads T5-XXL / CLIP.

  prompt_embeds: T5-XXL encoding of "" padded/truncated to --seq_len tokens
                 (FLUX's default is 512; the prompt is empty and constant, so a
                 short sequence keeps text tokens from dominating compute)
  pooled_embeds: CLIP-L pooled output of ""

Run on a login node (needs HF access to the model, CPU only):
  python script/depth/precompute_flux_text.py \
      --model /flash/project_465002934/model/FLUX.1-dev --out /flash/project_465002934/model/flux1_empty_text_L32.pt --seq_len 32
"""
import argparse

import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
parser.add_argument("--out", required=True)
parser.add_argument("--seq_len", type=int, default=32)
args = parser.parse_args()

with torch.no_grad():
    tok2 = T5TokenizerFast.from_pretrained(args.model, subfolder="tokenizer_2")
    t5 = T5EncoderModel.from_pretrained(args.model, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).eval()
    ids = tok2([""], padding="max_length", max_length=args.seq_len, truncation=True, return_tensors="pt").input_ids
    prompt_embeds = t5(ids)[0].float()                       # (1, L, 4096)

    tok1 = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    clip = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder", torch_dtype=torch.bfloat16).eval()
    ids1 = tok1([""], padding="max_length", max_length=tok1.model_max_length, truncation=True, return_tensors="pt").input_ids
    pooled = clip(ids1).pooler_output.float()                # (1, 768)

torch.save({"prompt_embeds": prompt_embeds, "pooled_embeds": pooled, "seq_len": args.seq_len}, args.out)
print("saved", args.out, tuple(prompt_embeds.shape), tuple(pooled.shape))
