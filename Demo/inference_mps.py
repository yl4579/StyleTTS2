"""
MPS inference demo for StyleTTS 2.

Usage:
    USE_MPS=1 python Demo/inference_mps.py
    USE_MPS=1 python Demo/inference_mps.py -p "Hello world"

Environment variables:
    USE_MPS  — set to 1 to use Apple Silicon MPS (default: 0, uses CUDA or CPU)
    USE_FP16 — set to 1 to run decoder in fp16 on MPS (default: 0)

Note: TextEncoder must stay on CPU because MPS does not support pack_padded_sequence.
All other modules run on the GPU device.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib'

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

import torch
# PyTorch 2.6+ defaults weights_only=True; upstream checkpoints need pickle
_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{**kwargs, 'weights_only': False})
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

import time
import yaml
import torchaudio
import librosa
import soundfile as sf
from collections import OrderedDict
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
import phonemizer
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# --- Device selection ---

USE_MPS = int(os.getenv("USE_MPS", "0"))
USE_FP16 = int(os.getenv("USE_FP16", "0")) and USE_MPS

if USE_MPS and torch.backends.mps.is_available():
    GPU_DEVICE = 'mps'
elif torch.cuda.is_available():
    GPU_DEVICE = 'cuda'
else:
    GPU_DEVICE = 'cpu'

USE_DTYPE = torch.float16 if USE_FP16 else torch.float32

print(f"Device: {GPU_DEVICE}, FP16: {USE_FP16}")

# --- Audio preprocessing ---

textcleaner = TextCleaner()
global_phonemizer = phonemizer.backend.EspeakBackend(
    language='en-us', preserve_punctuation=True, with_stress=True)

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


# --- Model loading ---

config = yaml.safe_load(open("Models/LibriTTS/config.yml"))

ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert, use_fp16=USE_FP16)

# Load checkpoint
params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu')
params = params_whole['net']


def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


for key in model:
    if key in params:
        try:
            model[key].load_state_dict(params[key])
        except Exception:
            model[key].load_state_dict(fix_state_dict(params[key]), strict=False)
        print(f'{key} loaded')

# Move to device — text_encoder stays on CPU (pack_padded_sequence)
for key in model:
    model[key].eval()
    if key != 'text_encoder':
        model[key].to(GPU_DEVICE)

if USE_FP16:
    model['decoder'].half()

# --- Diffusion sampler ---

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
    clamp=False
)


# --- Inference functions ---

def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    mel_tensor = preprocess(audio).to(GPU_DEVICE)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


def inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    ps = ps.replace('``', '"')
    ps = ps.replace("''", '"')

    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]])
        text_mask = length_to_mask(input_lengths)

        # TextEncoder on CPU (MPS doesn't support pack_padded_sequence)
        t_en = model.text_encoder(tokens, input_lengths, text_mask)

        # Transfer to GPU
        tokens = tokens.to(GPU_DEVICE)
        text_mask = text_mask.to(GPU_DEVICE)
        input_lengths = input_lengths.to(GPU_DEVICE)
        t_en = t_en.to(GPU_DEVICE)

        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        noise = torch.randn((1, 256)).unsqueeze(1).to(GPU_DEVICE)
        s_pred = sampler(
            noise=noise, embedding=bert_dur, embedding_scale=embedding_scale,
            features=ref_s, num_steps=diffusion_steps).squeeze(1)

        s, ref = s_pred[:, 128:], s_pred[:, :128]
        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)

        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # Encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device=GPU_DEVICE, dtype=torch.float32))

        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device=GPU_DEVICE, dtype=USE_DTYPE))

        out = model.decoder(
            asr, F0_pred.to(USE_DTYPE), N_pred.to(USE_DTYPE),
            ref.squeeze().unsqueeze(0).to(USE_DTYPE))

    return out.float().squeeze().cpu().numpy()[..., :-50]


# --- Main loop ---

def synthesize(text, ref_s):
    start = time.time()
    wav = inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)
    elapsed = time.time() - start
    duration = len(wav) / 24000
    sf.write("./synthesized.wav", wav, 24000)
    print(f"Saved synthesized.wav ({duration:.1f}s audio in {elapsed:.1f}s, RTF={elapsed/duration:.2f})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt', help='Text to synthesize (one-shot, no interactive loop)')
    parser.add_argument('-r', '--reference', default='Demo/reference_audio/1221-135767-0014.wav',
                        help='Reference audio path')
    args = parser.parse_args()

    print(f"\nReference: {args.reference}")
    ref_s = compute_style(args.reference)

    text = args.prompt or "StyleTTS 2 is a text to speech model that leverages style diffusion and adversarial training with large speech language models to achieve human level text to speech synthesis."
    synthesize(text, ref_s)

    if args.prompt:
        return

    while True:
        user_input = input("\nEnter text (or 'path|text' to change reference): ").strip()
        if not user_input:
            break

        parts = user_input.split('|', 1)
        if len(parts) > 1:
            style_path = parts[0].strip()
            text = parts[1].strip()
            print(f"Reference: {style_path}")
            ref_s = compute_style(style_path)
        else:
            text = parts[0]

        synthesize(text, ref_s)


if __name__ == "__main__":
    main()
