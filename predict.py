import time
import random
from collections import OrderedDict

import yaml
import nltk
import torch
import librosa
import numpy as np
import torchaudio
import phonemizer
from torch import nn
import torch.nn.functional as F
from munch import Munch
from pydub import AudioSegment
import IPython.display as ipd
from nltk.tokenize import word_tokenize
from cog import BasePredictor, Input, Path

from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, 
KarrasSchedule

textclenaer = TextCleaner()
nltk.download("punkt")


def load_model(config_path, ckpt_path):
    config = yaml.safe_load(open(config_path))

    # Load pretrained ASR model, F0 and BERT models
    plbert = load_plbert(config.get('PLBERT_dir', False))
    pitch_extractor = load_F0_models(config.get("F0_path", False))
    text_aligner = load_ASR_models(config.get("ASR_path", False), 
config.get("ASR_config", False))

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, 
plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to("cuda") for key in model]

    params_whole = torch.load(ckpt_path, map_location="cpu")
    params = params_whole["net"]

    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except:
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)

    _ = [model[key].eval() for key in model]
    return model, model_params

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions 
efficient"""
        self.device = "cuda"
        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us', preserve_punctuation=True, with_stress=True, 
words_mismatch="ignore"
        )
        self.model, _ = load_model(
            config_path="Models/LJSpeech/config.yml", 
ckpt_path="Models/LJSpeech/epoch_2nd_00100.pth"
        )
        self.model_ref, self.model_ref_config = load_model(
            config_path="Models/LibriTTS/config.yml", 
ckpt_path="Models/LibriTTS/epochs_2nd_00020.pth"
        )

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, 
rho=9.0),
            clamp=False
        )
        self.sampler_ref = DiffusionSampler(
            self.model_ref.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, 
rho=9.0), # empirical parameters
            clamp=False
        )
    def length_to_mask(self, lengths):
        mask = 
torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], 
-1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def preprocess(self, wave):
        to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        mean, std = -4, 4

        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / 
std
        return mel_tensor

    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model_ref.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = 
self.model_ref.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def inference(
        self, text, noise, s_prev=None, diffusion_steps=5, 
embedding_scale=1, alpha=0.7, pad=True
    ):
        text = text.strip()
        text = text.replace('"', '')
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = 
torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, 
text_mask)
            bert_dur = self.model.bert(tokens, 
attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise,
                embedding=bert_dur[0].unsqueeze(0), 
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale
            ).squeeze(0)

            # convex combination of previous and current style
            if s_prev is not None:
                s_pred = alpha * s_prev + (1 - alpha) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            d = self.model.predictor.text_encoder(d_en, s, input_lengths, 
text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            if pad: 
                pred_dur[-1] += 5

            pred_aln_trg = torch.zeros(input_lengths, 
int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 
1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ 
pred_aln_trg.unsqueeze(0).to(self.device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder(
                (t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),
                F0_pred, N_pred, ref.squeeze().unsqueeze(0)
            )
        return out.squeeze().cpu().numpy(), s_pred
    
    def inference_with_ref_old(
        self, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, 
embedding_scale=1
    ):
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = 
torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model_ref.text_encoder(tokens, input_lengths, 
text_mask)
            bert_dur = self.model_ref.bert(tokens, 
attention_mask=(~text_mask).int())
            d_en = self.model_ref.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler_ref(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s, 
                num_steps=diffusion_steps
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]
            d = self.model_ref.predictor.text_encoder(d_en, s, 
input_lengths, text_mask)

            x, _ = self.model_ref.predictor.lstm(d)
            duration = self.model_ref.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, 
int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 
1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ 
pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_ref_config.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model_ref.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_ref_config.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model_ref.decoder(asr, F0_pred, N_pred, 
ref.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the 
end of the model, need to be fixed later

    def inference_with_ref(
        self, text, ref_s, s_prev=None, alpha=0.3, beta=0.7, t=0.7, 
        diffusion_steps=5, embedding_scale=1, trim=50, longform=False
    ):
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        ps = ps.replace('``', '"')
        ps = ps.replace("''", '"')

        tokens = textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = 
torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model_ref.text_encoder(tokens, input_lengths, 
text_mask)
            bert_dur = self.model_ref.bert(tokens, 
attention_mask=(~text_mask).int())
            d_en = self.model_ref.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler_ref(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s, 
                num_steps=diffusion_steps
            ).squeeze(1)

            # convex combination of previous and current style
            if s_prev is not None:
                s_pred = t * s_prev + (1 - t) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            if s_prev is not None or longform==True:
                s_pred = torch.cat([ref, s], dim=-1)

            d = self.model_ref.predictor.text_encoder(d_en, s, 
input_lengths, text_mask)
            x, _ = self.model_ref.predictor.lstm(d)
            duration = self.model_ref.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, 
int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 
1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ 
pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_ref_config.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model_ref.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_ref_config.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model_ref.decoder(asr, F0_pred, N_pred, 
ref.squeeze().unsqueeze(0))
        
        return out.squeeze().cpu().numpy()[..., :-trim], s_pred
    
    def predict(
        self,
        text: str = Input(description="Text to convert to speech"),
        reference: Path = Input(description="Reference speech to copy 
style from", default=None),
        alpha: float = Input(
            description="Only used for long text inputs or in case of 
reference speaker, \
            determines the timbre of the speaker. Use lower values to 
sample style based \
            on previous or reference speech instead of text.", ge=0, le=1, 
default=0.3
        ),
        beta: float = Input(
            description="Only used for long text inputs or in case of 
reference speaker, \
            determines the prosody of the speaker. Use lower values to 
sample style based \
            on previous or reference speech instead of text.", ge=0, le=1, 
default=0.7
        ),
        diffusion_steps: int = Input(description="Number of diffusion 
steps", ge=0, le=50, default=10),
        embedding_scale: float = Input(description="Embedding scale, use 
higher values for pronounced emotion", ge=0, le=5, default=1),
        seed: int = Input(description="Seed for reproducibility", 
default=0)
    ) -> Path:
        """Run a single prediction on the model"""

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if reference is None:
            if len(text) >= 400:
                sentences = nltk.sent_tokenize(text)
                wavs = []
                s_prev = None
                for sent in sentences:
                    if sent.strip() == "": continue
                    noise = torch.randn(1, 1, 256).to(self.device)
                    wav, s_prev = self.inference(
                        sent, noise, s_prev=s_prev, alpha=alpha, 
                        diffusion_steps=diffusion_steps, 
embedding_scale=embedding_scale, pad=False
                    )
                    wavs.append(wav)
                wav = np.concatenate(wavs)
            else:
                noise = torch.randn(1, 1, 256).to(self.device)
                wav, _ = self.inference(text, noise, 
diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)

        else:
            ref_s = self.compute_style(str(reference))
            if len(text) >= 400:
                wavs = []
                s_prev = None
                sentences = nltk.sent_tokenize(text)
                for sent in sentences:
                    if sent.strip() == "": continue
                    wav, s_prev = self.inference_with_ref(
                        sent, ref_s, s_prev, alpha=alpha, beta=beta, 
t=0.7, trim=100,
                        diffusion_steps=diffusion_steps, 
embedding_scale=embedding_scale, longform=True 
                    )
                    wavs.append(wav)
                wav = np.concatenate(wavs)
            else:
                noise = torch.randn(1, 1, 256).to(self.device)
                ref_s = self.compute_style(str(reference))
                wav, _ = self.inference_with_ref(
                    text, ref_s, alpha=alpha, beta=beta, 
diffusion_steps=diffusion_steps, embedding_scale=embedding_scale
                )

        out_path = "/tmp/out.mp3"
        audio = ipd.Audio(wav, rate=24000, normalize=False)
        audio = AudioSegment(audio.data, frame_rate=24000, sample_width=2, 
channels=1)
        audio.export(out_path, format="mp3", bitrate="64k")
        return Path(out_path)

