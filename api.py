import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
import phonemizer
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from typing import Tuple, Type, Union
from numpy.typing import NDArray
import os
import nltk
nltk.download('punkt')


def load_phonemizer_configs_asr_f0_bert(language:str="en-us", config_path:str="./Configs/config.yml")->Tuple[any, dict, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    global_phonemizer = phonemizer.backend.EspeakBackend(language=language, preserve_punctuation=True,  with_stress=True)

    config = yaml.safe_load(open(config_path))


    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # load BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    return global_phonemizer, config, text_aligner, pitch_extractor, plbert

def load_model(weight_path:str, config:dict, 
               text_aligner:torch.nn.Module, pitch_extractor:torch.nn.Module,
                 plbert:torch.nn.Module, device:str='cpu')->Tuple[torch.nn.Module, any]:
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    params_whole = torch.load(weight_path, map_location='cpu')
    params = params_whole['net']
    

    for key in model:
        if key in params:
            # print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
    #             except:
    #                 _load(params[key], model[key])
    _ = [model[key].eval() for key in model]


    return model, model_params

def load_sampler(model:torch.nn.Module)->torch.nn.Module:
    sampler = DiffusionSampler(model.diffusion.diffusion,
                               sampler=ADPM2Sampler(),
                               sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
                               clamp=False)
    
    return sampler


class StyleTTS:
    def __init__(self, 
                 config_path:str="./Configs/config.yml", 
                 model_path:str=None, 
                 language:str="en-us", 
                 device:str='cpu',
                 load_from_HF:bool=True, 
                 model_remote_path:str="https://huggingface.co/yl4579/StyleTTS2-LibriTTS"):
        
        if load_from_HF is True:
            if model_path is None: 
                cwd = os.getcwd()
                model_path = os.path.join(cwd,"models_weight")
                if not os.path.exists(model_path):
                    os.makedirs(model_path, exist_ok=True)
                    os.system(f"git clone {model_remote_path} {model_path}")
                config_path = os.path.join(model_path, "Models", "LibriTTS", "config.yml")
                model_path = os.path.join(model_path, "Models", "LibriTTS", "epochs_2nd_00020.pth")

        self.model_remote_path = model_remote_path
        self.config_path = config_path
        self.model_path = model_path
        self.language = language
        self._device = device

        (self.global_phonemizer, 
         self.config, 
         self.text_aligner, 
         self.pitch_extractor, 
         self.plbert) = load_phonemizer_configs_asr_f0_bert(language=language, config_path=self.config_path)
        

        self.model, self.model_params = load_model(weight_path=model_path, 
                                                   config=self.config, 
                                                   text_aligner=self.text_aligner, 
                                                   pitch_extractor=self.pitch_extractor,
                                                   plbert=self.plbert,
                                                   device=device)
        
        self.sampler = load_sampler(model=self.model)

        self.textclenaer = TextCleaner()

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean, self.std = -4, 4

    def __call__(self, text:str, ref_s:NDArray=None, alpha:float=0.3, 
                 beta:float=0.7, diffusion_steps:float=5, embedding_scale:float=1) -> NDArray:
        return self.predict(text=text, 
                            ref_s=ref_s, 
                            alpha=alpha, 
                            beta=beta, 
                            diffusion_steps=diffusion_steps, 
                            embedding_scale=embedding_scale)

    def predict(self, text:str, ref_s:NDArray=None, alpha:float=0.3, 
                beta:float=0.7, diffusion_steps:float=5, embedding_scale:float=1) -> NDArray:
        
        if ref_s is None: ref_s = self.load_random_ref_s()

        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self._device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self._device)
            text_mask = self.length_to_mask(input_lengths).to(self._device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self._device),
                                  embedding=bert_dur,
                                  embedding_scale=embedding_scale,
                                  features=ref_s, # reference from the same speaker as the embedding
                                  num_steps=diffusion_steps).squeeze(1)


            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)


            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self._device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self._device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))


        return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later 

    def compute_style(self, wave=None, sr=None, path=None, device='cpu')->torch.Tensor:
        if path is not None:
            wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess(audio).to(device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)
    
    def length_to_mask(self, lengths:NDArray)->torch.Tensor:
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def preprocess(self, wave:NDArray)->torch.Tensor:
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def _predict_long_step(self, text:str, s_prev:NDArray, ref_s:NDArray=None, 
                           alpha:float=0.3, beta:float=0.7, t:float=0.7, 
                           diffusion_steps:int=5, embedding_scale:int=1)->NDArray:
        if ref_s is None: ref_s = self.load_random_ref_s()
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        ps = ps.replace('``', '"')
        ps = ps.replace("''", '"')

        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self._device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self._device)
            text_mask = self.length_to_mask(input_lengths).to(self._device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self._device),
                                            embedding=bert_dur,
                                            embedding_scale=embedding_scale,
                                                features=ref_s, # reference from the same speaker as the embedding
                                                num_steps=diffusion_steps).squeeze(1)

            if s_prev is not None:
                # convex combination of previous and current style
                s_pred = t * s_prev + (1 - t) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            s_pred = torch.cat([ref, s], dim=-1)

            d = self.model.predictor.text_encoder(d_en,
                                            s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)


            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self._device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self._device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))


        return out.squeeze().cpu().numpy()[..., :-100], s_pred # weird pulse at the end of the model, need to be fixed later
    
    def predict_long(self, text:str, ref_s:NDArray=None, alpha:float=0.3, 
                     beta:float=0.7, diffusion_steps:float=5, 
                     embedding_scale:float=1, t:float=.7) -> NDArray:
        if ref_s is None: ref_s = self.load_random_ref_s()
        sentences = text.split('.') # simple split by dot (what about split_and_recombine_text tortoise. I'll check it out later)
        wavs = []
        s_prev = None
        for text in sentences:
            if text.strip() == "": continue
            text += '.' # add it back

            wav, s_prev = self._predict_long_step(text,
                                                  s_prev,
                                                  ref_s,
                                                  alpha=alpha,
                                                  beta=beta,  # make it more suitable for the text
                                                  t=t,
                                                  diffusion_steps=diffusion_steps, 
                                                  embedding_scale=embedding_scale)
            wavs.append(wav)

        return np.concatenate(wavs, axis=0)

    def load_random_ref_s(self):
        return torch.randn(1, 256).to(self._device)
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device:str):
        self._device = device

    def to(self, device:str):
        self.device = device


if __name__ == "__main__":
    stts = StyleTTS()
    sr = 24000
    wave = np.random.randn(sr*10)
    
    print(stts("read this in a random voice").shape)
    print(stts.predict("read this in a random voice").shape)
    print(stts.predict_long("simple split by dot (what about split_and_recombine_text tortoise. I'll check it out later)").shape)


    