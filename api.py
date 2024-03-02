"""
StyleTTS2 API module.
"""
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

from .models import *
from .utils import *
from .text_utils import TextCleaner
import phonemizer
from .Utils.PLBERT.util import load_plbert
from .Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from typing import Tuple, Type, Union
from numpy.typing import NDArray
import os

# runs first time after installation only
import nltk
nltk.download('punkt')


import pathlib
ROOT = pathlib.Path(__file__).parent.resolve()


def load_phonemizer_configs_asr_f0_bert(language:str="en-us", 
                                        config_path:str="./Configs/config.yml",
                                        add_cwd:bool=True)->Tuple[any, dict, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """
    Load the necessary configurations and models for phonemizer, ASR, F0, and BERT.

    Args:
        language (str, optional): The language for the phonemizer backend. Defaults to "en-us".
        config_path (str, optional): The path to the configuration file. Defaults to "./Configs/config.yml".
        add_cwd (bool, optional): Whether to add the current working directory to the paths. This is used to load default models only.  Defaults to True.

    Returns:
        Tuple[any, dict, torch.nn.Module, torch.nn.Module, torch.nn.Module]: A tuple containing the global phonemizer,
        the configuration dictionary, the text aligner model, the pitch extractor model, and the BERT model.
    """
    global_phonemizer = phonemizer.backend.EspeakBackend(language=language, preserve_punctuation=True,  with_stress=True)
    
    if add_cwd is True:
        config_path = os.path.join(ROOT, config_path)
    config = yaml.safe_load(open(config_path))


    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    if add_cwd is True:
        ASR_path = os.path.join(ROOT, ASR_path)
        ASR_config = os.path.join(ROOT, ASR_config)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    if add_cwd is True:
        F0_path = os.path.join(ROOT, F0_path)
    pitch_extractor = load_F0_models(F0_path)

    # load BERT model
    BERT_path = config.get('PLBERT_dir', False)
    if add_cwd is True:
        BERT_path = os.path.join(ROOT, BERT_path)
    plbert = load_plbert(BERT_path)

    return global_phonemizer, config, text_aligner, pitch_extractor, plbert

def load_model(weight_path:str, config:dict, 
               text_aligner:torch.nn.Module, pitch_extractor:torch.nn.Module,
               plbert:torch.nn.Module, device:str='cpu')->Tuple[torch.nn.Module, any]:
    """
    Loads a pre-trained model with the specified weight path and configuration.

    Args:
        weight_path (str): The path to the pre-trained model weights.
        config (dict): The configuration dictionary for building the model.
        text_aligner (torch.nn.Module): The text aligner module. Returned by load_phonemizer_configs_asr_f0_bert.
        pitch_extractor (torch.nn.Module): The pitch extractor module. Returned by load_phonemizer_configs_asr_f0_bert.
        plbert (torch.nn.Module): The plbert module. Returned by load_phonemizer_configs_asr_f0_bert.
        device (str, optional): The device to load the model on. Defaults to 'cpu'.

    Returns:
        Tuple[torch.nn.Module, any]: A tuple containing the loaded model and its parameters.
    """
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

def load_sampler(model: torch.nn.Module) -> torch.nn.Module:
    """
    Loads a diffusion sampler for the given model.

    Args:
        model (torch.nn.Module): The model to load the sampler for. Returned by load_model.

    Returns:
        torch.nn.Module: The loaded diffusion sampler.
    """
    sampler = DiffusionSampler(model.diffusion.diffusion,
                               sampler=ADPM2Sampler(),
                               sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
                               clamp=False)
    
    return sampler


class StyleTTS:
    def __init__(self, 
                 config_path:str=None, 
                 model_path:str=None, 
                 language:str="en-us", 
                 device:str='cpu',
                 load_from_HF:bool=True, 
                 model_remote_path:str="https://huggingface.co/yl4579/StyleTTS2-LibriTTS"):
        """
        Initializes the API object for StyleTTS2.

        Args:
            config_path (str, optional): Path to the configuration file. Defaults to None.
            model_path (str, optional): Path to the model file. Defaults to None. If None, will use LJ Speech model.
            language (str, optional): Language code. Defaults to "en-us". More languages will be added in the future with multi language plbert.
            device (str, optional): Device to run the model on. Defaults to 'cpu'.
            load_from_HF (bool, optional): Whether to load the model from Hugging Face. Defaults to True.
            model_remote_path (str, optional): Remote path to the model. Defaults to "https://huggingface.co/yl4579/StyleTTS2-LibriTTS".
        """
        add_cwd = False
        if config_path is None: add_cwd = True

        if load_from_HF is True:
            if model_path is None: 
                
                model_path = os.path.join(ROOT,"models_weight")
                if not os.path.exists(model_path):
                    os.makedirs(model_path, exist_ok=True)
                    os.system(f"git clone {model_remote_path} {model_path}")
                config_path = os.path.join("models_weight", "Models", "LibriTTS", "config.yml")
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
         self.plbert) = load_phonemizer_configs_asr_f0_bert(language=language, 
                                                            config_path=self.config_path, 
                                                            add_cwd=add_cwd)


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
            """
            Call the model to generate speech with the given input text and optional reference style. wrapper for predict.

            Args:
                text (str): The input text for speech generation.
                ref_s (NDArray, optional): The reference style for speech generation. Defaults to None.
                alpha (float, optional): The weight of the reference style in the generated speech. Defaults to 0.3.
                beta (float, optional): The weight of the input text in the generated speech. Defaults to 0.7.
                diffusion_steps (float, optional): The number of diffusion steps for speech generation. Defaults to 5.
                embedding_scale (float, optional): The scale factor for the input text embedding. This is the classifier-free guidance scale. 
                                                    The higher the scale, the more conditional the style is to the input text and hence more emotional. 
                                                    Defaults to 1.

            Returns:
                NDArray: The generated speech waveform.
            """
            return self.predict(text=text, 
                                ref_s=ref_s, 
                                alpha=alpha, 
                                beta=beta, 
                                diffusion_steps=diffusion_steps, 
                                embedding_scale=embedding_scale)

    def predict(self, text:str, ref_s:NDArray=None, alpha:float=0.3, 
                    beta:float=0.7, diffusion_steps:float=5, embedding_scale:float=1) -> NDArray:
        """
        Generates speech waveform for the given input text.

        Args:
            text (str): The input text to be synthesized.
            ref_s (NDArray, optional): Reference speaker embedding. Returned by compute_style. Defaults to None.
            alpha (float, optional): Alpha value for controlling timbr. Defaults to 0.3 (70% of the reference timbre).
            beta (float, optional): Beta value for controlling the prosody. Defaults to 0.7 (30% of the reference prosody).
            diffusion_steps (float, optional): Number of diffusion steps for sampling the speech. Defaults to 5.
            embedding_scale (float, optional): The scale factor for the input text embedding. This is the classifier-free guidance scale. 
                                                    The higher the scale, the more conditional the style is to the input text and hence more emotional. 
                                                    Defaults to 1.

        Returns:
            NDArray: The generated speech waveform.
        """
        
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
        """
        Compute the style representation for the given audio. If path is provided, it will load the audio from the path.
        Otherwise, it will use the wave and sr arguments.

        Args:
            wave (np.ndarray, optional): Audio waveform. Defaults to None.
            sr (int, optional): Sample rate of the audio. Defaults to None.
            path (str, optional): Path to the audio file. Defaults to None.
            device (str, optional): Device to use for computation. Defaults to 'cpu'.

        Returns:
            torch.Tensor: Style representation tensor.
        """
        if path is not None:
            wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
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
        """
        Preprocesses the input waveform by converting it to a mel spectrogram tensor.

        Args:
            wave (numpy.ndarray): The input waveform.

        Returns:
            torch.Tensor: The preprocessed mel spectrogram tensor.
        """
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def predict_long_step(self, text:str, s_prev:NDArray, ref_s:NDArray=None, 
                           alpha:float=0.3, beta:float=0.7, t:float=0.7, 
                           diffusion_steps:int=5, embedding_scale:float=1)->NDArray:
        """
            Predicts the output audio waveform for a given input text and style.

            Args:
                text (str): The input text to be synthesized.
                s_prev (NDArray): The previous style embedding.
                ref_s (NDArray, optional): The reference style embedding. If not provided, a random reference style is loaded. Defaults to None.
                alpha (float, optional): Alpha value for controlling timbr. Defaults to 0.3 (70% of the reference timbre).
                beta (float, optional): Beta value for controlling the prosody. Defaults to 0.7 (30% of the reference prosody).
                t (float, optional): The convex combination factor between the previous and current style. Defaults to 0.7.
                diffusion_steps (int, optional): The number of diffusion steps. Defaults to 5.
                embedding_scale (float, optional): The scale factor for the input text embedding. This is the classifier-free guidance scale. 
                                                    The higher the scale, the more conditional the style is to the input text and hence more emotional. 
                                                    Defaults to 1.

            Returns:
                NDArray: The output audio waveform.
                NDArray: The predicted style embedding.
        """
        
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
        """
        Generates a long audio prediction based on the given text.

        Args:
            text (str): The input text to be synthesized.
            ref_s (NDArray, optional): The reference style embedding. If not provided, a random reference style is loaded. Defaults to None.
            alpha (float, optional): Alpha value for controlling timbr. Defaults to 0.3 (70% of the reference timbre).
            beta (float, optional): Beta value for controlling the prosody. Defaults to 0.7 (30% of the reference prosody).
            t (float, optional): The convex combination factor between the previous and current style. Defaults to 0.7.
            diffusion_steps (int, optional): The number of diffusion steps. Defaults to 5.
            embedding_scale (float, optional): The scale factor for the input text embedding. This is the classifier-free guidance scale. 
                                                    The higher the scale, the more conditional the style is to the input text and hence more emotional. 
                                                    Defaults to 1.

        Returns:
            NDArray: The generated audio waveform as a numpy array.
        """
        if ref_s is None: ref_s = self.load_random_ref_s()
        sentences = text.split('.') # simple split by dot (what about split_and_recombine_text tortoise. I'll check it out later)
        wavs = []
        s_prev = None
        for text in sentences:
            if text.strip() == "": continue
            text += '.' # add it back

            wav, s_prev = self.predict_long_step(text,
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
        """
        returns a random style embedding. This ruins the result. Use it only for testing.

        Returns:
            torch.Tensor: A random style embedding tensor.
        """
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


    