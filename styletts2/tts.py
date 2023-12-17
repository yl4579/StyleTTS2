import torch
import torchaudio
import yaml
import librosa
import nltk
import os
import random

import numpy as np
from nltk.tokenize import word_tokenize
from cached_path import cached_path

# Import necessary modules and functions
from .Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from .Utils.PLBERT.util import load_plbert
from .models import build_model, load_ASR_models, load_F0_models
from .text_utils import TextCleaner
from .utils import recursive_munch


class TTS:
    def __init__(self, model_params, model, device):
        if not nltk.find('tokenizers/punkt'):
            nltk.download('punkt')

        self.model_params = model_params
        self.model = model
        self.device = device
        self.text_cleaner = TextCleaner()
        self.schedule = KarrasSchedule(
            sigma_min=0.0001, sigma_max=3.0, rho=9.0)
        self.sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=self.schedule,
            clamp=False
        )

    @classmethod
    def load_model(cls, config_path, checkpoint_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = yaml.safe_load(open(str(cached_path(config_path))))

        # Process paths and load components
        ASR_config = cls.fix_path(config.get('ASR_config', False))
        ASR_path = cls.fix_path(config.get('ASR_path', False))
        F0_path = cls.fix_path(config.get('F0_path', False))
        BERT_path = cls.fix_path(config.get('PLBERT_dir', False))

        text_aligner = load_ASR_models(ASR_path, ASR_config)
        pitch_extractor = load_F0_models(F0_path)
        plbert = load_plbert(BERT_path)

        model_params = recursive_munch(config['model_params'])
        model = build_model(model_params, text_aligner,
                            pitch_extractor, plbert)

        # Load state dicts
        params_whole = torch.load(
            str(cached_path(checkpoint_path)), map_location='cpu')
        params = params_whole['net']
        cls.load_state_dicts(model, params)

        [model[key].eval() for key in model]
        [model[key].to(device) for key in model]

        return cls(model_params, model, device)

    @staticmethod
    def fix_path(path):
        # if path is relative, make it absolute
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), path)
        return path

    @staticmethod
    def load_state_dicts(model, params):
        for key in model:
            if key in params:
                state_dict = params[key]
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                try:
                    model[key].load_state_dict(state_dict)
                except RuntimeError as e:
                    print(f"Error loading state dict for {key}: {e}")

    @staticmethod
    def preprocess_audio(wave, mean=-4, std=4):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor

    @staticmethod
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(
            0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess_audio(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def inference(self, text, ref_s, prev_s=None, alpha=0.3, beta=0.7, t=0.7, phonemizer=None, diffusion_steps=5, embedding_scale=1):
        if phonemizer is None:
            raise ValueError("Phonemizer is required for inference")

        # Preprocess text
        text = text.strip()
        ps = phonemizer.phonemize([text])
        ps = ' '.join(word_tokenize(ps[0]))

        # Prepare tokens
        tokens = torch.LongTensor(
            [0] + self.text_cleaner(ps)).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor(
                [tokens.shape[-1]]).to(self.device)
            text_mask = self.length_to_mask(input_lengths).to(self.device)

            # Encode text
            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(
                tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            # Predict style
            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,  # reference from the same speaker as the embedding
                num_steps=diffusion_steps
            ).squeeze(1)

            if prev_s is not None:
                # Convex combination of previous and current style
                s_pred = t * prev_s + (1 - t) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            s_pred = torch.cat([ref, s], dim=-1)

            # Predict duration
            d = self.model.predictor.text_encoder(
                d_en, s, input_lengths, text_mask)
            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            # Create alignment target
            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # Encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            # Predict F0 and N
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            # Decode
            out = self.model.decoder(
                asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        # Fix weird pulse at the end later
        return out.squeeze().cpu().numpy()[..., :-100], s_pred
