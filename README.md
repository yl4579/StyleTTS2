# StyleTTS 2 API

[Original Repo](https://github.com/yl4579/StyleTTS2) - [CLI Tool](https://github.com/fakerybakery/styletTS2-cli) - **Streaming API**

(GPL licensed due to Phonemizer. Should I switch to OpenPhonemizer and make it MIT-licensed?)


StyleTTS 2 is by Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani. I am not affiliated with them.

> In this paper, we present StyleTTS 2, a text-to-speech (TTS) model that leverages style diffusion and adversarial training with large speech language models (SLMs) to achieve human-level TTS synthesis. StyleTTS 2 differs from its predecessor by modeling styles as a latent random variable through diffusion models to generate the most suitable style for the text without requiring reference speech, achieving efficient latent diffusion while benefiting from the diverse speech synthesis offered by diffusion models. Furthermore, we employ large pre-trained SLMs, such as WavLM, as discriminators with our novel differentiable duration modeling for end-to-end training, resulting in improved speech naturalness. StyleTTS 2 surpasses human recordings on the single-speaker LJSpeech dataset and matches it on the multispeaker VCTK dataset as judged by native English speakers. Moreover, when trained on the LibriTTS dataset, our model outperforms previous publicly available models for zero-shot speaker adaptation. This work achieves the first human-level TTS synthesis on both single and multispeaker datasets, showcasing the potential of style diffusion and adversarial training with large SLMs.

Paper: [https://arxiv.org/abs/2306.07691](https://arxiv.org/abs/2306.07691)

Audio samples: [https://styletts2.github.io/](https://styletts2.github.io/)

Online demo: [Hugging Face](https://huggingface.co/spaces/styletts2/styletts2) (thank [@fakerybakery](https://github.com/fakerybakery) for the wonderful online demo)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/) [![Slack](https://img.shields.io/badge/Join%20Our%20Community-Slack-blue)](https://join.slack.com/t/styletts2/shared_invite/zt-2805io6cg-0ROMhjfW9Gd_ix_FJqjGmQ)

## TODO
- [x] Training and inference demo code for single-speaker models (LJSpeech)
- [x] Test training code for multi-speaker models (VCTK and LibriTTS)
- [x] Finish demo code for multispeaker model and upload pre-trained models
- [x] Add a finetuning script for new speakers with base pre-trained multispeaker models
- [x] REST API
- [x] Importable inference script (PR #78)
- [x] Streaming API and WebSocket support
- [ ] Fix DDP (accelerator) for `train_second.py` **(I have tried everything I could to fix this but had no success, so if you are willing to help, please see [#7](https://github.com/yl4579/StyleTTS2/issues/7))**
- [ ] Pip package

## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
```
3. Install python requirements: 
```bash
pip install -r requirements.txt
```
On Windows add:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U
```
Also install phonemizer and espeak if you want to run the demo:
```bash
pip install phonemizer
sudo apt-get install espeak-ng
```
4. Download and extract the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/), unzip to the data folder and upsample the data to 24 kHz. The text aligner and pitch extractor are pre-trained on 24 kHz data, but you can easily change the preprocessing and re-train them using your own preprocessing. 
For LibriTTS, you will need to combine train-clean-360 with train-clean-100 and rename the folder train-clean-460 (see [val_list_libritts.txt](https://github.com/yl4579/StyleTTS/blob/main/Data/val_list_libritts.txt) as an example).

## Streaming API

You can use StyleTTS 2 in your projects by launching the HTTP API with streaming support. Synthesize text from your frontend apps, etc by making HTTP calls to the API server. The server uses Flask.

API documentation may be found in the [`API_DOCS.md`](API_DOCS.md) file.

Launch server:

```bash
python api.py
```

## WebSocket API

For real-time TTS streaming with chunked text input and low-latency audio output, use the WebSocket API powered by FastAPI.

**Features:**
- Real-time bidirectional communication
- Chunked text input (send text incrementally)
- Base64-encoded MP3 audio output
- GPU queue management for concurrent requests
- Idle timeout and connection management

**Quick Start:**

```bash
# Start WebSocket server (default port 8765)
python ws_server.py

# Or with custom port
python ws_server.py 9000
```

**Endpoints:**
- WebSocket: `ws://localhost:8765/ws/tts`
- Health Check: `http://localhost:8765/health`
- Voice List: `http://localhost:8765/voices`

**Test the WebSocket API:**

```bash
# Simple test
python test_ws_client.py --text "Hello world" --voice f-us-1 --output output.mp3

# Chunked streaming test
python test_ws_client.py --text "This is a longer text" --voice m-us-2 --chunked
```

Full WebSocket documentation: [`WEBSOCKET_DOCS.md`](WEBSOCKET_DOCS.md)

## Python API

You can now use StyleTTS 2 directly in your programs! A `pip`-compatible package is coming soon.

Multi-Speaker Inference:

```python
from scipy.io.wavfile import write
import msinference
text = 'Hello world!'
voice = msinference.compute_style('voice.wav')
wav = msinference.inference(text, voice, alpha=0.3, beta=0.7, diffusion_steps=7, embedding_scale=1)
write('result.wav', 24000, wav)
```

LJSpeech Inference:

```python
from scipy.io.wavfile import write
import ljinference
text = 'Hello world!'
noise = torch.randn(1,1,256).to('cuda' if torch.cuda.is_available() else 'cpu')
wav = ljinference.inference(text, noise, diffusion_steps=7, embedding_scale=1)
write('result.wav', 24000, wav)
```

For longer text, you can [help implement #54](https://github.com/yl4579/StyleTTS2/issues/54) or use Tortoise TTS for splitting:

```python
from tortoise.utils.text import split_and_recombine_text
import numpy as np
from scipy.io.wavfile import write
import msinference
text = 'Long text here...'
texts = split_and_recombine_text(text)
audios = []
voice = msinference.compute_style('voice.wav')
for t in texts:
    audios.append(msinference.inference(t, voice, alpha=0.3, beta=0.7, diffusion_steps=7, embedding_scale=1))
write('result.wav', 24000, np.concatenate(audios))
```

## GUI

You can run inference (finetuning coming soon) on a GUI based on the [online demo](https://huggingface.co/spaces/styletts2/styletts2) powered by Gradio.

```bash
python app.py
```

**NOTE: Only the multi-speaker tab supports long-text currently.**

Note: the online demo will be updated more frequently as changes are pushed directly to it (rather than through PRs). If you would like to use the latest (potentially unstable) version, use Docker:

```bash
docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all registry.hf.space/styletts2-styletts2:latest python app.py
```

## Training
First stage training:
```bash
accelerate launch train_first.py --config_path ./Configs/config.yml
```
Second stage training **(DDP version not working, so the current version uses DP, again see [#7](https://github.com/yl4579/StyleTTS2/issues/7) if you want to help)**:
```bash
python train_second.py --config_path ./Configs/config.yml
```
You can run both consecutively and it will train both the first and second stages. The model will be saved in the format "epoch_1st_%05d.pth" and "epoch_2nd_%05d.pth". Checkpoints and Tensorboard logs will be saved at `log_dir`. 

The data list format needs to be `filename.wav|transcription|speaker`, see [val_list.txt](https://github.com/yl4579/StyleTTS2/blob/main/Data/val_list.txt) as an example. The speaker labels are needed for multi-speaker models because we need to sample reference audio for style diffusion model training. 

### Important Configurations
In [config.yml](https://github.com/yl4579/StyleTTS2/blob/main/Configs/config.yml), there are a few important configurations to take care of:
- `OOD_data`: The path for out-of-distribution texts for SLM adversarial training. The format should be `text|anything`.
- `min_length`: Minimum length of OOD texts for training. This is to make sure the synthesized speech has a minimum length.
- `max_len`: Maximum length of audio for training. The unit is frame. Since the default hop size is 300, one frame is approximately `300 / 24000` (0.0125) second. Lowering this if you encounter the out-of-memory issue. 
- `multispeaker`: Set to true if you want to train a multispeaker model. This is needed because the architecture of the denoiser is different for single and multispeaker models.
- `batch_percentage`: This is to make sure during SLM adversarial training there are no out-of-memory (OOM) issues. If you encounter OOM problem, please set a lower number for this. 

### Pre-trained modules
In [Utils](https://github.com/yl4579/StyleTTS2/tree/main/Utils) folder, there are three pre-trained models: 
- **[ASR](https://github.com/yl4579/StyleTTS2/tree/main/Utils/ASR) folder**: It contains the pre-trained text aligner, which was pre-trained on English (LibriTTS), Japanese (JVS), and Chinese (AiShell) corpus. It works well for most other languages without fine-tuning, but you can always train your own text aligner with the code here: [yl4579/AuxiliaryASR](https://github.com/yl4579/AuxiliaryASR).
- **[JDC](https://github.com/yl4579/StyleTTS2/tree/main/Utils/JDC) folder**: It contains the pre-trained pitch extractor, which was pre-trained on English (LibriTTS) corpus only. However, it works well for other languages too because F0 is independent of language. If you want to train on singing corpus, it is recommended to train a new pitch extractor with the code here: [yl4579/PitchExtractor](https://github.com/yl4579/PitchExtractor).
- **[PLBERT](https://github.com/yl4579/StyleTTS2/tree/main/Utils/PLBERT) folder**: It contains the pre-trained [PL-BERT](https://arxiv.org/abs/2301.08810) model, which was pre-trained on English (Wikipedia) corpus only. It probably does not work very well on other languages, so you will need to train a different PL-BERT for different languages using the repo here: [yl4579/PL-BERT](https://github.com/yl4579/PL-BERT). You can also replace this module with other phoneme BERT models like [XPhoneBERT](https://arxiv.org/abs/2305.19709) which is pre-trained on more than 100 languages.

### Common Issues
- **Loss becomes NaN**: If it is the first stage, please make sure you do not use mixed precision, as it can cause loss becoming NaN for some particular datasets when the batch size is not set properly (need to be more than 16 to work well). For the second stage, please also experiment with different batch sizes, with higher batch sizes being more likely to cause NaN loss values. We recommend the batch size to be 16. You can refer to issues [#10](https://github.com/yl4579/StyleTTS2/issues/10) and [#11](https://github.com/yl4579/StyleTTS2/issues/11) for more details.
- **Out of memory**: Please either use lower `batch_size` or `max_len`. You may refer to issue [#10](https://github.com/yl4579/StyleTTS2/issues/10) for more information.
- **Non-English dataset**: You can train on any language you want, but the current bottleneck is PL-BERT. The pre-trained PL-BERT in English would still work for other languages, but it will not be as good as English (you may refer to [yl4579/StyleTTS#10](https://github.com/yl4579/StyleTTS/issues/10) and [#70](https://github.com/yl4579/StyleTTS2/issues/70) for some examples to train on Chinese datasets). We are currently planning on training multi-lingual PL-BERT models for the best performance. You can go to [#41](https://github.com/yl4579/StyleTTS2/issues/41) if you would like to help.

## Finetuning
The script is modified from `train_second.py` which uses DP, as DDP does not work for `train_second.py`. Please see the bold section above if you are willing to help with this problem. 
```bash
python train_finetune.py --config_path ./Configs/config_ft.yml
```
Please make sure you have the LibriTTS checkpoint downloaded and unzipped under the folder. The default configuration `config_ft.yml` finetunes on LJSpeech with 1 hour of speech data (around 1k samples) for 50 epochs. This took about 4 hours to finish on four NVidia A100. The quality is slightly worse (similar to NaturalSpeech on LJSpeech) than LJSpeech model trained from scratch with 24 hours of speech data, which took around 2.5 days to finish on four A100. The samples can be found at [#65 (comment)](https://github.com/yl4579/StyleTTS2/discussions/65#discussioncomment-7668393). 

If you are using a **single GPU** (because the script doesn't work with DDP) and want to save training speed and VRAM, you can do (thank [@korakoe](https://github.com/korakoe) for making the script at [#100](https://github.com/yl4579/StyleTTS2/pull/100)):
```bash
accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_accelerate.py --config_path ./Configs/config_ft.yml
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/Colab/StyleTTS2_Finetune_Demo.ipynb)

### Common Issues
[@Kreevoz](https://github.com/Kreevoz) has made detailed notes on common issues in finetuning, with suggestions in maximizing audio quality: [#81](https://github.com/yl4579/StyleTTS2/discussions/81). Some of these also apply to training from scratch. [@IIEleven11](https://github.com/IIEleven11) has also made a guideline for fine-tuning: [#128](https://github.com/yl4579/StyleTTS2/discussions/128).

- **Out of memory after `joint_epoch`**: This is likely because your GPU RAM is not big enough for SLM adversarial training run. You may skip that but the quality could be worse. Setting `joint_epoch` a larger number than `epochs` could skip the SLM advesariral training.

## Inference
Please refer to [Inference_LJSpeech.ipynb](https://github.com/yl4579/StyleTTS2/blob/main/Demo/Inference_LJSpeech.ipynb) (single-speaker) and [Inference_LibriTTS.ipynb](https://github.com/yl4579/StyleTTS2/blob/main/Demo/Inference_LibriTTS.ipynb) (multi-speaker) for details. For LibriTTS, you will also need to download [reference_audio.zip](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/reference_audio.zip) and unzip it under the `demo` before running the demo. 

- The pretrained StyleTTS 2 on LJSpeech corpus in 24 kHz can be downloaded at [https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main](https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main).

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/Colab/StyleTTS2_Demo_LJSpeech.ipynb)

- The pretrained StyleTTS 2 model on LibriTTS can be downloaded at [https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main). 

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/Colab/StyleTTS2_Demo_LibriTTS.ipynb)


You can import StyleTTS 2 and run it in your own code. However, the inference depends on a GPL-licensed package, so it is not included directly in this repository. A [GPL-licensed fork](https://github.com/NeuralVox/StyleTTS2) has an importable script, as well as an experimental streaming API, etc. A [fully MIT-licensed package](https://pypi.org/project/styletts2/) that uses gruut (albeit lower quality due to mismatch between phonemizer and gruut) is also available.  

***Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.*** 

### Common Issues
- **High-pitched background noise**: This is caused by numerical float differences in older GPUs. For more details, please refer to issue [#13](https://github.com/yl4579/StyleTTS2/issues/13). Basically, you will need to use more modern GPUs or do inference on CPUs.
- **Pre-trained model license**: You only need to abide by the above rules if you use **the pre-trained models** and the voices are **NOT** in the training set, i.e., your reference speakers are not from any open access dataset. For more details of rules to use the pre-trained models, please see [#37](https://github.com/yl4579/StyleTTS2/issues/37).

## References
- [archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)
- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [nii-yamagishilab/project-NN-Pytorch-scripts/project/01-nsf](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf)

## License

**NOTE: By contributing to this software you agree that the license may be changed in the future once I find a phonemizer replacement.**

This package depends on `phonemizer`, which is GPL licensed. [Check out the original repository for a MIT-licensed version w/o the API!](https://github.com/yl4579/StyleTTS2). I'm working on a permissively licensed Phonemizer - coming soon!

NOTE: By contributing to this project you agree that the authors may change the license in the future

Copyright (C) 2023 Aaron (Yinghao) Li (under the MIT license). 
Modifications copyright (C) 2023-2024 mrfakename.

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

**This software was previously licensed under the MIT license:**

MIT License

Copyright (c) 2023 Aaron (Yinghao) Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
