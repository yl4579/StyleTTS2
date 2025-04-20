#✅ Training & Debugging StyleTTS2
accelerate launch --num_processes=1 train_first.py --config_path ./Configs/config.yml
accelerate launch --num_processes=2 train_first.py --config_path ./Configs/config.yml
nohup accelerate launch --num_processes=1 train_first.py --config_path ./Configs/config.yml > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
nohup accelerate launch --num_processes=2 train_first.py --config_path ./Configs/config.yml > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
tail -f logs/train_*.log
vi Configs/config.yml
accelerate config

# 🧪 CUDA, GPU & PyTorch Checks
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.get_device_name(0))"
pip freeze | grep torch

# 📦 Python + Env Setup
# Add deadsnakes PPA
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install python3.10 python3.10-dev python3.10-venv
apt install sox gh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch==2.2.2 torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121

# 📁 Dataset & File Navigation
ls, cd, pwd, cat metadata.csv, ls -ltr
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvjf LJSpeech-1.1.tar.bz2

# Git workflows:
git clone https://github.com/jmaty/StyleTTS2.git
git diff, git status, git push, git checkout nan_fix
gh auth login

# SSH & SCP:
scp, ssh root@ip, vi ~/.ssh/authorized_keys
ssh-keygen
# + copying public key to remote

#Audio conversion and inspection:
for file in *.wav; do   sox "$file" -r 24000 -b 16 -c 1 "../../LJSpeech-1.1-24k/wavs/$file"; done
sox --i LJ001-0001.wav
play LJ050-0278.wav

# Troubleshooting:
dmesg | grep -i kill
ps -ef | grep train
kill -9 PID
