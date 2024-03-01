import os
from sys import platform
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
   name='StyleTTS2',
   version='2.0',
   description='A text-to-speech (TTS) model that leverages style diffusion and adversarial training with large speech language models (SLMs) to achieve human-level TTS synthesis.',
   license='MIT',
   package_dir={'styletts2':'./'},
   long_description=open('README.md').read(),
   install_requires=required,
   url="https://github.com/yl4579/StyleTTS2.git",
   
)