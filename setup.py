import os
from setuptools import setup, find_packages


setup(
   name='StyleTTS2',
   version='2.0',
   description='A text-to-speech (TTS) model that leverages style diffusion and adversarial training with large speech language models (SLMs) to achieve human-level TTS synthesis.',
   license='MIT',
   package_dir={'styletts2':'./'},
   long_description=open('README.md').read(),
   packages=find_packages(),
   install_requires=find_packages(),
   url="https://github.com/yl4579/StyleTTS2.git",
)