from setuptools import setup, find_packages

setup(
    name="styletts2",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "cached_path",
        "nltk",
        "scipy",
        "numpy",
        "munch",
        "librosa",
        "sounddevice",
        "einops",
        "einops_exts",
        "transformers",
        "matplotlib",
        "monotonic_align @ git+https://github.com/resemble-ai/monotonic_align.git",
    ]
)
