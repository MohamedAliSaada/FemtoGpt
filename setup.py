from setuptools import setup, find_packages

setup(
    name="FemtoGpt",
    version="0.1",
    description="A builder to GPT models from screatch",
    author="Mohamed saada",
    packages=find_packages(),
    install_requires=["tensorflow>=2.11"],
)
