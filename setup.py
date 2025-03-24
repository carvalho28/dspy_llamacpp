from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dspy_llamacpp",
    version="0.1.0",
    author="Diogo Carvalho",
    author_email="carvalhoo287@gmail.com",
    description="A lightweight Python wrapper for llama.cpp with seamless dspy integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carvalho28/dspy_llamacpp",
    packages=find_packages(),
    install_requires=[
        "dspy",
        "python-dotenv",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires='>=3.6',
    keywords="llama cpp dspy integration",
    license="MIT",
)