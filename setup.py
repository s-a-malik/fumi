import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zanim",
    version="0.0.1",
    author=
    "Matthew Jackson, Shreshth Malik, Michael Matthews, Yousuf Mohamed-Ahmed",
    author_email="youmed.tech@gmail.com",
    description=
    "Package for the NLP group-project on multi-modal few-shot learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/s-a-malik/zero-shot",
    install_requires=[
        'torch>=1.0.0',
        "tqdm==4.60.0",
        "torchvision==0.9.1",
        "torch==1.8.1",
        "numpy==1.20.2",
        "torchmeta==1.7.0",
        "h5py==3.2.1",
        "wandb==0.10.26",
        "pandas==1.2.4",
        "gensim==4.0.1",
        "nltk==3.6.1",
        "transformers==4.5.1",
    ],
    include_package_data=True,
    packages=["zanim"],
    python_requires=">=3.6",
)