import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zeroshot",
    version="0.0.1",
    author=
    " Matthew Jackson, Shreshth Malik, Michael Matthews, Yousuf Mohamed-Ahmed",
    author_email="youmed.tech@gmail.com",
    description=
    "Package for the NLP group-project on multi-modal few-shot learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/s-a-malik/zero-shot",
    install_requires=['torch>=1.0.0'],
    include_package_data=True,
    packages=["zanim"],
    python_requires=">=3.6",
)