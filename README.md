# NLP group project - Zero Shot Image Classification

## Dataset 

The `links.txt` file contains the animalia links for 1570 species that could be used to build our dataset.

You can verify that it contains no duplicates using (on UNIX) `sort links.txt | uniq -d`, this looks at all the links and calculates duplicates. If there are no duplicates, the output should be empty.

## am3-pytorch

Reimplementation of [AM3](https://papers.nips.cc/paper/2019/file/d790c9e6c0b5e02c87b375e782ac01bc-Paper.pdf) ([original codebase](https://github.com/ElementAI/am3)) in PyTorch using Torchmeta library.

