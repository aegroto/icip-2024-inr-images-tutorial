# Implicit Image Compression: Encoding Pictures with Implicit Neural Representations
### IEEE International Conference on Image Processing  (ICIP) 2024

## Setup 

- Make sure to clone the *live* branch
- Download and install the [pixi](https://pixi.sh/) project manager
- Run ```pixi r capabilities``` and check if CUDA is available in your system. Although this is not mandatory, it is highly recommended as the code is computationally intensive.

## Tutorial Abstract

Implicit Neural Representations (INRs) are a very recent paradigm for information representation where discrete data are interpreted as continuous functions from coordinates to samples. In the case of images, this function maps each pixel’s coordinates to the colour of the pixels. A neural network is then over-fit to this function, then the image is reconstructed through inference of this network. By following this workflow, the image data are encoded as network parameters.

Recent researches refer to this paradigm as Implicit Image Compression and have demonstrated how codecs based on this emerging approach obtain good visual and quantitative results and can outperform well-established codecs, while not suffering from long-known defects such as block artifacts. Plus, it is possible to fit directly on specific metrics during the training process but, in contrast with other learned methods such as autoencoders, no pre-trained models are needed to encode images.

This tutorial will begin with a brief introduction to the concepts behind this innovative technique, and then a Python implementation of a basic yet complete image codec using INRs will be presented, focusing on modularity and ease of comparison between different results, to provide a speed-up to the research on the field. An in-depth overview of the contribution of each network architecture choice and common model compression techniques, such as quantization, will be presented, such that the audience will achieve a consistent knowledge of the field and will be suddenly able to experiment with the state-of-the-art of implicit image compression.
