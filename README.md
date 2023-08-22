<h1 align="center"> Transformers Architecture Implementation from Scratch </h1>

<p align="center">
  <img alt="Python Version" src="https://img.shields.io/badge/Python-3.x-blue.svg">
  <a href="https://github.com/TheFaheem/Transformers/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
</p>

## Table of Contents

- [Introduction](#introduction)
- [Motivation](#motivation)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Architecture Overview](#architecture-overview)
- [Get Involved](#get-involved)
- [Getting Connected](#connect-with-me)
- [License](#license)

## Introduction

Here I present My Comprehensive Implementation of the Transformers architecture from scratch, which serves as the foundation for many of the current state-of-the-art language models. By practically applying the deep learning knowledge and NLP principles I've acquired thus far, I'm Able to Build this from ground up, which helps me understand the inner working of the transfomer architecture. I beleive, this repository offers a opportunity for those who wants to dive deep into a fundamental transformer architecture.


## Motivation

As a NLP Enthusiast, I was just curios to demystify the renowned Transformers architecture. I wanted to go beyond the surface and understand the inner workings of attention mechanisms, positional encodings, and self/casual attention.

## Getting Started

If you want to play around with the code to test it out or if you want to use this codebase as the base and modify it further, just follow the these steps to get started,
    
### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.10+
- Your favorite development environment (e.g., Jupyter Notebook, VSCode)
- Ensure you have high computational systems, if you want to train the model.

### Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/TheFaheem/Transformers.git
   cd Transformers
   ```
2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
And That's it, You are good to go.

## Usage

```py
from transfomers import TransformersNet

d_model = 512 # Embedding Dimension
inp_vocab_size = 20 # Vocabulary Size of the Input/Source
target_vocab_size = 30 # Vocabulary Size of Target for Projection
input_max_len =  10 # Maximum Sequence Length of the input
target_max_len = 10 # Maximum Sequence Length of the Target
n_blocks = 2 # Number of Encoder/Decoder block for the Model
expansion_factor = 4 # This Determine the Inner Dimension of the feed forward layer
n_heads = 8 # Number of Attention Heads
dropout_size = None # percentage of the layer to drop inbetween the layers to prevent overfitting and stablize the training
batch_size = 32 # Number of input sequence to pass in at a time

model = TransfomersNet(
    d_model,
    inp_vocab_size,
    target_vocab_size,
    input_max_len,
    target_max_len,
    n_blocks = n_blocks,
    expansion_factor=expansion_factor,
    n_heads = n_heads,
    dropout_size = dropout_size
)

output = model(x, y)
# where x, y is the input sequence and target sequence of shape (batch_size, sequence_len) and returns output of shape
# (batch_size, sequence_len, target_vocab_size) where output is the probablity distribution for every word over entire target vocabulary.
```

If You Explore the Codebase and Dug some code a little bit, You'll Find that, Everything is well documented, from arguments, explaination for the argument, inputs to forward pass upto what it will return for every module. Taste Some Code bro!

## Architecture Overview

Check out [Architechture.md](https://github.com/TheFaheem/Transformers/blob/main/Architecture.md) to Get Overview of Each Module in the Transformer Architecture

## Get Involved

I Encourage you to explore the codebase and dug some, analyze the implementation, and use this repository as a base and modify it further according to your need, use it as a resource to enhance your understanding of the Transformers Architecture. Whether you want to improve the code, fix a bug, or wants to add new features just create pull request, I'll check it as soon as i can.

## Acknowledgments

I am immensely grateful for the resources, research papers especialy [Attention is All You Need, Vaswani et al](https://arxiv.org/abs/1706.03762) based on which this repo was created and other open-source projects that have contributed to my learning journey.

## Connect with Me

I'm excited to connect with fellow learners, enthusiasts, and professionals. If you have any questions, suggestions, or just want to chat, feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/thefaheem/) or [Twitter](https://twitter.com/faheem_nlp).

## License

This project is licensed under the terms of the [MIT License](https://github.com/TheFaheem/Transformers/blob/main/LICENSE)
