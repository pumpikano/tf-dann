# Domain-Adversarial Training of Neural Networks in Tensorflow

"[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf)" introduced a simple and effective method for accomplishing domain adaptation with SGD with a gradient reversal layer. This work was elaborated and extended in "[Domain-Adversarial Training of Neural Networks](http://jmlr.org/papers/volume17/15-239/15-239.pdf)".

This repository implements the MNIST domain adaptation experiment presented in "[Domain-Adversarial Training of Neural Networks](http://jmlr.org/papers/volume17/15-239/15-239.pdf)" in Tensorflow. It differs in the details of the optimization (using the `AdamOptimizer` rather than the learning rate schedule detailed in the paper), but implements the same model described in the paper for MNIST experiments.

## Gradient Reversal Layer

The `FlipGradientOp` used in this repo is available as a user op [here](https://github.com/pumpikano/tensorflow/tree/gradient_reversal). If you are using a source installation, you can compile the op with bazel: `bazel build -c opt //tensorflow/core/user_ops:flip_gradient.so`. See https://www.tensorflow.org/versions/r0.9/how_tos/adding_an_op/index.html#adding-a-new-op for instructions to compile with a binary installation.

## Build MNIST-M dataset

The MNIST-M dataset consists of MNIST digits blended with random color patches from the [BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500) dataset. To generate a MNIST-M dataset, first download the BSDS500 dataset and run the `create_mnistm.py` script:

```bash
curl -O http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
python create_mnistm.py
```

This may take a couple minutes and should result in a `mnistm_data.pkl` file containing generated images.

