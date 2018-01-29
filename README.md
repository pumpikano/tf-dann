# Domain-Adversarial Training of Neural Networks in Tensorflow

"[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf)" introduced a simple and effective method for accomplishing domain adaptation with SGD with a gradient reversal layer. This work was elaborated and extended in "[Domain-Adversarial Training of Neural Networks](http://jmlr.org/papers/volume17/15-239/15-239.pdf)". For more information as well as a link to an equivalent implementation in Caffe, see http://sites.skoltech.ru/compvision/projects/grl/.

The `Blobs-DANN.ipynb` shows some basic experiments on a very simple dataset. The `MNIST-DANN.ipynb` recreates the MNIST experiment from the papers on a synthetic dataset. Instructions to generate the synthetic dataset are below.

Requires TensorFlow>=1.0 and tested with Python 2.7 and Python 3.4.

## Gradient Reversal Layer

The `flip_gradient` operation is implemented in Python by using `tf.gradient_override_map` to override the gradient of `tf.identity`. Refer to `flip_gradient.py` to see how this is implemented.

```python
from flip_gradient import flip_gradient

# Flip the gradient of y w.r.t. x and scale by l (defaults to 1.0)
y = flip_gradient(x, l)
```

## MNIST Experiments

The `MNIST-DANN.ipynb` notebook implements the MNIST experiments for the paper with the same model and optimization parameters, including the learning rate and adaptation parameter schedules. Rough results are below (more training would likely improve results - # epochs are not reported in the paper).

| Method | Target acc (paper) | Target acc (this repo w/ 10 epochs) |
| ------ | ------------------ | ----------------------------------- |
| Source Only | 0.5225 | 0.4801 |
| DANN | 0.7666 | 0.7189 |


### Build MNIST-M dataset

The MNIST-M dataset consists of MNIST digits blended with random color patches from the [BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500) dataset. To generate a MNIST-M dataset, first download the BSDS500 dataset and run the `create_mnistm.py` script:

```bash
curl -L -O http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
python create_mnistm.py
```

This may take a couple minutes and should result in a `mnistm_data.pkl` file containing generated images.


## Contribution

It would be great to add other experiments to this repository. Feel free to make a PR if you decide to recreate other results from the papers or new experiments entirely.
