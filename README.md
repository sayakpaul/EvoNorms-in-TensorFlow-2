Presents implementations of `EvoNormB0` and `EvoNormS0` layers as proposed in [Evolving Normalization-Activation Layers](https://arxiv.org/pdf/2004.02967.pdf) by Liu et al. The authors showed the results with these layers tested on MobileNetV2, ResNets, MnasNet, and EfficientNets. However, I tried a **Mini Inception architecture** as shown in [this blog post](https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/) with the **CIFAR10** dataset.

## Acknowledgements
- [Hanxiao Liu](https://github.com/quark0) for helping me to [correct the implementation](https://github.com/sayakpaul/EvoNorms-in-TensorFlow-2/issues/1). 

## TensorFlow version
2.2.0-rc3 (the version when I was testing the code on Colab)

## About the files
- `Mini_Inception_BN_ReLU.ipynb`: Shows a bunch of experiments with the Mini Inception architecture and BN-ReLU combination.
- `Mini_Inception_EvoNorm.ipynb`: Shows implementations of `EvoNormB0` and `EvoNormS0` layers and experiments with the Mini Inception architecture. 
- `Mini_Inception_EvoNorm_Sweep.ipynb`: Does a hyperparameter search on the `groups` hyperparameter of `EvoNormS0` layers along with a few other hyperparameters. 
- `layer_utils`: Ships `EvoNormB0` and `EvoNormS0` layers as stand-alone classes in `tf.keras`. 

## Experimental Summary
Follow experimental summary [here](https://bit.ly/3arUw9q).

## References
- [3 ways to create a Keras model with TensorFlow 2.0 (Sequential, Functional, and Model Subclassing)](https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/) by PyImageSearch
- [Evolving Normalization-Activation Layers](https://www.youtube.com/watch?v=RFn5eH5ZCVo) video guide by Henry AI Labs
- [EvoNorms_PyTorch](https://github.com/lonePatient/EvoNorms_PyTorch)
