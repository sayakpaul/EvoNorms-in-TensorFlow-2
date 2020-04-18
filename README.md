Presents implementations of `EvoNormB0` and `EvoNormS0` layers as proposed in [Evolving Normalization-Activation Layers](https://arxiv.org/pdf/2004.02967.pdf) by Liu et al. The authors showed the results with these layers tested on MobileNetV2, ResNets, and EfficientNets. However, I tried a **Mini Inception architecture** as shown in [this blog post](https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/) with the **CIFAR10** dataset.

## TensorFlow version
2.2.0-rc3 (the version when I was testing the code on Colab)

## About the files
- `Mini_Inception_BN_ReLU.ipynb`: Shows a bunch of experiments with the Mini Inception architecture and BN-ReLU combination.
- `Mini_Inception_EvoNorm.ipynb`: Shows implementations of `EvoNormB0` and `EvoNormS0` layers and experiments with the Mini Inception architecture. 
- `layer_utils`: Ships `EvoNormB0` and `EvoNormS0` layers as stand-alone classes in `tf.keras`. 

## Performance comparison

## Experimental Summary
Follow experimental summary here: https://app.wandb.ai/sayakpaul/evonorm-tf2. 

## References
- [Evolving Normalization-Activation Layers](https://www.youtube.com/watch?v=RFn5eH5ZCVo) video guide by Henry AI Labs.
- https://github.com/lonePatient/EvoNorms_PyTorch.
