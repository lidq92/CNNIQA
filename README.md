# CNNIQA
Pytorch implementation of the following paper:
Kang L, Ye P, Li Y, et al. Convolutional neural networks for no-reference image quality assessment[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014: 1733-1740.

### Note
The optimizer is chosen as Adam here, instead of the SGD with momentum in the paper.

## Training
```bash
CUDA_VISIBLE_DEVICES=1 python CNNIQA.py 0 config.yaml LIVE CNNIQA
```
Before training, the `im_dir` in `config.yaml` must to be specified.

### Visualization
```bash
tensorboard --logdir='./logs' --port=6006
```
## Requirements
- Pytorch 
- Tensorflow-tensorboard if `enableTensorboard` in `config.yaml` is true.