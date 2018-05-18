# CNNIQA
PyTorch implementation of the following paper:
[Kang L, Ye P, Li Y, et al. Convolutional neural networks for no-reference image quality assessment[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014: 1733-1740.](http://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf)

### Note
The optimizer is chosen as Adam here, instead of the SGD with momentum in the paper.

## Training
```bash
CUDA_VISIBLE_DEVICES=0 python CNNIQA.py 0 config.yaml LIVE CNNIQA
```
Before training, the `im_dir` in `config.yaml` must to be specified.

### Visualization
```bash
tensorboard --logdir='./logs' --port=6006
```
## Requirements
- PyTorch 
- TensorFlow-TensorBoard if `enableTensorboard` in `config.yaml` is `True`.