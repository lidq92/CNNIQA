# CNNIQA
PyTorch 0.4 implementation of the following paper:
[Kang L, Ye P, Li Y, et al. Convolutional neural networks for no-reference image quality assessment[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014: 1733-1740.](http://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf)

### Note
The optimizer is chosen as Adam here, instead of the SGD with momentum in the paper.

## Training
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --exp_id=0 --database=LIVE
```
Before training, the `im_dir` in `config.yaml` must to be specified.
Train/Val/Test split ratio in intra-database experiments can be set in `config.yaml` (default is 0.6/0.2/0.2).

## Evaluation
Test Demo
```bash
python test_demo.py --im_path=data/I03_01_1.bmp
```
### Cross Dataset
TODO

### Visualization
```bash
tensorboard --logdir=tensorboard_logs --port=6006
```
## Requirements
- PyTorch 0.4
- TensorboardX 1.2, TensorFlow-TensorBoard
- [pytorch/ignite](https://github.com/pytorch/ignite)