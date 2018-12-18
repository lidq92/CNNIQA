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
```bash
python test_cross_dataset.py --help
```
TODO: add metrics calculation. SROCC, KROCC can be easily get. PLCC, RMSE, MAE, OR should be calculated after a non-linear fitting since the quality score ranges are not the same across different IQA datasets.

### Visualization
```bash
tensorboard --logdir=tensorboard_logs --port=6006 # in the server
ssh -L 6006:localhost:6006 user@host # in your PC, then see the visualization in your PC
```
## Requirements
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- PyTorch 0.4
- TensorboardX 1.2, TensorFlow-TensorBoard
- [pytorch/ignite](https://github.com/pytorch/ignite)
