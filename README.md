# CNNIQA
PyTorch 1.3 implementation of the following paper:
[Kang L, Ye P, Li Y, et al. Convolutional neural networks for no-reference image quality assessment[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014: 1733-1740.](http://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf)

### Note
- The optimizer is chosen as Adam here, instead of the SGD with momentum in the paper.
- the mat files in data/ are the information extracted from the datasets and the index information about the train/val/test split. The subjective scores of LIVE is from the [realigned data](http://live.ece.utexas.edu/research/Quality/release2/dmos_realigned.mat).

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
tensorboard --logdir=tensorboard_logs --port=6006 # in the server (host:port)
ssh -p port -L 6006:localhost:6006 user@host # in your PC. See the visualization in your PC
```
## Requirements
```bash
conda create -n reproducibleresearch pip python=3.6
source activate reproducibleresearch
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
source deactive
```
- Python 3.6.8
- PyTorch 1.3.0
- TensorboardX 1.9, TensorFlow 2.0.0
- [pytorch/ignite 0.2.1](https://github.com/pytorch/ignite)

Note: You need to install the right CUDA version.

## TODO (If I have free time)
- Simplify the code
- Report results on some common databases
- etc.
