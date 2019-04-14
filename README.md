# img2midi: A model for midi generation using image

## Requirements
- Python 3.5
- cuda 9.0

## Install
- `sudo apt-get install libasound2-dev`
- `conda create -n py35 python=3.5 anaconda`
- `source activate py35`
- `pip install git+https://github.com/vishnubob/python-midi@feature/python3`
- `git clone https://github.com/Liu-Ze/DeepJ.git`
- `cd DeepJ`
- `pip install -r requirements.txt`
- `conda install pytorch torchvision cudatoolkit=9.0 -c pytorch`

## RUN DEEPJ
- `python generate.py --model archives/model.pt --style w1 w2 w3 w4 (wi means the weight of style i : [Baroque, Classical, Romantic, Modern])`

## RUN Place365
- `python run_placesCNN_basic.py`

## RUN img2midi
`python img2midi.py --model archives/model.pt --fname test.jpg [--arch resnet18 --style w1 w2 w3 w4 ...]`

## music style
- Baroque(巴洛克): 俗丽凌乱, 运动, 转变
- Classical(古典主义): 平衡, 清晰, 刚毅
- Romantic(浪漫主义): 情感热烈, 激情, 直觉, 想像力
- Modern(现代主义)

