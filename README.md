<h1>Ship Detection</h1>

## Installation

```bash
conda create --name mmd python=3.8
conda activate mmd
conda install "pytorch==1.10.1" "torchvision==0.11.2" "cudatoolkit==10.2.*" -c pytorch -y

pip install openmim
mim install "mmengine==0.8.4"
mim install "mmcv==2.0.1"
mim install "mmdet==3.0.0"
 
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
git checkout "d50ab76"
pip install -v -e .
```

Or 

```
conda create --name rtmdet python=3.8 -y
conda activate rtmdet
conda install pytorch==1.12.1 torchvision cudatoolkit -c pytorch
pip install openmim==0.3.9
mim install mmcv-full==1.7.2
mim install mmdet==2.28.2
pip install "mmrotate @ git+ssh://git@github.com/open-mmlab/mmrotate.git@v1.0.0rc1"
```

## Sample usage

### Training

```bash
conda activate mmd
python train.py
```

### Inference
```bash
conda activate mmd
python infer.py
```
