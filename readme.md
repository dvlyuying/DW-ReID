## Dataset

Our synthetic datasets are at following:

SW-DukeMTMC-reID:
[Rainy weather](https://drive.google.com/file/d/1ZC-hOWHfZAytAef0KZvNLhfnCI364a6k), 
[Snowy weather](https://drive.google.com/file/d/13xcn1oQIUJbYVXeHRCfSwDiostSurXaF), 
[Hazy weather](https://drive.google.com/file/d/1guBDf15tbySUUUxKvFmjz6fxDVodKdpE),
Normal weather [DukeMTMC-reID](https://arxiv.org/abs/1609.01775)

SW-Market-1501:
[Rainy weather](https://drive.google.com/file/d/1wtkZjidVV2anpMCT58jlgdqQ3XDQjvzO), 
[Snowy weather](https://drive.google.com/file/d/16tvPXMVJJS1ecd0xj4EnJ1RzKDJReP29), 
[Hazy weather](https://drive.google.com/file/d/1NYmGS-xRkcz3ugWDqtf8zZ7GjpbF9qIG),
Normal weather [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)


### Notice
The full training code in `processor/processor_dwreid_stage2.py` will be open-sourced after acceptance.

### Requirements
```
conda create -n dw-reid python=3.8
conda activate dw-reid
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install yacs
pip install timm
pip install scikit-image
pip install tqdm
pip install ftfy
pip install regex
```

### Train
1. Edit config file (for person ReID): `configs/person/vit_dwreid.yml`
2. Set these fields in the config:
```
DATASETS:
  NAMES: ('dukemtmc')
  ROOT_DIR: ('/your_dataset_dir')
OUTPUT_DIR: '/your_output_dir'
```
3. Run training:
```
python train_dwreid.py --config_file configs/person/vit_dwreid.yml
```

### Test
1. Prepare a trained checkpoint path for `TEST.WEIGHT`.
2. Run standard test:
```
python test_dwreid.py --config_file configs/person/vit_dwreid.yml TEST.WEIGHT '/your_output_dir/your_model.pth' TEST.WEATHER no
```
3. Run weather test (requires released weather dataset):
```
python test_dwreid.py --config_file configs/person/vit_dwreid.yml TEST.WEIGHT '/your_output_dir/your_model.pth' TEST.WEATHER yes
```

### Notes
- Current scripts `train_dwreid.py` and `test_dwreid.py` set `CUDA_VISIBLE_DEVICES="1"` inside code. Change that line if you want another GPU.
