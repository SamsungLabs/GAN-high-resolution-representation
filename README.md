# Learning High-Resolution Domain-Specific Representations with a GAN Generator

![](readme-images/GAN-vis.gif)

This repository contains the pytorch implementation for the method described in the paper "Learning High-Resolution Domain-Specific
Representations with a GAN Generator".

### Installation

- Clone this repo and install dependencies:
```bash
git clone https://github.com/saic-vul/GAN-high-resolution-representation.git
cd GAN-high-resolution-representation
pip3 install -r requirements.txt
```

- Download [stylegan2-models](https://drive.google.com/open?id=1uCAo0X1kdXM9wPmt_gcdnpmUDPGmACkx) converted to pytorch and unzip archive to `stylegan2-models` directory.

- Download [annotated samples](https://drive.google.com/open?id=143dRAyJcRDqygepSz8lIr8ElAnwF3xp_) and unzip archive to `experiments` directory to reproduce experiment with FFHQ hair segmentation.

Be sure that your project structure is 
```
    .
    ├── configs
    ├── experiments
    │   ├── ffhq-hair
    │   |   ├── checkpoints
    |   |   ├── data
    |   |   └── generated
    ├── lib
    ├── stylegan-models
    │   ├── cars.tar
    |   └── ffhq.tar
    └── ...
```

### FFHQ hair segmentation experiment 

#### Prepare config file
```bash
cp config.yml.example config.yml
```
You can specify directory with experiment using `BASE_DIR` parameter, by default it is set to `experiments/ffhq-hair`.

#### Train the Decoder
Train the Decoder using annotated samples from `BASE_DIR/data` (there are 20 annotated samples for ffhq-hair experiment):
```bash
python3 main.py train
```
The decoder weights will be saved to `experiments/ffhq-hair/checkpoints/checkpoint_last.params`.

#### Generate synthetic dataset
```bash
python3 main.py generate
```
This will create `BASE_DIR/dataset/train_generated` directory with generated fake images and synthetic annotation. By default 10000 samples are created.

#### Train and test DeepLabV3+ model on generated samples
Train:
```bash
cd deeplabv3plus/experiments/rgb_segmentation/01_hair_deeplabv3_ffhq_pretrain_gan
python3 main.py train --batch-size=4 --gpus=0 --test-batch-size=4 --workers=4 --kvstore=nccl --input-path "../../../../experiments/ffhq-hair/dataset"
```
This will create subdirectory in `01_hair_deeplabv3_ffhq_pretrain_gan/runs` with logs and checkpoints.

Test:
```bash
python3 main.py test --batch-size=8 --gpus=0,1 --workers=4 --kvstore=local --input-path "../../../../experiments/ffhq-hair/dataset" runs/<run_subdirectory>
```

#### Train and test DeepLabV3+ model on real samples
Train:
```bash
cd deeplabv3plus/experiments/rgb_segmentation/00_hair_deeplabv3_ffhq_pretrain_no_gan
python3 main.py train --batch-size=4 --gpus=1 --test-batch-size=4 --workers=4 --kvstore=nccl --input-path "../../../../experiments/ffhq-hair/dataset"
```
This will create subdirectory in `01_hair_deeplabv3_ffhq_pretrain_gan/runs` with logs and checkpoints.

Test:
```bash
python3 main.py test --batch-size=8 --gpus=0,1 --workers=4 --kvstore=local --input-path "../../../../experiments/ffhq-hair/dataset" runs/<run_subdirectory> 
```

### Interactive annotator

You can annotate segmentation mask using interactive annotator.

#### Prepare config file
```bash
cp config.yml.example config.yml
```
Specify directory with experiment using `BASE_DIR` parameter, set to `experiments/<your-experiment-name>`.

#### Run annotator
```bash
python3 main.py
```

The user interface will open, which will display a generated by GAN image and a number of buttons at the bottom.
To draw a mask hold the left mouse button and drag it over the area of interest.
Use the mouse wheel to increase or decrease the brush size.
By default, the mask is filled with an ignore label. Therefore, **it is necessary to draw both the positive mask and the negative mask.**  Hold down CTRL to switch to negative mode.

OK button - Add the current image with annotataion to the dataset (`experiments/<experiment-name>/data`) and go to the next image

SKIP button - Go to the next image without adding the current image to the dataset

RETRAIN button - Train the decoder using all annotated images, model is saved in `experiments/<experiment-name>/checkpoints`

GENERATE button - Generate synthetic dataset, images with annotations are saved in `experiments/<experiment-name>/generated`
