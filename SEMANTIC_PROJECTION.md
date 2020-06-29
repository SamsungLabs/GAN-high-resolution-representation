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