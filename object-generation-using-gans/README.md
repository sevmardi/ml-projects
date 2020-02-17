# Image synthesis using Adversarial Generation for Data Augmentation

### Prerequisites
- pip install -r requirements.txt 
- Tested on Linux
- Python 3.6.5
- Pytorch 0.4.1
- OpenCV 3.4.4 

## Recommended
NVIDIA GPU (12G or 24G memory) + CUDA cuDNN

## Preparing
Prepare your data before training. The format of your data should follow the file in `datasets`. 
Please note that the pedestrains selection were made manully and there is no automated process for this. 

## Training stage
Running the training script
```
bash scripts/train_unet_256.sh
```

## Testing stage
Running the testing script
```
bash scripts/test_unet_256.sh
```


## Vision
Run `python -m visdom.server` to see the training process.



## Creating your own dataset
- After selection is done, you need to resize the dataset into 512x256. 
- After resizing, you apply noise using the pixel-label wise to draw a bounding box and apply peper-and-noise on selected images.
(unfortunately I don't have automated process for this) 

```sh
# Resize source images
python tools/process.py \
  --input_dir photos/original \
  --operation resize \
  --output_dir photos/resized


# Combine resized images with blanked images
python tools/process.py \
  --input_dir photos/resized \
  --b_dir photos/blank \
  --operation combine \
  --output_dir photos/combined

```




## Acknowledgments
Heavily borrow the code from <a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix">pix2pix</a>, <a href="https://github.com/yueruchen/Pedestrian-Synthesis-GAN">Pedestrian-Synthesis-GAN</a> and 
<a href="https://github.com/NVIDIA/pix2pixHD">pix2pixHD</a>


