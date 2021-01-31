# TensorFlow2-UNet

This is repository implementing U-Net (https://arxiv.org/abs/1505.04597) using TensorFlow 2.

Just repository to memo what I've studied about TensorFlow.

This repository includes:
* A ipython notebook file that make image dataset to tf_records file
* A python file that implement U-Net
* A ipython notebook file that load data from tf_record, augment and train U-Net

# Dataset

I used Penn-Fudan dataset (https://www.cis.upenn.edu/~jshi/ped_html/).
Maybe this dataset is not proper dataset for U-net.
However, I just wanted to write down what I've studied.

```
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip .
unzip PennFudanPed.zip
```

# Docker Settings

1. Build docker image using docker_settings/Dockerfile
2. Change image name, ports, volume paths, environments, shm_size as you want
3. Run docker compose and test U-Net!
