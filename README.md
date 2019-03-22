# MaskRCNN with Knowledge Distillation (TBD)
Mask RCNN with pytorch backend for pedestrian detection on Caltech dataset.

## Features
- ROI Align layer
- Better default configurations for ResNet18 training
- Support CityPersons

## Requirements
1. [pytorch](http://pytorch.org/)
```bash
pip install pytorch==0.4.1
pip install torchvision
```
2. [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) for visualization
```bash
pip install tensorboardX==1.2
pip install tensorflow
```
## How to run the code
1. Compiling libs
```bash
cd ./libs
make
```
2. Download Caltech dateset

    [Check the Caltech dataset's homepage.](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)

3. Add soft link to the /data folder
```bash
cd /data
ln -s /home/data/Caltech caltech
```

4. Train the teacher model at first
```bash
sh exper/caltech/maskrcnn/train_test_resnet50_softmax.sh
```

5. Then, train the student model
```bash
sh /exper/caltech/maskrcnn/train_test_resnet18_softmax.sh
```
6. You may need to modify the .yaml file for your own training settings.
