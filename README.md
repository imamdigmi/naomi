## Naomi
[![Build Status](https://travis-ci.org/imamdigmi/naomi.svg?branch=master)](https://travis-ci.org/imamdigmi/naomi)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

Deep Learning Pattern Recognition for Plate Number Vechile with Convolutional Neural Network Algorithm Using TensorFlow and Python

## Usage
Convert dataset meta labels xml to csv

```
python3 xml_to_csv.py
```

Convert csv dataset labels to TFRecord file format

```
python3 generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
python3 generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
```

Grap the COCO models

```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
```
or you can click [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz)

Start training the model

```
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```

Start jupyter notebook inside th directory to see the results
```
jupyter notebook --notebook-dir=.
```