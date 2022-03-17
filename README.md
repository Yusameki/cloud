# cloud_nict

The implementation of ConvLSTM by Keras.

## Requirements

* python 3.x
* tensorflow-gpu >= 2.3
* pillow
* numpy
* opencv
* matplotlib

## How to use

Put the datasets in `/datasets` folder.  
e.g. /datasets/{`phase`}/{`year`}/NC_H08_{year}{month}{day}_{hour}{minute}_r14_FLDK.02701_02601_tbb_14_4326.tif  
{`phase`} = train/test  
{`year`} = 2017/2018/2019

* Use `prepare.py` to set the datasets.  
* Use `train.py` to train a model.
