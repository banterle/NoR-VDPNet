NoR-VDPNet
==========
NoR-VDPNet is a deep-learning based metric trained on [HDR-VDP 2.2](http://hdrvdp.sourceforge.net/wiki/).
NoR-VDPNet is a no-reference metric, so it requires a single image in order to asses its quality. 
This can be trained on High Dynamic Range (HDR) images or Standard Dynamic Range (SDR) images (i.e., classic 8-bit images).
Note that SDR images needs to be scene referred; i.e., you need to apply a monitor model and to obtain

DEPENDENCIES:
==============

Requires the PyTorch library along with Image, NumPy, SciPy, Matplotlib, glob2, pandas, and scikit-learn.

As the first step, you need to follow the [instructions for installing PyTorch](http://pytorch.org/).

To install dependencies, please use the following command: 
```bash
pip3 install numpy, scipy, matplotlib, glob2, pandas, image, scikit-learn. 
```

USAGE:
======

DATASET PREPARATION:
====================
If you want to create your own dataset, as a first step you need to run [HDR-VDP](http://hdrvdp.sourceforge.net/wiki/)
on all the images of your dataset using a pair of images <original, distorted> saving the Q value of HDR-VDP.
Then, you need to keep as a dataset only the distorted images and their associated Q values.

You need to organize your files 
using the following folder hierarchy:
```
__dataset_folder/:
  |_______stim/
  |_______data.csv
```

In the ```stim/``` folder, you need to place JPG/PNG/MAT files. In the ```data.csv``` file
you need to list the files from ```stim/``` that you want to use in the training and
the HDR-VDP 2.2's Quality value. See this ```data.csv``` file example:
```
Distorted,Q
img000.png,95.33
img001.jpg,73.23
img002.jpg,87.57
img003.jpg,71.23
img005.png,82.30
```

For loading HDR images, we use MAT files. Note that an image in this format need to be stored
as a variable ```image```.


TRAINING:
=========
Regarding SDR images with different distortions, we suggest these training parameters:
```
Learning Rate: 1e-4
Batch Size: 16
Epochs: 1024
```

Regarding HDR images that have been compressed using JPEG-like algorithms, we suggest these training parameters:
```
Learning Rate: 1e-4
Batch Size: 16
Epochs: 1024
```

REFERENCE:
==========

If you use NoR-VDPNet in your work, please cite it using this reference:

@inproceedings{Banterle+2020,

author       = "Banterle, Francesco and Artusi, Alessandro and Moreo, Alejandro and Carrara, Fabio",

booktitle    = "IEEE International Conference on Image Processing (ICIP)",

month        = "October",

year         = "2020",

publisher    = "IEEE",

keywords     = "HDR-VDP, HDRI, HDR, SDR, LDR",

url          = "http://vcg.isti.cnr.it/Publications/2020/BAMC20"

}
 
