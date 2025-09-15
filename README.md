NoR-VDPNet
==========
NoR-VDPNet is a deep-learning based no-reference metric trained on [HDR-VDP](http://hdrvdp.sourceforge.net/wiki/).
Traditionally, HDR-VDP requires a reference image, which is not possible to have in some scenarios.

![HDR-VDP](images/hdrvdp.png?raw=true "HDR-VDP")

NoR-VDPNet is a no-reference metric, so it requires a single image in order to asses its quality. NoR-VDPNet can be trained on High Dynamic Range (HDR) images or Standard Dynamic Range (SDR) images (i.e., classic 8-bit images).

![NoR-VDPNet](images/our.png?raw=true "NoR-VDPNet")


DEPENDENCIES:
==============

Requires the PyTorch library along with Image, NumPy, SciPy, Matplotlib, glob2, pandas, and scikit-learn.

As the first step, you need to follow the [instructions for installing PyTorch](http://pytorch.org/).

To install dependencies, please use the following command: 

```bash
pip3 install numpy, scipy, matplotlib, glob2, pandas, image, scikit-learn, opencv-python. 
```

HOW TO RUN IT:
==============
To run our metric on a folder of images (i.e., JPEG, PNG, EXR, HDR, and MAT files),
you need to launch the file ```norvdpnet.py```. Some examples:

Testing SDR images for the trained distortions (see the paper):

```
python3 norvdpnet.py SDR /home/user00/images_to_be_sdr/
```

Testing HDR images after JPEG-XT compression:

```
python3 norvdpnet.py HDR_COMP /home/user00/images_to_be_hdr/
```

Testing HDR images after tone mapping operators:

```
python3 norvdpnet.py SDR_TMO /home/user00/images_to_be_sdr/
```

Testing images after inverse tone mapping operators:

```
python3 norvdpnet.py HDR_ITMO /home/user00/images_to_be_hdr/
```

WEIGHTS DOWNLOAD:
=================
Weights can be downloaded at this <a href="https://www.banterle.com/francesco/projects/norvdpnet/weights.zip">link</a>.

Note that these weights are meant to model ONLY determined distortions; please see reference to have a complete overview.

DO NOT:
=======
There are many people use NoR-VDPNet in an appropriate way:

1) Please do not use weights_nor_sdr for HDR images;

2) Please do not use weights_nor_jpg_xt for SDR images;

3) Please do not use weights_nor_tmo for HDR images; only gamma-encoded SDR images!!!

4) Please do not use weights_nor_itmo for SDR images;

5) Please do not use weights for different distortions.

DATASET PREPARATION:
====================
If you want to create your own dataset for a given distortion (note you can apply more distortions), 
the first step is to apply such distortion to a set of input original images. Then, the second step is to run
[HDR-VDP](http://hdrvdp.sourceforge.net/wiki/) on all pair of images <original, distorted> saving the Q value of HDR-VDP.
At this point, you can discard the original images keeping only the distorted ones and the Q values output by HDR-VDP.

Files need to be organized using the following folder hierarchy:

```
__dataset_folder/:
  |_______stim/
  |_______data.csv
```

JPG/PNG/EXR/HDR/MAT files for distorted images go in the ```stim/``` folder, and the Q values and links to their
respective image need to be stored in the ```data.csv``` file. Please have a look at this ```data.csv``` file example:

```
Distorted,Q
stim/img000.png,95.33
stim/img001.jpg,73.23
stim/img002.jpg,87.57
stim/img003.jpg,71.23
stim/img005.png,82.30
```

When using the .mat file format for HDR images, such images need to be stored as a variable ```image```.


TRAINING:
=========
If you want to train our metric, you need to run ```train.py``` file. This line shows how to
train the metric for a dataset in the folder ```/home/users00/data1``` for 75 epochs with batch size 16
and learning rate 1e-4:

```
python3 train.py /home/users00/data1 -e 75 --lr=1e-4 -b 32
```

Note that the folder ```data1``` needs to contain the file ```data.csv``` and the subfolder ```stim```.

In our paper, we trained SDR and HDR datasets with these paramters:

```
Learning Rate: 1e-4
Batch Size: 32
Epochs: 75
```

REFERENCE:
==========

If you use NoR-VDPNet in your work, please cite it using this reference:

```
@inproceedings{Banterle+2020,
author       = "Banterle, Francesco and Artusi, Alessandro and Moreo, Alejandro and Carrara, Fabio",
title        = "Nor-Vdpnet: A No-Reference High Dynamic Range Quality Metric Trained On Hdr-Vdp 2",
booktitle    = "IEEE International Conference on Image Processing (ICIP)",
month        = "October",
year         = "2020",
publisher    = "IEEE",
keywords     = "HDR-VDP, HDRI, HDR, SDR, LDR",
url          = "http://vcg.isti.cnr.it/Publications/2020/BAMC20"
}
```
 
