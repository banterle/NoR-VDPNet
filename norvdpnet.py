#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

from QModel import QModel
import os
import argparse
from util import load_image

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Eval Q regressor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, help='SDR or HDR distortions')
    parser.add_argument('img_folder', type=str, help='Base dir of run to evaluate')
    parser.add_argument('-cs', '--colorspace', type=str, default='REC709', help='Color space of the input images')
    parser.add_argument('--color', type=str, default='gray', help='Enable/Disable color inputs')

    args = parser.parse_args()
    
    bGrayscale = (args.color == 'gray')
        
    if args.mode == 'SDR':
        model = QModel('weights/norvdpnet_sdr.pth', bGrayscale)
    elif args.mode == 'HDR':
        model = QModel('weights/norvdpnet_hdrc.pth', bGrayscale)

    names_mat = [f for f in os.listdir(args.img_folder) if f.endswith('.mat')]
    names_hdr = [f for f in os.listdir(args.img_folder) if f.endswith('.hdr')]
    names_exr = [f for f in os.listdir(args.img_folder) if f.endswith('.exr')]
    names_hdr = sorted(names_mat + names_hdr + names_exr)

    names_jpg = [f for f in os.listdir(args.img_folder) if f.lower().endswith('.jpg')]
    names_jpeg = [f for f in os.listdir(args.img_folder) if f.lower().endswith('.jpeg')]
    names_png = [f for f in os.listdir(args.img_folder) if f.lower().endswith('.png')]
    names_sdr = sorted(names_jpg + names_jpeg + names_png)
    
    names = names_hdr + names_sdr
    
    for name in names:
        stim = load_image(os.path.join(args.img_folder, name), grayscale = bGrayscale, colorspace = args.colorspace)
        p_model = float(model.predict(stim))
        print(name + " Q: " + str(round(p_model * 10000)/100))

    del model
