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
    parser.add_argument('run', type=str, help='Base dir where weights are')
    parser.add_argument('data', type=str, help='Base dir of run to evaluate')
    parser.add_argument('-cs', '--colorspace', type=str, default='REC709', help='Color space of the input images')
    parser.add_argument('--color', type=str, default='gray', help='Enable/Disable color inputs')

    args = parser.parse_args()
    
    bGrayscale = (args.color == 'gray')
        
    model = QModel(args.run, grayscale)
    
    names_mat = [f for f in os.listdir(args.data) if f.endswith('.mat')]
    names_hdr = [f for f in os.listdir(args.data) if f.endswith('.hdr')]
    names_exr = [f for f in os.listdir(args.data) if f.endswith('.exr')]
    name_hdr = sorted(names_mat + names_hdr + names_exr)

    names_jpg = [f for f in os.listdir(args.data) if f.endswith('.jpg')]
    names_png = [f for f in os.listdir(args.data) if f.endswith('.png')]
    names_sdr = sorted(names_jpg + names_png)
    
    names = name_hdr + name_sdr
    
    for name in names:
        stim = load_image(os.path.join(args.data, name), grayscale = bGrayscale, log_range = True, colorspace = args.colorspace)
        p_model = float(model.predict(stim))
        print(name + " Q: " + str(round(p_model * 10000)/100))

    del model
