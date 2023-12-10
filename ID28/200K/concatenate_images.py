#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:54:09 2023

@author: stevengebel
"""

"""Concatenate images"""

import numpy as np
import PIL
from PIL import Image

list_im =   ['200K_linecuts_-1.jpg', '200K_linecuts_0.jpg',
            '200K_linecuts_1.jpg']
# ['200K_map_-1.jpg', '200K_map_0.jpg',
            # '200K_map_1.jpg']
# ['pno1_0KL_0.5A.img.jpg', 'pno1_H0L_0.5A.img.jpg', 
#             'pno1_HK0_0.5A.img.jpg']
#['200K_linecuts_-1.jpg', '200K_linecuts_0.jpg',
#            '200K_linecuts_1.jpg']
# ['pno1_1KL_0.5A.img_linecuts.jpg', 'pno1_H1L_0.5A.img_linecuts.jpg', 
#             'pno1_HK1_0.5A.img_linecuts.jpg']
# ['200K_compilation_-1.jpg', '200K_compilation_0.jpg',
#             '200K_compilation_1.jpg']
# ['pno1_1KL_0.5A.img.jpg', 'pno1_H1L_0.5A.img.jpg', 
#             'pno1_HK1_0.5A.img.jpg']

imgs    = [ Image.open(i) for i in list_im ]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

"""Horizontal Stacking
"""
imgs_comb = Image.fromarray( imgs_comb)
imgs_comb.save( '200K_linecuts_compilation.jpg' )  #DAC_gasket_collapse_shattering

"""Vertical Stacking
"""
# imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
# imgs_comb = Image.fromarray( imgs_comb)
# imgs_comb.save( '200K_linecuts_-1.jpg' )