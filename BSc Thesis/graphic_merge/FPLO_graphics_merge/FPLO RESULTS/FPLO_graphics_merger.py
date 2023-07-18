import numpy as np
import PIL
from PIL import Image

list_im = ['EuGa2Al2_300K_band33.jpg', 'EuGa2Al2_300K_band34.jpg', 'EuGa2Al2_300K_band35.jpg']
    #[ '6_4.jpg']
    #['EuGa2Al2_2GPa_band33_Nesting.jpg', 'EuGa2Al2_5GPa_band33_Nesting.jpg']
    #['EuGa2Al2_5GPa_band34.jpg', 'EuGa2Al2_5GPa_band35.jpg']
    #['EuGa2Al2_APDOS_0GPa.jpg', 'EuGa2Al2_APDOS_2GPa.jpg', 'EuGa2Al2_APDOS_5GPa.jpg']
    #['EuGa2Al2_0GPa_band33.jpg', 'EuGa2Al2_2GPa_band33.jpg', 'EuGa2Al2_5GPa_band33.jpg']
#6
    #['Nesting_EuGa4_p-dep.jpg', 'APDOS_EuGa4_p-dep.jpg.jpg']
    #['EuGa4_APDOS_0GPa.jpg', 'EuGa4_APDOS_2GPa.jpg', 'EuGa4_APDOS_5GPa.jpg']
    #['Nesting_EuGa4_p-dep', 'APDOS_EuGa4_p-dep.jpg']
#5
    #['EuGa4_0GPa_band43_Nesting.jpg', 'EuGa4_2GPa_band43_Nesting.jpg', 'EuGa4_5GPa_band43_Nesting.jpg']
    #4
    #['EuGa4_5GPa_band42+43.jpg', 'EuGa4_5GPa_band44.jpg', 'EuGa4_5GPa_band45.jpg']
    #3
    #['EuGa2Al2_300K_band33.jpg', 'EuGa2Al2_300K_band34.jpg', 'EuGa2Al2_300K_band35.jpg', 'EuGa2Al2_APDOS_300K.jpg']
#2
    # ['Nesting_EuGa4_T-dep.jpg','APDOS_EuGa4_T-dep.jpg']
    #['EuGa4_300K_band43_Nesting.jpg', 'EuGa4_250K_band43_Nesting.jpg', 'EuGa4_200K_band43_Nesting.jpg']
    #['EuGa4_300K_band42+43.jpg', 'EuGa4_300K_band44.jpg', 'EuGa4_300K_band45.jpg']
    #1
    #['EuGa2Al2_300K.jpg', 'EuGa2Al2_250K.jpg', 'EuGa2Al2_200K.jpg']


    # ['EuGa2Al2_2GPa_band33_Nesting.jpg', 'EuGa2Al2_5GPa_band33_Nesting.jpg']

    #['EuGa2Al2_0GPa.jpg','EuGa2Al2_2GPa.jpg','EuGa2Al2_5GPa.jpg']

    # ['EuGa2Al2_5GPa_band33.jpg', 'EuGa2Al2_5GPa_band34.jpg',
    #       'EuGa2Al2_5GPa_band35.jpg', 'EuGa2Al2_APDOS_5GPa.jpg']

    # ['SrGa4_WHOLE.jpg', 'SrGa2Al2_WHOLE.jpg']

    # ['SrGa2Al2_band37.jpg', 'SrGa2Al2_band38.jpg']

    #['EuGa4_300K.jpg', 'EuGa4_250K.jpg', 'EuGa4_200K.jpg', 'EuGa4_0GPa.jpg',
    #       'EuGa4_2GPa.jpg', 'EuGa4_5GPa.jpg']

    #['EuGa4_5GPa_band42+43.jpg', 'EuGa4_5GPa_band43_Nesting.jpg',
    #       'EuGa4_5GPa_band44.jpg', 'EuGa4_5GPa_band45.jpg', 'EuGa4_APDOS_5GPa.jpg']
imgs    = [ Image.open(i) for i in list_im ]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

# save that beautiful picture
imgs_comb = Image.fromarray( imgs_comb)
imgs_comb.save( 'EuGa2Al2_T-dep2.jpg' )

"""for a vertical stacking it is simple: use vstack
"""
#imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
#imgs_comb = Image.fromarray( imgs_comb)
#imgs_comb.save( '6.2v2.jpg' )