import numpy as np
import PIL
from PIL import Image

list_im =  ['B_isoshitttt.jpg', 'DAC_V_comparison.jpg']
    #['FINAL_OVERVIEW_STAVINOHA_3.jpg', 'FINAL_OVERVIEW_STAVINOHA_2.jpg', 'FINAL_OVERVIEW_STAVINOHA.jpg']

#    ['FINAL_OVERVIEW_STAVINOHA_3.jpg', 'FINAL_OVERVIEW_STAVINOHA_2.jpg', 'FINAL_OVERVIEW_STAVINOHA.jpg']

# ['FINAL_OVERVIEW_STAVINOHA_3.jpg', 'FINAL_OVERVIEW_STAVINOHA_2.jpg', 'FINAL_OVERVIEW_STAVINOHA.jpg']

#['stavinoah1_euga2al2_1_Version3.jpg', 'stavinoah1_euga2al2_2_Version3.jpg', 'stavinoah1_euga2al2_3_Version3.jpg'
#             ,'stavinoah1_euga2al2_4_Version3.jpg', 'stavinoah1_euga2al2_5_Version3.jpg', 'stavinoah1_euga2al2_6_Version3.jpg']

#['DAC_EuGa4_structure_1_Version3.jpg', 'DAC_EuGa4_structure_2_Version3.jpg', 'DAC_EuGa4_structure_3_Version3.jpg'
#            ,'DAC_EuGa4_structure_4_Version3.jpg', 'DAC_EuGa4_structure_5_Version3.jpg', 'DAC_EuGa4_structure_6_Version3.jpg']

# ['stavinoah1_euga2al2_1_Version3.jpg', 'stavinoah1_euga2al2_2_Version3.jpg', 'stavinoah1_euga2al2_3_Version3.jpg'
#             ,'stavinoah1_euga2al2_4_Version3.jpg', 'stavinoah1_euga2al2_5_Version3.jpg', 'stavinoah1_euga2al2_6_Version3.jpg']
# ['stavinoah1_euga4_1_Version3.jpg', 'stavinoah1_euga4_2_Version3.jpg', 'stavinoah1_euga4_3_Version3.jpg'
#             ,'stavinoah1_euga4_4_Version3.jpg', 'stavinoah1_euga4_5_Version3.jpg', 'stavinoah1_euga4_6_Version3.jpg']

# ['image106.jpg', 'image104.jpg', 'image105.jpg'] # ['ac.jpg', 'd11(2).jpg', 'ztheta.jpg'] ['DAC_EuGa4_structure_1.jpg', 'DAC_EuGa4_structure_2.jpg', 'DAC_EuGa4_structure_3.jpg']# ['M(1)M(2).jpg', 'AlGa_planes.jpg']# ['stavinoah1_euga2al2_1.jpg', 'stavinoah1_euga2al2_2.jpg', 'stavinoah1_euga2al2_3.jpg'] #['stavinoah1_euga4_1.jpg', 'stavinoah1_euga4_2.jpg', 'stavinoah1_euga4_3.jpg']
imgs    = [ Image.open(i) for i in list_im ]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

# # save that beautiful picture
# imgs_comb = Image.fromarray( imgs_comb)
# imgs_comb.save( 'Appendix_figures.jpg' ) #stavinoha_euga2al2 #DAC_EuGa4_structure #Gasket_collapse
#
"""for a vertical stacking it is simple: use vstack
"""
imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
imgs_comb = Image.fromarray( imgs_comb)
imgs_comb.save( 'Appendix_figures.jpg' )