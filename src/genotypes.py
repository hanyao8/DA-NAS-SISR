from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'skip',
    'conv3x3',
    # 'conv3x3_d2',
    # 'conv3x3_d4',
    'residual',
    'dwsblock',
]

#PRIMITIVES_attn_image=[
#    'epab_spatiochannel',
#    'separable_spatial',
#    'separable_channel'
#]
#PRIMITIVES_attn_image=[
#    'epab_spatiochannel',
#    'separable_spatial_patched32',
#    'separable_channel'
#]

#PRIMITIVES_attn_video=[
#    'epab_spatiotemporal',
#    'epab_spatiochannel',
#    'separable_spatial',
#    'separable_channel',
#    'separable_temporal'
#]