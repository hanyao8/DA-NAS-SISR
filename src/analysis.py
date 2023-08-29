import torch
import os 
import numpy as np
from sparsemax import Sparsemax
from sparsestmax import sparsestmax
import torch.nn.functional as F

# sparse_type = 'sparsemax'
# load_path = '/srv/beegfs02/scratch/generative_modeling/data/wuyan/AutoSR/AutoSR_v3_fusedmax_new/search/ckpt/search/search2_nf32_sparsemax_order_1e-1_flops_1e-20'
# state = torch.load(os.path.join(load_path, 'checkpoint.pt'))
# epoch = state['epoch']
# if sparse_type == 'sparsemax':
# # save = state['save']
#     sparsemax = Sparsemax(dim=-1)
#     alpha = sparsemax(state['alpha'])
#     beta = sparsemax(state['beta'])
#     ratio = F.softmax(state['ratio'], dim=-1)
#     gamma = sparsemax(state['gamma'])
# elif sparse_type == 'sparsestmax':
#     alpha = sparsestmax(state['alpha'], rad_in=epoch/100)
#     beta = sparsestmax(state['beta'], rad_in=epoch/100)
#     ratio = F.softmax(state['ratio'], dim=-1)
#     gamma = sparsestmax(state['gamma'], rad_in=epoch/100)


def generate_path_list(num_cell, num_up, num_cell_path_search):
    root = [0 for _ in range(num_cell-num_cell_path_search)]
    paths = []
    stack = [(root, 0, num_cell-num_cell_path_search)]
    length = []
    temp_length = 0
    while stack:
        # print(stack)
        temp_path, level, layer = stack.pop()
        if layer < num_cell:
            stack.append((temp_path + [0], level, layer+1))
        if level < num_up:
            stack.append((temp_path + [1], level+1, layer))
        if level == num_up:
            temp_length += 1
        if level == num_up and layer == num_cell:
            paths.append(temp_path)
            length.append(temp_length)
            temp_length = 0
    return paths
def path_generator(num_cell, num_up, num_cell_path_search, gamma_index):
    root_cell = [i for i in range(num_cell-num_cell_path_search)]
    root = [0 for _ in range(num_cell-num_cell_path_search)]
    cell_index = num_cell-num_cell_path_search
    paths = []
    cells = []
    feature_map_index = 0
    stack = [(root, root_cell, 0, num_cell-num_cell_path_search)]
    while stack:
        # print(stack)
        temp_path, temp_cell, level, layer = stack.pop()
        feature_map_index += 1
        paths.append(temp_path)
        cells.append(temp_cell)
        if layer < num_cell:
            stack.append((temp_path + [0], temp_cell + [cell_index], level, layer+1))
            cell_index += 1
        if level < num_up:
            stack.append((temp_path + [1], temp_cell, level+1, layer))
    return paths[gamma_index], cells[gamma_index]


gamma = [0.0000, 0.0000, 0.0000, 0.0000, 0.0213, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0065, 0.0000, 0.0000, 0.0000, 0.0225, 0.0228, 0.0000, 0.0000,
        0.0176, 0.0000, 0.0081, 0.0000, 1.1543, 0.0000, 0.0000, 0.0111, 0.0111,
        0.0000, 0.0600, 0.0244, 0.0526, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.4835, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0167, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0585, 0.0000, 0.0206, 0.0000, 0.0086, 0.0000, 0.0000,
        0.0000]
gamma_index = np.argmax(gamma)
# gamma_index = 1
print('gamma index:', gamma_index)
path, cell = path_generator(num_cell=5, num_up=2, num_cell_path_search=4, gamma_index=gamma_index)
print(path)
print(cell)
# print('epoch:', epoch)
# print('path weight:', gamma)
# # print('gamma index:', gamma_index)
# print('path:',path)
# # print('cell:', cell)
# op_param = alpha[cell]
# op_list = alpha.argmax(-1)[cell]
# # quantize_list = beta.argmax(-1)[cell]
# ratio_param = ratio[cell]
# ratio_list = ratio.argmax(-1)[cell]
# print('op para:', op_param)
# print('op list:', op_list)
# # print('quantize list:', quantize_list)
# # print('ratio para,:', ratio_param)
# print('ratio list:', ratio_list)

# # for path, weight in zip(paths, gamma):
# #     # print(path, ':', [round(i,4) for i in weight.cpu().detach().numpy()], round(sum(weight.cpu().detach().numpy()),4))
# #     print ('{:<10}{:<55}{:<10}'.format(str(path), str([round(i,4) for i in weight.cpu().detach().numpy()]), str(round(sum(weight.cpu().detach().numpy()),4))))


