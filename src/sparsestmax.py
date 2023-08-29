from __future__ import division

import math
from math import sqrt
import torch
from torch.autograd import Variable
from sparsemax import Sparsemax
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#def rad_heuristic(K,sp=0.5,schedule=0):
#   print('K=',K)
#    u = torch.ones(K)/K
#    a = torch.ones(K)/(K-1)
#    a[-1] = 0
#    b = torch.zeros(K)
#    b[-1] = 1

#    r_min = torch.norm(u-a)
#    r_max = torch.norm(u-b)

#    #print('r_min: ',r_min)
#    #print('r_max: ',r_max)
#    #r = ((r_max+r_min)/2).item()
#    if schedule==0:
#        r = (sp*r_max + (1-sp)*r_min).item()
#    elif schedule==1:
#        r = (sp*r_max).item()

#    return r


def sparsemax(v, z=1):
    v_sorted, _ = torch.sort(v, dim=0, descending=True)
    cssv = torch.cumsum(v_sorted, dim=0) - z
    ind = torch.arange(1, 1 + len(v)).float().to(v.device)
    print(v_sorted)
    print(cssv)
    print(ind)
    cond = v_sorted - cssv / ind > 0
    rho = ind.masked_select(cond)[-1]
    tau = cssv.masked_select(cond)[-1] / rho
    w = torch.clamp(v - tau, min=0)
    return w / z


# def sparsestmax(v, rad_in=0, u_in=None):
#     sparsemax = Sparsemax(dim=-1)
#     w = sparsemax(v)
#     print('sparsemax result:', w)
#     if max(w) - min(w) == 1:
#         return w
#     ind = torch.tensor(w > 0).float()
#     u = ind / torch.sum(ind)
#     if u_in is None:
#         rad = rad_in
#     else:
#         rad = sqrt(rad_in ** 2 - torch.sum((u - u_in) ** 2))
#     distance = torch.norm(w - u)
#     if distance >= rad:
#         return w
#     p = rad * (w - u) / distance + u
#     if min(p) < 0:
#         return sparsestmax(p, rad, u)
#     return p.clamp_(min=0, max=1)


def sparsestmax(v, rad_in=0, u_in=None):
    sparsemax = Sparsemax(dim=-1)
    w = sparsemax(v)
    dim = w.dim()
    if dim == 1:
        if max(w) - min(w) == 1:
            return w
        ind = torch.tensor(w > 0).float()
        u = ind / torch.sum(ind)
        if u_in is None:
            rad = rad_in
        else:
            rad = sqrt(max(rad_in ** 2 - torch.sum((u - u_in) ** 2), 0))

        distance = torch.norm(w - u)
        if distance >= rad:
            return w
        p = rad * (w - u) / (distance+1e-20) + u
        if min(p) < 0:
            return sparsestmax(p, rad, u)
        return p.clamp_(min=0, max=1)

    w_shape_original = w.size()
    if dim>2: #if dim  == 3:
        w = torch.reshape(w, (-1, w.size()[-1])) #reshape to (a*b,c)

    #index = torch.tensor((torch.max(w, dim=-1)[0]-torch.min(w, dim=-1)[0]) == 1).int().tolist()
    index = torch.tensor((torch.max(w, dim=-1)[0]-torch.min(w, dim=-1)[0]) == 1, dtype=torch.int32)
    ind = torch.tensor(w>0).float()
    u = ind / torch.sum(ind, dim=-1)[:, None]
    if u_in is None:
        rad = rad_in
    else:
        rad = torch.sqrt(max(rad_in ** 2 - torch.sum((u - u_in) ** 2, 0)))
        # print('rad:', rad.size())
        # rad = rad[:, None]
        # print('rad:', rad.size())
    distance = torch.norm(w-u, dim=-1)

    # index += torch.where(distance >= rad)[0].tolist()
    # print('rad:',rad.size())
    # print('distance:',distance.size())
    # print('add index:', torch.where(torch.gt(distance[:, None], rad).int()==1)[0])
    index += torch.tensor(distance >= rad, dtype=torch.int32)
    #index = [a+b for a,b in zip(index,torch.tensor(distance >= rad).int().tolist())]
    # index += torch.tensor(torch.gt(distance[:, None], rad).int()==1)[0].int().tolist()

    p = rad * (w - u) / (distance[:, None]+1e-20) + u
    index_return = torch.tensor(torch.min(p, dim=-1)[0] < 0).long()

    for i,k in enumerate(index_return):
        if k==1:
            p[i] = sparsestmax(p[i, :], rad, u[i])

    for i,k in enumerate(index):
        if k==1:
            p[i] = w[i]

    if dim>2:
        return torch.reshape(p.clamp_(min=0, max=1), w_shape_original)
    #else:
    #    return p.clamp_(min=0, max=1)


# def sparsestmax(v, rad_in=0, u_in=None):
#     sparsemax = Sparsemax(dim=-1)
#     w = sparsemax(v)
#     # print('sparsemax result:', w)
#     if w.dim() == 1:
#         if max(w) - min(w) == 1:
#             return w
#         ind = torch.tensor(w > 0).float()
#         u = ind / torch.sum(ind)
#         if u_in is None:
#             rad = rad_in
#         else:
#             # print('u_in:', u_in)
#             # temp = rad_in ** 2 - torch.sum((u - u_in) ** 2)
#             # if temp > 0:
#             rad = sqrt(rad_in ** 2 - torch.sum((u - u_in) ** 2))
#             # else:
#                 # rad = 0
#         distance = torch.norm(w - u)
#         if distance >= rad:
#             return w
#         p = rad * (w - u) / (distance+1e-20) + u
#         if min(p) < 0:
#             return sparsestmax(p, rad, u)
#         return p.clamp_(min=0, max=1)
#     else:
#         a, b, c = w.size()
#         w = torch.reshape(w, (-1, w.size()[-1]))
#         for i in range(w.size()[0]):
#             if max(w[i]) - min(w[i]) == 1:
#                 continue
#             ind = torch.tensor(w[i] > 0).float()
#             u = ind / torch.sum(ind)
#             # print('u=', u)
#             if u_in is None:
#                 rad = rad_in
#             else:
#                 rad = sqrt(rad_in ** 2 - torch.sum((u - u_in) ** 2))
#             distance = torch.norm(w[i] - u)
#             if distance >= rad:
#                 continue
#             p = rad * (w[i] - u) / (distance+1e-20) + u
#             # print('p=', p)
#             if min(p) < 0:
#                 # print('yes')
#                 w[i] = sparsestmax(p, rad, u)
#                 continue
#             w[i] = p.clamp_(min=0, max=1)
#         return torch.reshape(w, (a, b, c))

# sparse = Sparsemax(dim=-1)
# v = Variable(1e-3*torch.randn(21, 4, 5).cuda(), requires_grad=True)
# # start = time.time()
# # print('sparse:', sparse(v))
# # print('sparse time:', time.time()-start)
# # print('sparsest:',sparsestmax(v, rad_in=0))
# # print(time.time()-start)
# output = sparsestmax(v)
# output = torch.norm(torch.norm(torch.norm(output, p=2), p=2), p=2)
# output.backward()
# print(v.grad)