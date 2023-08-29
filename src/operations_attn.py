import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import utils.profiling
import util_gan.psnr


class SimpleNonLocal_Block_Video_NAS(nn.Module):
    def __init__(self, nf, mode):
        super(SimpleNonLocal_Block_Video_NAS, self).__init__()
        self.convx1 = nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.convx2 = nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.convx4 = nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.mode = mode
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        assert mode in ['spatial', 'channel', 'temporal'], 'Mode from NL Block not recognized.'

    def forward(self, x1):
        if self.mode == 'channel':
            x = x1.clone()
            xA = torch.sigmoid(self.convx1(x))
            xB = self.convx2(x)*xA
            x = self.convx4(xB)
        elif self.mode == 'temporal':
            x = x1.permute(0, 2, 1, 3, 4).contiguous()  # BTCHW to BCTHW
            intm=self.convx1(x)
            xA = self.sigmoid(intm)
            xB = self.convx2(x)*xA
            xB = self.convx4(xB)
            x = xB.permute(0, 2, 1, 3, 4).contiguous()  # BCTHW to BTCHW
        return x + x1


class Separable_NL_Spatial(nn.Module):
    def __init__(self,nf,num_frames):
        super(Separable_NL_Spatial, self).__init__()
        #self.convA1 = nn.Conv2d(num_frames*nf,num_frames*nf, kernel_size=1, padding=0, stride=1, bias=True)
        self.convA1T = nn.Conv3d(num_frames, num_frames, 1, 1, 0, bias=True)
        self.convA1C = nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        #self.convB1 = nn.Conv2d(num_frames*nf,num_frames*nf,kernel_size=1, padding=0, stride=1,bias=True)
        self.convB1T = nn.Conv3d(num_frames, num_frames, 1, 1, 0, bias=True)
        self.convB1C = nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.convD1T = nn.Conv3d(num_frames, num_frames, 1, 1, 0, bias=True)
        self.convD1C = nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        #self.convD1 = nn.Conv2d(num_frames*nf,num_frames*nf,kernel_size=1, padding=0, stride=1,bias=True)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        #self.scale=nn.Parameter(torch.ones([1,1],dtype=torch.float)*torch.sqrt(torch.Tensor([np.sqrt(7*64)]).float()), requires_grad=True) 
        self.scale=nn.Parameter(torch.ones([1,1],dtype=torch.float)*torch.sqrt(torch.Tensor([np.sqrt(num_frames*64)]).float()), requires_grad=True) 
        #self.scale= torch.nn.Parameter(w)

        self.debug=False
        #self.debug=True

    def forward(self,x):
        #print('\noperations_attn spatial_attn line 59')
        #profiling_message = utils.profiling.memory_usage()
        B,N,C,H,W=x.shape
        #print('operations_attn line 64 x.shape:',x.shape)
        #print(x.shape)
        #x = x.view(B, N*C, H, W)
        x_p=x.permute(0,2,1,3,4)
        #print('\noperations_attn spatial_attn line 65')
        #profiling_message = utils.profiling.memory_usage()
        A1T = self.convA1T(x).view(B, N*C, H*W).permute(0, 2, 1)  # [B, H*W, N*C]
        #print(A1T.shape)
        #print('\noperations_attn spatial_attn line 69')
        #profiling_message = utils.profiling.memory_usage()
        A1C= self.convA1C(x_p).permute(0,2,1,3,4).contiguous().view(B, N*C, H*W).permute(0, 2, 1)  # [B, H*W, N*C]
        B1T= self.convB1T(x).view(B, N*C, H*W)
        #print(B1T.shape)
        B1C= self.convB1C(x_p).permute(0,2,1,3,4).contiguous().view(B, N*C, H*W)

        #print('\noperations_attn spatial_attn line 70')
        #profiling_message = utils.profiling.memory_usage()

        #A1=self.relu(self.convA1(F))
        #B1=self.relu(self.convB1(F))
        #A1=A1.permute(0,3,4,1,2).contiguous().view(B,H*W,T*C)
        #B1 =B1.permute(0, 3, 4, 1, 2).contiguous().view(B, H * W, T * C).permute(0,2,1)

        M1T=torch.matmul(A1T,B1T)

        #print('\noperations_attn spatial_attn line 75')
        #profiling_message = utils.profiling.memory_usage()

        M1C=torch.matmul(A1C, B1C)

        #print('\noperations_attn spatial_attn line 80')
        #profiling_message = utils.profiling.memory_usage()

        #M1_actT=self.softmax(M1T/self.scale) #self.relu(M1)*act_weights[0]+self.sigmoid(M1)*act_weights[1]+self.softmax(M1)*act_weights[2]
        #M1_actC=self.softmax(M1C/self.scale) #self.relu(M1)*act_weights[0]+self.sigmoid(M1)*act_weights[1]+self.softmax(M1)*act_weights[2]
        M1T=self.softmax(M1T/self.scale)
        M1C=self.softmax(M1C/self.scale)

        D1T=self.convD1T(x).view(B, N*C, H*W).permute(0, 2, 1)
        D1C=self.convD1C(x_p).permute(0,2,1,3,4).contiguous().view(B, N*C, H*W).permute(0, 2, 1)

        #E1T= torch.matmul(M1_actT, D1T).permute(0, 2, 1).contiguous().view(B, N, C, H, W)
        #E1C= torch.matmul(M1_actC, D1C).permute(0, 2, 1).contiguous().view(B, N, C, H, W)

        E1T= torch.matmul(M1T, D1T).permute(0, 2, 1).contiguous().view(B, N, C, H, W)
        E1C= torch.matmul(M1C, D1C).permute(0, 2, 1).contiguous().view(B, N, C, H, W)
        
        if self.debug:
            print("spatial forward")
            print(x.shape)

            print("C",C)

            print("A1T %s"%str(A1T.shape))
            print("A1C %s"%str(A1C.shape))

            print("B1T %s"%str(B1T.shape))
            print("B1C %s"%str(B1C.shape))

            print("D1T %s"%str(D1T.shape))
            print("D1C %s"%str(D1C.shape))

            print("M1T %s"%str(M1T.shape))
            print("M1C %s"%str(M1C.shape))

            print("E1T %s"%str(E1T.shape))
            print("E1C %s"%str(E1C.shape))

        #print('\noperations_attn spatial_attn line 112')
        #profiling_message = utils.profiling.memory_usage()
            
        return E1T+E1C+x.view(B, N, C, H, W)


class Vit_Spatial_Map(nn.Module): #attention layer from vit, global attn
    def __init__(self,nf,num_frames,image_size,patch_size):
        super(Vit_Spatial_Map, self).__init__()
        #self.patch_size = e.g. 16, 32
        assert num_frames==1 #temporary
        self.image_size = image_size #32
        self.patch_size = patch_size #4

        self.heads = max(1,self.patch_size//2)
        self.num_patches = (self.image_size//self.patch_size)**2 #64
        self.patch_dim = nf*self.patch_size*self.patch_size #1024
        self.dim = self.patch_dim//16 #64

        self.linear_embed = nn.Linear(self.patch_dim,self.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))
        self.linear_q = nn.Linear(self.dim, self.dim, bias = False)
        self.linear_k = nn.Linear(self.num_patches, self.num_patches, bias = False)
        self.linear_v = nn.Linear(self.dim, self.dim, bias = False)
        self.avgpool = nn.AvgPool1d(self.dim//self.heads)
        self.attend = nn.Softmax(dim = -1)
        self.sigmoid = nn.Sigmoid()
        self.linear_out = nn.Linear(self.dim,self.patch_dim) #check performance

        self.attn_visual = False

    def forward(self,x):
        B,N,C,H,W=x.shape
        print("Vit_Spatial_Map, %s"%(str(x.shape)))
        assert (H==W and H==self.image_size)
        P = self.patch_size
        Np = (H//P) * (W//P)
        assert(Np==self.num_patches)
        assert(C*P*P==self.patch_dim)

        x = x.view(B*N,C,H,W)
        x = F.unfold(x,kernel_size=P,stride=P) #output B*N,C*P*P,Np
        x = x.permute(0,2,1)

        x = self.linear_embed(x)
        x += self.pos_embedding
        q = self.linear_q(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3) #B*N,h,Np,d/h

        q = self.avgpool(q.contiguous().view(B*N*self.heads,Np,self.dim//self.heads)) #(+) check shapes, use of contiguous
        q = q.view(B*N*self.heads,Np) #B*N*h,Np,1 -> B*N*h,Np
        k = self.sigmoid(self.linear_k(q))
        attn = torch.diag_embed(k).view(B*N,self.heads,Np,Np) #(Wmap) B*N,h,Np,Np

        v = self.linear_v(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3)
        out = torch.matmul(attn, v)
        out = out.permute(0,2,1,3).contiguous().view(B*N,Np,self.dim)
        out = self.linear_out(out)

        out = out.permute(0,2,1)
        out = F.fold(out,output_size=(H,W),kernel_size=(P,P),stride=P) #expect output B*N,C,H,W
        out = out.view(B,N,C,H,W)

        return out


class Vit_Channel_Map(nn.Module): #attention layer from vit, global attn
    def __init__(self,nf,num_frames,image_size):
        super(Vit_Channel_Map, self).__init__()
        #self.patch_size = e.g. 16, 32
        assert num_frames==1 #temporary
        self.image_size = image_size #32
        self.nf = nf #64

        self.heads = max(1,self.image_size//16)
        self.dim = (self.image_size**2)//16

        self.linear_embed = nn.Linear(self.image_size**2,self.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.nf, self.dim))
        self.linear_q = nn.Linear(self.dim, self.dim, bias = False)
        self.linear_k = nn.Linear(self.nf, self.nf, bias = False)
        self.linear_v = nn.Linear(self.dim, self.dim, bias = False)
        self.avgpool = nn.AvgPool1d(self.dim//self.heads)
        self.attend = nn.Softmax(dim = -1)
        self.sigmoid = nn.Sigmoid()
        self.linear_out = nn.Linear(self.dim,self.image_size**2) #check performance

        self.attn_visual = False

    def forward(self,x):
        B,N,C,H,W=x.shape
        print("Vit_Channel_Map, %s"%(str(x.shape)))
        assert (H==W and H==self.image_size)
        assert (C==self.nf)

        x = x.view(B*N,C,H*W)
        x = self.linear_embed(x)
        x += self.pos_embedding

        q = self.linear_q(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3) #B*N,h,C,d/h

        q = self.avgpool(q.contiguous().view(B*N*self.heads,C,self.dim//self.heads)) #(+) check shapes, use of contiguous
        q = q.view(B*N*self.heads,C) #B*N*h,C,1 -> B*N*h,C
        k = self.sigmoid(self.linear_k(q))
        attn = torch.diag_embed(k).view(B*N,self.heads,C,C) #(Wmap) B*N,h,C,C

        v = self.linear_v(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3)
        out = torch.matmul(attn, v)
        out = out.permute(0,2,1,3).contiguous().view(B*N,C,self.dim)
        out = self.linear_out(out) #B*N,C,H*W

        out = out.view(B*N,C,H,W)
        out = out.view(B,N,C,H,W)

        return out


class Vit_Spatiochannel_Map(nn.Module): #attention layer from vit, global attn
    def __init__(self,nf,num_frames,image_size,patch_size):
        super(Vit_Spatiochannel_Map, self).__init__()
        #self.patch_size = e.g. 16, 32
        assert num_frames==1 #temporary
        self.image_size = image_size #32
        self.patch_size = patch_size #16
        self.nf = nf

        self.num_patches = (self.image_size//self.patch_size)**2 #4
        #self.patch_dim = nf*(self.patch_size**2) #16k
        #self.dim = self.patch_dim//16
        self.dim = min(64,self.patch_size**2) #64
        self.heads = max(1,self.dim//32)

        self.linear_embed = nn.Linear(self.patch_size**2,self.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches*self.nf, self.dim))
        self.linear_q = nn.Linear(self.dim, self.dim, bias = False)
        self.linear_k = nn.Linear(self.num_patches*self.nf, self.num_patches*self.nf, bias = False)
        self.linear_v = nn.Linear(self.dim, self.dim, bias = False)
        self.avgpool = nn.AvgPool1d(self.dim//self.heads)
        self.attend = nn.Softmax(dim = -1)
        self.sigmoid = nn.Sigmoid()
        self.linear_out = nn.Linear(self.dim,self.patch_size**2) #check performance

        self.attn_visual = False

    def forward(self,x):
        B,N,C,H,W=x.shape
        print("Vit_Spatiochannel_Map, %s"%(str(x.shape)))
        assert (H==W and H==self.image_size)
        P = self.patch_size
        Np = (H//P) * (W//P)
        assert(Np==self.num_patches)
        #assert(C*P*P==self.patch_dim)

        x = x.view(B*N,C,H,W)
        x = F.unfold(x,kernel_size=P,stride=P) #output B*N,C*P*P,Np
        x = x.permute(0,2,1) #output B*N,Np,C*P*P
        x = x.view(B*N,Np,C,P*P)
        x = x.contiguous().view(B*N,Np*C,P*P) #(B*N,256,256) #(+) check shapes, use of contiguous

        x = self.linear_embed(x) #(B*N,256,64)
        x += self.pos_embedding
        q = self.linear_q(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3) #B*N,h,Np*C,d/h

        q = self.avgpool(q.contiguous().view(B*N*self.heads,Np*C,self.dim//self.heads)) #(+) check shapes, use of contiguous
        q = q.view(B*N*self.heads,Np*C) #B*N*h,Np*C,1 -> B*N*h,Np*C
        k = self.sigmoid(self.linear_k(q))
        attn = torch.diag_embed(k).view(B*N,self.heads,Np*C,Np*C) #(Wmap) B*N,h,Np*C,Np*C

        v = self.linear_v(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3)
        out = torch.matmul(attn, v)
        out = out.permute(0,2,1,3).contiguous().view(B*N,Np*C,self.dim)
        out = self.linear_out(out) #output B*N,Np*C,P*P

        out = out.view(B*N,Np,C,P*P)
        out = out.view(B*N,Np,C*P*P)
        out = out.permute(0,2,1)
        out = F.fold(out,output_size=(H,W),kernel_size=(P,P),stride=P) #expect output B*N,C,H,W
        out = out.view(B,N,C,H,W)

        return out


class Vit_Spatial(nn.Module): #attention layer from vit, global attn
    def __init__(self,nf,num_frames,image_size,patch_size):
        super(Vit_Spatial, self).__init__()
        #self.patch_size = e.g. 16, 32
        assert num_frames==1 #temporary
        self.image_size = image_size #lowres image size (120,120)
        self.patch_size = patch_size

        self.heads = max(1,self.patch_size//2)
        self.num_patches = (self.image_size//self.patch_size)**2
        self.patch_dim = nf*self.patch_size*self.patch_size
        #self.dim = 1024
        self.dim = self.patch_dim//16
        #self.dim = self.patch_dim//32

        self.linear_embed = nn.Linear(self.patch_dim,self.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))
        self.linear_q = nn.Linear(self.dim, self.dim, bias = False)
        self.linear_k = nn.Linear(self.dim, self.dim, bias = False)
        self.linear_v = nn.Linear(self.dim, self.dim, bias = False)
        self.attend = nn.Softmax(dim = -1)
        self.linear_out = nn.Linear(self.dim,self.patch_dim) #check performance

        self.attn_visual = False

    #def fold_out(self,x,out_shape):
    #    B,N,C,H,W=out_shape
    #    P = self.patch_size
    #    Np = (H//P) * (W//P)
    #    out = torch.zeros((B*N,C,H,W))
    #    p=0
    #    for i in range((H//P)):
    #        for j in range((W//P)):
    #            for bn in range(B*N):
    #                out[bn,:,i*P:(i+1)*P,j*P:(j+1)*P] = x[bn,p].view(C,P,P)
    #            p+=1
    #    #print(p)
    #    return(x.view(B,N,C,H,W))

    def forward(self,x):
        B,N,C,H,W=x.shape
        print("Vit_Spatial, %s"%(str(x.shape)))
        #print(x.shape)
        #print(self.image_size)
        assert (H==W and H==self.image_size)
        P = self.patch_size
        Np = (H//P) * (W//P)
        assert(Np==self.num_patches)
        assert(C*P*P==self.patch_dim)

        #B,N,C,H,W to B*N,Np,C*P*P
        #x = x.view(B*N,C,H,W).unfold(2,P,P).unfold(3,P,P).permute(0,2,3,1,4,5).contiguous().view(B*N,Np,C*P*P)
        #print(x.shape)
        x = x.view(B*N,C,H,W)
        x = F.unfold(x,kernel_size=P,stride=P) #output B*N,C*P*P,Np
        x = x.permute(0,2,1)

        x = self.linear_embed(x)
        x += self.pos_embedding
        q = self.linear_q(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3)
        k = self.linear_k(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3)
        v = self.linear_v(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3)

        dots = torch.matmul(q, k.transpose(-1, -2)) * ((self.dim/self.heads) ** -0.5)
        #fyi dots is the matrix with O((H/P)^4) memory. Large H/P causes memory problem for forward pass.
        attn = self.attend(dots)
        #print(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0,2,1,3).contiguous().view(B*N,Np,self.dim)
        out = self.linear_out(out)

        if self.attn_visual:
            with open('/scratch_net/kringel/hchoong/github/attention-nas/TrilevelNAS_attbranch/output/output_2022_03_02_04_07_01/attn_visual6.txt',
             'a') as f:
                #np.save(f, str(np.array([1, 2])))
                #f.write(str(np.array([1, 2])))

                #f.write(str(attn[0,0,34,:]))
                #f.write(str(attn[0,0,:,34]))
                #f.write(str(attn[0,1,34,:]))
                #f.write(str(attn[0,1,:,34]))

                idx = 34
                f.write("idx="+str(idx))
                #result = (attn[0,0,idx,:]+attn[0,0,:,idx]+attn[0,1,idx,:]+attn[0,1,:,idx])/4
                result = (dots[0,0,idx,:]+attn[0,1,idx,:])/2
                #result = dots[0,0,idx,:]
                f.write(str(result))
                f.write("\n")

                idx = 53
                f.write("idx="+str(idx))
                #result = (attn[0,0,idx,:]+attn[0,0,:,idx]+attn[0,1,idx,:]+attn[0,1,:,idx])/4
                result = (dots[0,0,idx,:]+attn[0,1,idx,:])/2
                #result = dots[0,0,idx,:]
                f.write(str(result))
                f.write("\n")

                idx = 15
                f.write("idx="+str(idx))
                #result = (attn[0,0,idx,:]+attn[0,0,:,idx]+attn[0,1,idx,:]+attn[0,1,:,idx])/4
                result = (dots[0,0,idx,:]+attn[0,1,idx,:])/2
                #result = dots[0,0,idx,:]
                f.write(str(result))
                f.write("\n")
                f.close()

        #print('operations_attn line 195')
        #print('out.shape:',out.shape) #expect B*N,Np,C*P*P
        #B*N,Np,C*P*P to B,N,C,H,W
        #out = self.fold_out(out,out_shape=(B,N,C,H,W)) #look to improve with better vectorization
        out = out.permute(0,2,1)
        out = F.fold(out,output_size=(H,W),kernel_size=(P,P),stride=P) #expect output B*N,C,H,W
        out = out.view(B,N,C,H,W)

        #print('operations_attn line 203')
        #print('out.shape:',out.shape) #expect B*N,Np,C*P*P
        #print('operations_attn line 213 Vit_Spatial %s'%(utils.profiling.memory_usage()))

        return out


class Vit_Channel(nn.Module):
    def __init__(self,nf,num_frames,image_size):
        super(Vit_Channel, self).__init__()
        assert num_frames==1 #temporary
        self.image_size = image_size #patch=4, "window"=32x32, crop=240x240
        self.nf = nf
        
        self.heads = max(1,self.image_size//16)
        self.dim = (self.image_size**2)//16

        self.linear_embed = nn.Linear(self.image_size**2,self.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.nf, self.dim))
        self.linear_q = nn.Linear(self.dim, self.dim, bias = False)
        self.linear_k = nn.Linear(self.dim, self.dim, bias = False)
        self.linear_v = nn.Linear(self.dim, self.dim, bias = False)
        self.attend = nn.Softmax(dim = -1)
        self.linear_out = nn.Linear(self.dim,self.image_size**2) #check performance

    def forward(self,x):
        #print('Vit_Channel')
        B,N,C,H,W=x.shape
        print("Vit_Channel, %s"%(str(x.shape)))
        assert (H==W and H==self.image_size)
        assert (C==self.nf)

        x = x.view(B*N,C,H,W)
        x = x.contiguous().view(B*N,C,H*W)
        x = self.linear_embed(x)
        x += self.pos_embedding
        q = self.linear_q(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3)
        k = self.linear_k(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3)
        v = self.linear_v(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3)

        dots = torch.matmul(q, k.transpose(-1, -2)) * ((self.dim/self.heads) ** -0.5)
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.permute(0,2,1,3).contiguous().view(B*N,C,self.dim)
        out = self.linear_out(out) #B*N,C,H*W
        
        out = out.view(B*N,C,H,W)
        out = out.view(B,N,C,H,W)

        return out


class Vit_Spatiochannel(nn.Module):
    def __init__(self,nf,num_frames,image_size,patch_size):
        super(Vit_Spatiochannel, self).__init__()
        #self.patch_size = e.g. 16, 32
        assert num_frames==1 #temporary
        self.image_size = image_size #32
        self.patch_size = patch_size #16
        self.nf = nf

        self.num_patches = (self.image_size//self.patch_size)**2 #4
        #self.patch_dim = nf*(self.patch_size**2) 
        #self.dim = self.patch_dim//16
        self.dim = min(64,self.patch_size**2) #64
        self.heads = max(1,self.dim//32)

        self.linear_embed = nn.Linear(self.patch_size**2,self.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches*self.nf, self.dim))
        self.linear_q = nn.Linear(self.dim, self.dim, bias = False)
        self.linear_k = nn.Linear(self.dim, self.dim, bias = False)
        self.linear_v = nn.Linear(self.dim, self.dim, bias = False)
        self.attend = nn.Softmax(dim = -1)
        self.linear_out = nn.Linear(self.dim,self.patch_size**2) #check performance

        self.attn_visual = False

    def forward(self,x):
        #print('Vit_SC')
        B,N,C,H,W=x.shape
        print("Vit_Spatiochannel, %s"%(str(x.shape)))
        assert (H==W and H==self.image_size)
        P = self.patch_size
        Np = (H//P) * (W//P) #4
        assert(Np==self.num_patches)
        #assert(C*P*P==self.patch_dim)

        x = x.view(B*N,C,H,W)
        x = F.unfold(x,kernel_size=P,stride=P) #output B*N,C*P*P,Np
        x = x.permute(0,2,1) #output B*N,Np,C*P*P
        x = x.view(B*N,Np,C,P*P)
        x = x.contiguous().view(B*N,Np*C,P*P) #(B*N,256,256) #(+) check shapes, use of contiguous

        x = self.linear_embed(x)
        x += self.pos_embedding
        q = self.linear_q(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3)
        k = self.linear_k(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3)
        v = self.linear_v(x).view(B*N,-1,self.heads,self.dim//self.heads).permute(0,2,1,3)

        dots = torch.matmul(q, k.transpose(-1, -2)) * ((self.dim/self.heads) ** -0.5)
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.permute(0,2,1,3).contiguous().view(B*N,Np*C,self.dim)
        out = self.linear_out(out) #output B*N,Np*C,P*P

        out = out.view(B*N,Np,C,P*P)
        out = out.view(B*N,Np,C*P*P)
        out = out.permute(0,2,1)
        out = F.fold(out,output_size=(H,W),kernel_size=(P,P),stride=P) #expect output B*N,C,H,W
        out = out.view(B,N,C,H,W)

        return out


class Spatial_Attn_Wrapper(nn.Module):
    def __init__(self,nf,num_frames,attn_type):
        super(Spatial_Attn_Wrapper, self).__init__()
        #self.patch_size = 32 #should be multiple of 32 and/or 16
        #for vit_attn: expect patch_size=256
        #self.attn_spatial = attn_spatial(nf=nf,num_frames=num_frames)
        #Caution: Beware the meaning of image_size and patch_size for ViT attention
        
        if attn_type == 'separable_spatial_patched32':
            self.op_img_size = 32
            self._op = Separable_NL_Spatial(nf=nf,num_frames=num_frames)
        elif attn_type == 'separable_spatial_patched16':
            self.op_img_size = 16
            self._op = Separable_NL_Spatial(nf=nf,num_frames=num_frames)
        elif attn_type == 'vit_spatial_patched32':
            self.op_img_size = 96
            self._op = Vit_Spatial(nf=nf,num_frames=num_frames,image_size=self.op_img_size,patch_size=32)
        elif attn_type == 'vit_spatial_patched16':
            self.op_img_size = 96
            self._op = Vit_Spatial(nf=nf,num_frames=num_frames,image_size=self.op_img_size,patch_size=16)
        elif attn_type == 'vit_spatial_patched16_image96':
            self.op_img_size = 96
            self._op = Vit_Spatial(nf=nf,num_frames=num_frames,image_size=self.op_img_size,patch_size=16)
        elif attn_type == 'vit_spatial_patched16_image32':
            self.op_img_size = 32
            self._op = Vit_Spatial(nf=nf,num_frames=num_frames,image_size=self.op_img_size,patch_size=16)
        elif attn_type == 'vit_spatial_patched8_image32':
            self.op_img_size = 32
            self._op = Vit_Spatial(nf=nf,num_frames=num_frames,image_size=self.op_img_size,patch_size=8)
        elif attn_type == 'vit_spatial_patched4_image32':
            self.op_img_size = 32
            self._op = Vit_Spatial(nf=nf,num_frames=num_frames,image_size=self.op_img_size,patch_size=4)
        elif attn_type == 'vit_channel_image32':
            self.op_img_size = 32
            self._op = Vit_Channel(nf=nf,num_frames=num_frames,image_size=self.op_img_size)
        elif attn_type == 'vit_spatiochannel_patched4_image32':
            self.op_img_size = 32
            self._op = Vit_Spatiochannel(nf=nf,num_frames=num_frames,image_size=self.op_img_size,patch_size=4)
        elif attn_type == 'vit_spatiochannel_patched16_image32':
            self.op_img_size = 32
            self._op = Vit_Spatiochannel(nf=nf,num_frames=num_frames,image_size=self.op_img_size,patch_size=16)
        elif attn_type == 'vit_spatial_map_patched4_image32':
            self.op_img_size = 32
            self._op = Vit_Spatial_Map(nf=nf,num_frames=num_frames,image_size=self.op_img_size,patch_size=4)
        elif attn_type == 'vit_channel_map_image32':
            self.op_img_size = 32
            self._op = Vit_Channel_Map(nf=nf,num_frames=num_frames,image_size=self.op_img_size)
        elif attn_type == 'vit_spatiochannel_map_patched16_image32':
            self.op_img_size = 32
            self._op = Vit_Spatiochannel_Map(nf=nf,num_frames=num_frames,image_size=self.op_img_size,patch_size=16)
        else:
            raise(Exception)

        #self.conv_fusion = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)
        self.conv_fusion = nn.Conv2d(2*nf, nf, 1, 1, 0, bias=True)

    def unfold_A(self,x_attn_A,sizes): #B,N,C,Nph*P,Npw*P
        B,N,C,H,W,P,Nph,Npw = sizes
        x_attn_A = x_attn_A.view(B*N,C,Nph*P,Npw*P) #output B*N,C,Nph*P,Npw*P
        x_attn_A = F.unfold(x_attn_A,kernel_size=P,stride=P) #output B*N,C*P*P,Nph*Npw
        #Nph*Npw total num of patches. For ViT this should be 1.
        x_attn_A = x_attn_A.permute(0,2,1) #output B*N,Nph*Npw,C*P*P
        x_attn_A = x_attn_A.view(B,N,Nph*Npw,C,P,P)
        x_attn_A = x_attn_A.permute(0,2,1,3,4,5) #B,Nph*Npw,N,C,P,P
        #print('line 242 x_attn_A.shape:',x_attn_A.shape)
        x_attn_A = x_attn_A.contiguous().view(B*Nph*Npw,N,C,P,P)
        return x_attn_A

    def fold_A(self,x_attn_A,sizes): #input: B*Nph*Npw,N,C,P,P
        B,N,C,H,W,P,Nph,Npw = sizes
        x_attn_A = x_attn_A.view(B,Nph*Npw,N,C*P*P)
        x_attn_A = x_attn_A.permute(0,2,3,1) #B,N,C*P*P,Nph*Npw
        x_attn_A = x_attn_A.view(B*N,C*P*P,Nph*Npw)
        x_attn_A = F.fold(x_attn_A,output_size=(Nph*P,Npw*P),kernel_size=(P,P),stride=P) #output B*N,C,Nph*P,Npw*P
        x_attn_A = x_attn_A.view(B,N,C,Nph*P,Npw*P)
        return x_attn_A

    def unfold_B(self,x_attn_B,sizes): #input B,N,C,P,Npw*P
        B,N,C,H,W,P,Nph,Npw = sizes
        x_attn_B = x_attn_B.view(B*N,C,P,Npw*P)
        x_attn_B = F.unfold(x_attn_B,kernel_size=P,stride=P) #output B*N,C*P*P,Npw
        x_attn_B = x_attn_B.permute(0,2,1) #output B*N,Npw,C*P*P
        x_attn_B = x_attn_B.view(B,N,Npw,C,P,P)
        x_attn_B = x_attn_B.permute(0,2,1,3,4,5) #B,Npw,N,C,P,P
        #print('line 268 x_attn_B.shape:',x_attn_B.shape)
        x_attn_B = x_attn_B.contiguous().view(B*Npw,N,C,P,P)
        #x_attn_B = x_attn_B.view(B*Npw,N,C,P,P)
        return x_attn_B

    def fold_B(self,x_attn_B,sizes): #input: B*Npw,N,C,P,P
        B,N,C,H,W,P,Nph,Npw = sizes
        x_attn_B = x_attn_B.view(B,Npw,N,C*P*P)
        x_attn_B = x_attn_B.permute(0,2,3,1) #B,N,C*P*P,Npw
        x_attn_B = x_attn_B.view(B*N,C*P*P,Npw)
        x_attn_B = F.fold(x_attn_B,output_size=(P,Npw*P),kernel_size=(P,P),stride=P) #output B*N,C,P,Npw*P
        x_attn_B = x_attn_B.view(B,N,C,P,Npw*P)
        return x_attn_B

    def unfold_C(self,x_attn_C,sizes): #input B,N,C,Nph*P,P
        B,N,C,H,W,P,Nph,Npw = sizes
        x_attn_C = x_attn_C.view(B*N,C,Nph*P,P)
        x_attn_C = F.unfold(x_attn_C,kernel_size=P,stride=P) #output B*N,C*P*P,Npw
        x_attn_C = x_attn_C.permute(0,2,1) #output B*N,Nph,C*P*P
        x_attn_C = x_attn_C.view(B,N,Nph,C,P,P)
        x_attn_C = x_attn_C.permute(0,2,1,3,4,5) #B,Nph,N,C,P,P
        #print('line 289 x_attn_C.shape:',x_attn_C.shape)
        x_attn_C = x_attn_C.contiguous().view(B*Nph,N,C,P,P)
        return x_attn_C

    def fold_C(self,x_attn_C,sizes): #input: B*Nph,N,C,P,P
        B,N,C,H,W,P,Nph,Npw = sizes
        x_attn_C = x_attn_C.view(B,Nph,N,C*P*P)
        x_attn_C = x_attn_C.permute(0,2,3,1) #B,N,C*P*P,Nph
        x_attn_C = x_attn_C.view(B*N,C*P*P,Nph)
        x_attn_C = F.fold(x_attn_C,output_size=(Nph*P,P),kernel_size=(P,P),stride=P) #output B*N,C,Nph*P,P
        x_attn_C = x_attn_C.view(B,N,C,Nph*P,P)
        return x_attn_C

    def forward(self,x):
        x_in = x.clone()
        B,N,C,H,W=x.shape
        x = x.repeat(1,1,2,1,1)
        P = self.op_img_size
        assert P<=H and P<=W
        Nph = n_patches_h = H//P
        Npw = n_patches_w = W//P

        #print(Nph," ",Npw)

        #Area A
        x_attn_A1 = x_in[:,:,:,:Nph*P,:Npw*P] #B,N,C,Nph*P,Npw*P, square area, includes NW corner
        x_attn_A2 = x_in[:,:,:,H-Nph*P:,W-Npw*P:] #B,N,C,Nph*P,Npw*P, square area, includes SE corner

        #print("test unfold_A,fold_A")
        #test_tensor = self.fold_A(self.unfold_A(x_attn_A1,sizes=(B,N,C,H,W,P,Nph,Npw)),sizes=(B,N,C,H,W,P,Nph,Npw))
        #check_psnr_result = util_gan.psnr.calculate_psnr(x_attn_A1.cpu().detach().numpy(),test_tensor.cpu().detach().numpy())
        #print("check psnr:",check_psnr_result)
        #assert str(check_psnr_result)=='inf'

        #For ViT: B,N,C,P,P. Should be one patch.
        if Nph>1 or Npw>1: #for patched convolutional attn
            x_attn_A1 = self.unfold_A(x_attn_A1,sizes=(B,N,C,H,W,P,Nph,Npw))
            x_attn_A2 = self.unfold_A(x_attn_A2,sizes=(B,N,C,H,W,P,Nph,Npw))

        #print('x_attn_A1.shape:',x_attn_A1.shape)
        x_attn_A1 = self._op(x_attn_A1)
        x_attn_A2 = self._op(x_attn_A2)

        if Nph>1 or Npw>1: #for patched convolutional attn
            x_attn_A1 = self.fold_A(x_attn_A1,sizes=(B,N,C,H,W,P,Nph,Npw))
            x_attn_A2 = self.fold_A(x_attn_A2,sizes=(B,N,C,H,W,P,Nph,Npw))

        x[:,:,:C,:Nph*P,:Npw*P] = x_attn_A1
        x[:,:,C:,H-Nph*P:,W-Npw*P:] = x_attn_A2

        #Area B Heightwise/vertical remainder
        if H>n_patches_h*P:
            x_attn_B1 = x_in[:,:,:,-P:,:Npw*P] #row, includes SW corner
            x_attn_B2 = x_in[:,:,:,:P,W-Npw*P:] #row, includes NE corner

            #print("test unfold_B,fold_B")
            #test_tensor = self.fold_B(self.unfold_B(x_attn_B1,sizes=(B,N,C,H,W,P,Nph,Npw)),sizes=(B,N,C,H,W,P,Nph,Npw))
            #check_psnr_result = util_gan.psnr.calculate_psnr(x_attn_B1.cpu().detach().numpy(),test_tensor.cpu().detach().numpy())
            #print("check psnr:",check_psnr_result)
            #assert str(check_psnr_result)=='inf'

            if Npw>1:
                x_attn_B1 = self.unfold_B(x_attn_B1,sizes=(B,N,C,H,W,P,Nph,Npw))
                x_attn_B2 = self.unfold_B(x_attn_B2,sizes=(B,N,C,H,W,P,Nph,Npw))

            #print('x_attn_B1.shape:',x_attn_B1.shape)
            x_attn_B1 = self._op(x_attn_B1)
            x_attn_B2 = self._op(x_attn_B2)

            if Npw>1:
                x_attn_B1 = self.fold_B(x_attn_B1,sizes=(B,N,C,H,W,P,Nph,Npw))
                x_attn_B2 = self.fold_B(x_attn_B2,sizes=(B,N,C,H,W,P,Nph,Npw))

            x[:,:,:C,-P:,:Npw*P] = x_attn_B1
            x[:,:,C:,:P,W-Npw*P:] = x_attn_B2

        #Area C Widthwise/horizontal remainder
        if H>n_patches_h*P:
            x_attn_C1 = x_in[:,:,:,:Nph*P,-P:] #column, includes NE corner
            x_attn_C2 = x_in[:,:,:,H-Nph*P:,:P] #column, includes SW corner

            #print("test unfold_C,fold_C")
            #test_tensor = self.fold_C(self.unfold_C(x_attn_C1,sizes=(B,N,C,H,W,P,Nph,Npw)),sizes=(B,N,C,H,W,P,Nph,Npw))
            #check_psnr_result = util_gan.psnr.calculate_psnr(x_attn_C1.cpu().detach().numpy(),test_tensor.cpu().detach().numpy())
            #print("check psnr:",check_psnr_result)
            #assert str(check_psnr_result)=='inf'

            if Nph>1:
                x_attn_C1 = self.unfold_C(x_attn_C1,sizes=(B,N,C,H,W,P,Nph,Npw))
                x_attn_C2 = self.unfold_C(x_attn_C2,sizes=(B,N,C,H,W,P,Nph,Npw))

            #print('x_attn_C1.shape:',x_attn_C1.shape)
            x_attn_C1 = self._op(x_attn_C1)
            x_attn_C2 = self._op(x_attn_C2)

            if Nph>1:
                x_attn_C1 = self.fold_C(x_attn_C1,sizes=(B,N,C,H,W,P,Nph,Npw))
                x_attn_C2 = self.fold_C(x_attn_C2,sizes=(B,N,C,H,W,P,Nph,Npw))

            x[:,:,:C,:Nph*P,-P:] = x_attn_C1
            x[:,:,C:,H-Nph*P:,:P] = x_attn_C2

        #Area D corner remainder
        if H>n_patches_h*P and W>n_patches_w*P:
            #print('x_attn_D1.shape:',x_in[:,:,:,-P:,-P:].shape)
            x[:,:,:C,-P:,-P:] = self._op(x_in[:,:,:,-P:,-P:])
            x[:,:,C:,:P,:P] = self._op(x_in[:,:,:,:P,:P])

        x = self.conv_fusion(x.view(B*N,2*C,H,W)).view(B,N,C,H,W)

        #print('operations_attn line 399 Spatial_Attn_Wrapper %s'%(utils.profiling.memory_usage()))

        return x


class Separable_NL_Spatial_patched(nn.Module):
    def __init__(self,nf,num_frames,patch_size):
        super(Separable_NL_Spatial_patched, self).__init__()
        self.patch_size = patch_size
        self.sdp_attn_spatial = Separable_NL_Spatial(nf=nf,num_frames=num_frames)
        self.conv_fusion = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)

    def forward(self,x):
        x_in = x.clone()
        B,N,C,H,W=x.shape
        x = x.repeat(1,1,2,1,1)
        P = self.patch_size
        n_full_patches_h = H//self.patch_size
        n_full_patches_w = W//self.patch_size

        for i in range(n_full_patches_h):
            for j in range(n_full_patches_w):
                #x[:,:,:,i*P:(i+1)*P,j*P:(j+1)*P] = self.sdp_attn_spatial(x_in[:,:,:,i*P:(i+1)*P,j*P:(j+1)*P])
                #(+) to do: += or =?
                x[:,:,:C,i*P:(i+1)*P,j*P:(j+1)*P] = self.sdp_attn_spatial(x_in[:,:,:,i*P:(i+1)*P,j*P:(j+1)*P])
                x[:,:,C:,H-(i+1)*P:H-i*P,W-(j+1)*P:W-j*P] = self.sdp_attn_spatial(x_in[:,:,:,H-(i+1)*P:H-i*P,W-(j+1)*P:W-j*P])

        remainder_h = H-n_full_patches_h*P
        remainder_w = W-n_full_patches_w*P

        if remainder_h>0:
            for j in range(n_full_patches_w):
                x[:,:,:C,-P:,j*P:(j+1)*P] = self.sdp_attn_spatial(x_in[:,:,:,-P:,j*P:(j+1)*P])
                x[:,:,C:,:P,W-(j+1)*P:W-j*P] = self.sdp_attn_spatial(x_in[:,:,:,:P,W-(j+1)*P:W-j*P])

        if remainder_w>0:
            for i in range(n_full_patches_h):
                x[:,:,:C,i*P:(i+1)*P,-P:] = self.sdp_attn_spatial(x_in[:,:,:,i*P:(i+1)*P,-P:])
                x[:,:,C:,H-(i+1)*P:H-i*P,:P] = self.sdp_attn_spatial(x_in[:,:,:,H-(i+1)*P:H-i*P,:P])

        if remainder_h>0 and remainder_w>0:
            x[:,:,:C,-P:,-P:] = self.sdp_attn_spatial(x_in[:,:,:,-P:,-P:])
            x[:,:,C:,:P,:P] = self.sdp_attn_spatial(x_in[:,:,:,:P,:P])

        x = self.conv_fusion(x.view(B*N,2*C,H,W)).view(B,N,C,H,W)
        return x


class Separable_NL_Spatial_patched32(Separable_NL_Spatial_patched):
    def __init__(self,nf,num_frames):
        super(Separable_NL_Spatial_patched32, self).__init__(nf,num_frames,patch_size=32)


class Separable_NL_Spatial_patched16(Separable_NL_Spatial_patched):
    def __init__(self,nf,num_frames):
        super(Separable_NL_Spatial_patched16, self).__init__(nf,num_frames,patch_size=16)


class Separable_NL_Channel(nn.Module):
    def __init__(self,nf,num_frames):
        super(Separable_NL_Channel, self).__init__()
        self.convA2T= nn.Conv3d(num_frames, num_frames, 1, 1, 0, bias=True)
        self.convA2C= nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.convB2T= nn.Conv3d(num_frames, num_frames, 1, 1, 0, bias=True)
        self.convB2C= nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.convD2T= nn.Conv3d(num_frames, num_frames, 1, 1, 0, bias=True)
        self.convD2C= nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        #self.scale=nn.Parameter(torch.ones([1,1],dtype=torch.float)*torch.sqrt(torch.Tensor([np.sqrt(7*16*16)]).float()), requires_grad=True)
        self.scale = nn.Parameter(torch.ones([1,1],dtype=torch.float)*torch.sqrt(torch.Tensor([np.sqrt(num_frames*16*16)]).float()), requires_grad=True)

        self.debug=False
        #self.debug=True

    def forward(self,x):
        B,N,C,H,W=x.shape
        #print('operations_attn line 425 x.shape:',x.shape)
        x_p=x.permute(0,2,1,3,4)
        A2T= self.convA2T(x).view(B, N, C, H, W).permute(0, 2, 1, 3, 4).contiguous().view(B, C, N*H*W)
        A2C= self.convA2C(x_p).permute(0,2,1,3,4).contiguous().view(B, N, C, H, W).permute(0, 2, 1, 3, 4).contiguous().view(B, C, N*H*W)
        B2T= self.convB2T(x).view(B, N, C, H, W).permute(0, 1, 3, 4, 2).contiguous().view(B, N*H*W, C)
        B2C= self.convB2C(x_p).permute(0,2,1,3,4).contiguous().view(B, N, C, H, W).permute(0, 1, 3, 4, 2).contiguous().view(B, N*H*W, C)
        D2T= self.convD2T(x).view(B, N, C, H, W).permute(0, 2, 1, 3, 4).contiguous().view(B, C, N*H*W)
        D2C= self.convD2C(x_p).permute(0,2,1,3,4).contiguous().view(B, N, C, H, W).permute(0, 2, 1, 3, 4).contiguous().view(B, C, N*H*W)
        M2T=self.softmax(torch.matmul(A2T, B2T)/self.scale)
        M2C=self.softmax(torch.matmul(A2C, B2C)/self.scale)
        E2T= torch.matmul(M2T, D2T).view(B, C, N, H, W).permute(0, 2, 1, 3, 4)
        E2C= torch.matmul(M2C, D2C).view(B, C, N, H, W).permute(0, 2, 1, 3, 4)

        if self.debug:
            print("channel forward")
            print(x.shape)
            print("C",C)

            print("A2T %s"%str(A2T.shape))
            print("A2C %s"%str(A2C.shape))

            print("B2T %s"%str(B2T.shape))
            print("B2C %s"%str(B2C.shape))

            print("D2T %s"%str(D2T.shape))
            print("D2C %s"%str(D2C.shape))

            print("M2T %s"%str(M2T.shape))
            print("M2C %s"%str(M2C.shape))

            print("E2T %s"%str(E2T.shape))
            print("E2C %s"%str(E2C.shape))

        return E2C+E2T+x.view(B,N,C,H,W)


class Separable_NL_Temporal(nn.Module):
    def __init__(self,nf,num_frames):
        super(Separable_NL_Temporal, self).__init__()
        nf=128
        #num_frames=7
        self.convA3T= nn.Conv3d(num_frames, num_frames, 1, 1, 0, bias=True)
        self.convA3C= nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.convB3T= nn.Conv3d(num_frames, num_frames, 1, 1, 0, bias=True)
        self.convB3C= nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.convD3T= nn.Conv3d(num_frames, num_frames, 1, 1, 0, bias=True)
        self.convD3C= nn.Conv3d(nf, nf, 1, 1, 0, bias=True)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.scale= nn.Parameter(torch.ones([1,1],dtype=torch.float)*torch.sqrt(torch.Tensor([np.sqrt(64*16*16)]).float()), requires_grad=True) #torch.nn.Parameter(torch.Tensor([np.sqrt(64*16*16)]))

    def forward(self,x):
        B,N,C,H,W=x.shape
        #print('operations_attn line 479 x.shape:',x.shape)
        x_p=x.permute(0,2,1,3,4)
        A3T= self.convA3T(x).view(B, N, C, H, W).view(B, N, C*H*W)
        A3C= self.convA3C(x_p).permute(0,2,1,3,4).contiguous().view(B, N, C, H, W).view(B, N, C*H*W)
        B3T= self.convB3T(x).view(B, N, C, H, W).view(B, N, C*H*W).permute(0, 2, 1)
        B3C= self.convB3C(x_p).permute(0,2,1,3,4).contiguous().view(B, N, C, H, W).view(B, N, C*H*W).permute(0, 2, 1)
        D3T= self.convD3T(x).view(B, N, C, H, W).view(B, N, C*H*W)
        D3C= self.convD3C(x_p).permute(0,2,1,3,4).contiguous().view(B, N, C, H, W).view(B, N, C*H*W)
        M3T=self.softmax(torch.matmul(A3T, B3T)/self.scale)
        M3C=self.softmax(torch.matmul(A3C, B3C)/self.scale)
        E3T= torch.matmul(M3T, D3T).view(B, N, C, H, W)
        E3C= torch.matmul(M3C, D3C).view(B, N, C, H, W)
        return E3T+E3C+x.view(B,N,C,H,W)


class EPAB_SpatioChannel(nn.Module):
    def __init__(self, nf=128, num_frames=7):
        super(EPAB_SpatioChannel, self).__init__()
        self.NL_Block_Vid_channel = SimpleNonLocal_Block_Video_NAS(num_frames, 'channel')

    def forward(self, F):
        B, T, C, H, W = F.shape
        channel = self.NL_Block_Vid_channel(F)
        out = channel +  F
        return out


class EPAB_SpatioTemporal(nn.Module):
    def __init__(self, nf=128, num_frames=7):
        super(EPAB_SpatioTemporal, self).__init__()
        self.NL_Block_Vid_temporal = SimpleNonLocal_Block_Video_NAS(nf, 'temporal')

    def forward(self, F):
        B,T,C,H,W=F.shape
        temporal = self.NL_Block_Vid_temporal(F)
        out = temporal +  F
        return out

#OPS_Attention={
        #'epab_spatiotemporal': lambda nf,num_frames: EPAB_SpatioTemporal(),
        #'epab_spatiochannel': lambda nf,num_frames: EPAB_SpatioChannel(),
        #'separable_channel' : lambda nf,num_frames:  Separable_NL_Channel(num_frames,64),
        #'separable_temporal' : lambda nf,num_frames:  Separable_NL_Temporal(num_frames,64),
        #'separable_spatial' : lambda nf,num_frames: Separable_NL_Spatial(num_frames,64),
        # }

#'separable_spatial' : lambda nf,num_frames: Separable_NL_Spatial(nf=nf,num_frames=num_frames),
#        'vit_spatial_patched32' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,patch_size=224,attn_spatial=Vit_Spatial32),
#        'vit_spatial_patched16' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,patch_size=224,attn_spatial=Vit_Spatial16)
#        'separable_spatial_patched32' : lambda nf,num_frames: Separable_NL_Spatial_patched32(nf=nf,num_frames=num_frames),
#        'separable_spatial_patched16' : lambda nf,num_frames: Separable_NL_Spatial_patched16(nf=nf,num_frames=num_frames),
#        'separable_spatial_patched32' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_spatial='separable_spatial_patched32'),
#        'separable_spatial_patched16' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_spatial='separable_spatial_patched16'),
OPS_Attention={
        'epab_spatiotemporal': lambda nf,num_frames: EPAB_SpatioTemporal(nf=nf,num_frames=num_frames),
        'epab_spatiochannel': lambda nf,num_frames: EPAB_SpatioChannel(nf=nf,num_frames=num_frames),
        'separable_channel' : lambda nf,num_frames:  Separable_NL_Channel(nf=nf,num_frames=num_frames),
        'separable_temporal' : lambda nf,num_frames:  Separable_NL_Temporal(nf=nf,num_frames=num_frames),
        'separable_spatial' : lambda nf,num_frames: Separable_NL_Spatial(nf=nf,num_frames=num_frames),
        'separable_spatial_patched32_original' : lambda nf,num_frames: Separable_NL_Spatial_patched32(nf=nf,num_frames=num_frames),
        'separable_spatial_patched16_original' : lambda nf,num_frames: Separable_NL_Spatial_patched16(nf=nf,num_frames=num_frames),
        'separable_spatial_patched32_wrapper' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='separable_spatial_patched32'),
        'separable_spatial_patched16_wrapper' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='separable_spatial_patched16'),
        'vit_spatial_patched32' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='vit_spatial_patched32'),
        'vit_spatial_patched16' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='vit_spatial_patched16'),
        'vit_spatial_patched16_image96' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='vit_spatial_patched16_image96'),
        'vit_spatial_patched16_image32' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='vit_spatial_patched16_image32'),
        'vit_spatial_patched8_image32' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='vit_spatial_patched8_image32'),
        'vit_spatial_patched4_image32' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='vit_spatial_patched4_image32'),
        'vit_channel_image32' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='vit_channel_image32'),
        'vit_spatiochannel_patched4_image32' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='vit_spatiochannel_patched4_image32'),
        'vit_spatiochannel_patched16_image32' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='vit_spatiochannel_patched16_image32'),
        'vit_spatial_map_patched4_image32' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='vit_spatial_map_patched4_image32'),
        'vit_channel_map_image32' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='vit_channel_map_image32'),
        'vit_spatiochannel_map_patched16_image32' : lambda nf,num_frames: Spatial_Attn_Wrapper(nf=nf,num_frames=num_frames,attn_type='vit_spatiochannel_map_patched16_image32')
}