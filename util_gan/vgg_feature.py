from torchvision.models.vgg import vgg19
import torch
import torch.nn as nn
# torch.distributed.init_process_group(backend="nccl")

# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)
class VGGFeature(nn.Module):
    def __init__(self, before_act=True, feature_layer=34):
        super(VGGFeature, self).__init__()
        self.vgg = vgg19(pretrained=True)
        self.feature_layer = feature_layer

        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        if before_act:
            self.features = nn.Sequential(*list(self.vgg.features.children())[:(feature_layer+1)])#.to(device)
        else:
            self.features = nn.Sequential(*list(self.vgg.features.children())[:(feature_layer)])#.to(device)

        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def __call__(self,x):
        x = (x - self.mean) / self.std
        # print('x size:', x.size())
        x_vgg = self.features(x)
        # print('x_vgg size:', x_vgg.size())
        return x_vgg
