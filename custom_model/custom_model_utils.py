
from torch import nn

def count_frozen_params(model):
    total_params = 0
    frozen_params = 0

    for param in model.parameters():
        total_params += param.numel()  # Contar el número total de parámetros

        if not param.requires_grad:
            frozen_params += param.numel()  # Contar el número de parámetros congelados

    return total_params, frozen_params


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
            
def weights_init_attention(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.constant_(m.weight, 0.0)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def load_param(self, trained_path):
    param_dict = torch.load(trained_path)
    if 'state_dict' in param_dict:
        param_dict = param_dict['state_dict']
    for i in param_dict:
        if 'classifier' in i:
            continue
        self.state_dict()[i].copy_(param_dict[i])