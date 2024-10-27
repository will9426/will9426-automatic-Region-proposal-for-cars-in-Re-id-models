import os
import numpy as np
from torchvision.utils import save_image
import os.path as osp
import torch
from torch import nn
import numpy as np
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .BasicBottleneck import ResidualBlock, ResNeXtBottleneck
from .custom_model_utils import *
import random
from scipy.stats.qmc import Halton
import torch.nn.functional as F

class Baseline(nn.Module):
    in_planes = 2048
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice,
                 mode='DPRC', prop_num=10, reduction=16, multi_nums=2, embed_num=128, temp=10.0):
        super(Baseline, self).__init__()
        self.model_name = model_name
        self.mode = mode
        self.prop_num = prop_num
        self.reduction = reduction
        self.multi_nums = multi_nums
        self.embed_num = embed_num
        '''The backbone network.'''
        self.base = ResNet(last_stride=last_stride,
                            block=Bottleneck,
                            layers=[3, 4, 6, 3])
        
        if pretrain_choice == 'imagenet':
            self.base.load_param(osp.expanduser(model_path))
            print('Loading pretrained ImageNet model......')

        '''calculate the channels of multi output.'''
        self.num_features = []
        for i in range(self.multi_nums):
            index = len(self.base._modules['layer' + str(4 - i)]._modules) - 1
            bn_count = len(
                [x for x in self.base._modules['layer' + str(4 - i)]._modules[str(index)]._modules if 'bn' in x])
            self.num_features.append(
                eval('self.base.layer' + str(4 - i) + str([index]) + '.bn' + str(bn_count) + '.num_features'))
        self.num_features = self.num_features[::-1]

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        self.bottleneck = []
        self.classifier = []
        self.refine_concat = []
        self.refine_prop = []
        self.refine_base = []

        self.bottleneck.append(nn.BatchNorm1d(self.embed_num * 2))
        self.bottleneck[0].bias.requires_grad_(False)
        self.bottleneck[0].apply(weights_init_kaiming)
        self.bottleneck = nn.Sequential(*self.bottleneck)

        self.classifier.append(nn.Linear(self.embed_num * 2, self.num_classes, bias=False))
        self.classifier[0].apply(weights_init_classifier)
        self.classifier = nn.Sequential(*self.classifier)

        self.refine_concat = []
        self.refine_prop = []
        self.refine_base = []

        # Bucle para crear y agregar los bloques a las listas
        for i in range(len(self.num_features)):
            self.refine_concat.append(self.create_refine_block(self.num_features[i] * 2, self.embed_num * 2))
            self.refine_prop.append(self.create_refine_block(self.num_features[i], self.embed_num))
            self.refine_base.append(self.create_refine_block(self.num_features[i], self.embed_num))

        # Convertir listas a nn.Sequential
        self.refine_concat = nn.Sequential(*self.refine_concat)
        self.refine_prop = nn.Sequential(*self.refine_prop)
        self.refine_base = nn.Sequential(*self.refine_base)

        total_params, frozen_params = count_frozen_params(self.base)
        print("Total de parámetros:", total_params)
        print("Parámetros congelados:", frozen_params)
        print("Parámetros entrenables:", total_params - frozen_params)

    def create_refine_block(self,in_channels, out_channels):
        block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            ResidualBlock(input_channels=in_channels, output_channels=out_channels),
            nn.Dropout(0.1),
        )
        block.apply(weights_init_kaiming)
        return block

    def generate_contour_masks(self,images, threshold1=100, threshold2=200):
        """
        Genera máscaras binarias basadas en el contorno de cada imagen usando el detector de bordes de Canny.
        
        :param images: Tensor de imágenes de entrada con forma [batch, channels, height, width].
        :param threshold1: Primer umbral para el algoritmo de Canny.
        :param threshold2: Segundo umbral para el algoritmo de Canny.
        :return: Tensor de máscaras binarias con forma [batch, 1, height, width].
        """
        batch_size, channels, height, width = images.shape
        contour_masks = torch.zeros((batch_size, 1, height, width), device=images.device)
        
        for i in range(batch_size):
            # Convertir la imagen de tensor a numpy y a escala de grises
            image_np = images[i].cpu().numpy().transpose(1, 2, 0)
            gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype('uint8')
            
            # Detectar los bordes usando el algoritmo de Canny
            edges = cv2.Canny(gray_image, threshold1, threshold2)
            
            # Convertir la máscara de bordes a un tensor y añadirlo al batch
            contour_masks[i, 0] = torch.tensor(edges, device=images.device)
        
        # Normalizar las máscaras a valores binarios
        contour_masks = (contour_masks > 0).float()
        
        return contour_masks

    def generate_contour_based_masks(self,contour_tensor, num_regions=8, output_size=14):
        batch_size, _, height, width = contour_tensor.shape
        masks = torch.zeros((batch_size, num_regions, height, width), device=contour_tensor.device)
        kernel = np.ones((2, 2), np.uint8)
        # Dividimos el contorno en regiones de igual tamaño
        region_width = width // num_regions
        for b in range(batch_size):
            for i in range(num_regions):
                start = i * region_width
                end = (i + 1) * region_width if i < num_regions - 1 else width  # Asegura que la última franja cubra el resto de la imagen
                # Convertir la franja a numpy
                region = contour_tensor[b, 0, :, start:end].cpu().numpy()

                # Dilatar la franja
                dilated_region = cv2.dilate(region.astype(np.uint8), kernel, iterations=2)

                # Asignar la franja dilatada a la máscara
                masks[b, i, :, start:end] = torch.tensor(dilated_region, device=contour_tensor.device)



        # Redimensionar a 14x14
        masks = F.interpolate(masks, size=(output_size, output_size), mode='nearest')

        return masks

    def extract_proposal_feature(self, base_feat, proposal, prop_num=10):
        prop_feat = base_feat.new_zeros(size=base_feat.size(), requires_grad=base_feat.requires_grad)
        prop_feat = prop_feat.unsqueeze(1)
        prop_feat = prop_feat.repeat(1, prop_num, 1, 1, 1)
        for num in range(prop_num):
            prop = proposal[:, num, :, :].unsqueeze(1)
            prop_feat[:, num, :, :, :] = base_feat * prop
        return prop_feat

    def generate_qmc_masks(self, batch_size, num_proposals, image_size, mask_size):
        qmc = Halton(d=2)
        samples = qmc.random(n=batch_size * num_proposals)
        
        # Normalizar y escalar muestras a las dimensiones de la imagen
        samples = (samples * (image_size - mask_size)).astype(int)
        
        masks = torch.zeros((batch_size, num_proposals, image_size, image_size), device=0)
        
        for b in range(batch_size):
            for p in range(num_proposals):
                x_start, y_start = samples[b * num_proposals + p]
                
                # Definir las franjas uniformes
                num_bands = 8
                band_width = image_size // num_bands
                
                # Crear una franja de máscara con ancho específico
                y_band = (p % num_bands) * band_width
                y_end_band = min(y_band + band_width, image_size)
                
                # No debe ser mayor al 30% del área
                x_end = min(x_start + mask_size, image_size)
                y_end = min(y_start + mask_size, y_end_band)
                
                # Crear la máscara con la región propuesta
                mask = torch.zeros((image_size, image_size), device=0)
                mask[x_start:x_end, y_start:y_end] = 1
                
                # Validar el área de la máscara
                mask_area = mask.sum().item() / (image_size * image_size)
                if mask_area > 0.5:
                    mask = mask * (0.5 / mask_area)
                
                # Asignar la máscara al batch
                masks[b, p] = mask
                
                # Asegurarse de que no hay máscara con solo 0 o 1 píxeles
                if mask.sum().item() <= 1:
                    masks[b, p] = torch.zeros((image_size, image_size), device=0)
        
        return masks

    def generate_contour_masks_resnet(self, images, layer_name='layer1'):
        """
        Genera máscaras binarias basadas en el contorno de las características intermedias de ResNet.
        
        :param images: Tensor de imágenes de entrada con forma [batch, channels, height, width].
        :param layer_name: El nombre de la capa de ResNet que se utilizará para la extracción de características.
        :return: Tensor de máscaras binarias con forma [batch, 1, height, width].
        """
        # Pasar las imágenes a través de las primeras capas de ResNet
        features = images
        for name, module in self.base._modules.items():
            features = module(features)
            if name == layer_name:
                break
        
        # Tomar la magnitud de los gradientes como contornos
        contour_masks = torch.mean(features, dim=1, keepdim=True)  # Promediar sobre el canal de características
        contour_masks = torch.abs(contour_masks)  # Tomar el valor absoluto (magnitud)
        
        # Normalizar a valores binarios
        contour_masks = (contour_masks > contour_masks.mean()).float()
        
        return self.generate_contour_based_masks(contour_masks)


    def forward(self, x):
        x, _ = x
        if self.mode=='mask':  
            proposal=_
        if self.mode=='QMC':        
            proposal = self.generate_qmc_masks(x.shape[0],8,14,9) 
        if self.mode=='DPRC':
            proposal = self.generate_contour_masks_resnet(x) 
        features = {}
        num = 0
        count=0
        for name, module in self.base._modules.items():
            '''added for backbone.'''
            if name == 'avgpool':
                break
            x = module(x) #
            if 'layer' not in name:
                continue
            if 'resnet' in self.model_name and int(name[-1]) <= 4 - self.multi_nums:
                continue
            base_feat = x.clone()
            prop_feat_m = self.extract_proposal_feature(base_feat, proposal, prop_num=self.prop_num)
            prop_feat_m = prop_feat_m.sum((-2, -1)) / proposal.unsqueeze(2).repeat(1, 1, base_feat.size(1), 1, 1).sum((-2, -1)).type(torch.float)
            if proposal.dim() == 4:
              reduce = torch.sigmoid(proposal.mean(dim=[2, 3], keepdim=True))
            elif proposal.dim() == 2:
              reduce = torch.sigmoid(proposal.mean(dim=1, keepdim=True))
            else:
              raise ValueError(f"Unsupported number of dimensions: {proposal.dim()}")
            prop_feat_m = proposal*reduce

            prop_feat_m = prop_feat_m.sum(1, keepdim=True) * base_feat

            prop_feat_m = prop_feat_m + base_feat
            global_prop_feat = torch.cat((base_feat, prop_feat_m), 1)
            global_prop_feat = self.refine_concat[num](global_prop_feat)

            global_prop_feat = self.gap(global_prop_feat)                            # (b, 2048, 1, 1)
            global_prop_feat = global_prop_feat.view(global_prop_feat.shape[0], -1)  # flatten to (bs, 2048)

            # global feature and part-guided feature.
            prop_feat_m = self.refine_prop[num](prop_feat_m)
            base_feat = self.refine_base[num](base_feat)

            prop_feat_m = self.gap(prop_feat_m)
            prop_feat_m = prop_feat_m.view(prop_feat_m.shape[0], -1)
            #print(base_feat.shape)

            base_feat = self.gap(base_feat)
            #print(base_feat.shape)
            base_feat = base_feat.view(base_feat.shape[0], -1)


            # last neck for softmax function.
            if self.neck == 'no':
                feat = global_prop_feat.clone()
            elif self.neck == 'bnneck':
                feat = self.bottleneck[num](global_prop_feat)
            if self.training:
                cls_score = self.classifier[num](feat)

                if name not in features:
                    features[name] = {}
                features[name]['global'] = global_prop_feat
                features[name]['reduce'] = reduce
                features[name]['prop'] = prop_feat_m
                features[name]['base'] = base_feat
                features[name]['cls'] = cls_score
            else:
                if name not in features:
                    features[name] = {}
                features[name]['feat'] = feat
                features[name]['global'] = global_prop_feat
                features[name]['reduce'] = reduce
                features[name]['base'] = base_feat
                features[name]['prop'] = prop_feat_m

            num += 1

        return features

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])