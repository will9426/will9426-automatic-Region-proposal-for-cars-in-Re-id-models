# encoding: utf-8
"""
@author:  will
@contact: williamramirez2694@gmail.com
"""
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# Ejemplo: guardando imágenes en un bucle
from torchvision.utils import save_image
from utils.reid_metric import R1_mAP, R1_mAP_reranking

class FeatureAdjuster(nn.Module):
    def __init__(self, in_channels=512, out_channels=8):
        super(FeatureAdjuster, self).__init__()
        # Convolución 1x1 para reducir la dimensionalidad de 512 a 8
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, features):
        # Ajustar de [128, 512, 14, 14] a [128, 8, 14, 14]
        adjusted_features = self.conv1x1(features)
        return adjusted_features

def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            if len(batch) == 4:
                data, pids, camids, img_paths = batch
                proposal = None
            else:
                data, proposal, pids, camids, img_paths = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            proposal = proposal.to(device) if torch.cuda.device_count() >= 1 and proposal is not None else proposal
            features = model((data, proposal))
            #print(model)
            #print(features['layer4']['feat'].shape)
            #print(data.shape,proposal.shape,features['layer4']['feat'].shape)
            # Instanciar el ajustador de características
            i=0
            for img,img01,proposal in zip(img_paths,data,proposal):
                tensor_permuted = img01.permute(1, 2, 0)
                proposal_permuted = proposal.permute(1, 2, 0)
                proposal_permuted = F.interpolate(proposal_permuted.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                feature_tensor= features['layer4']['prop'][i]
                reshaped_tensor = feature_tensor.view(1, 1, 32, 16)
               # print(reshaped_tensor.shape)  # Debería imprimir torch.Size([1, 1, 32, 16])
                interpolated_tensor = F.interpolate(reshaped_tensor, size=(224, 224), mode='bilinear', align_corners=False)
                #print(interpolated_tensor.shape)  # Debería imprimir torch.Size([1, 1, 224, 224])
                final_tensor = interpolated_tensor.squeeze()
                final_tensor[final_tensor<0]*-1
                #print(final_tensor.min())

                #print(final_tensor.shape)  # Debería imprimir torch.Size([224, 224])                
                i+=1
                # Remover la dimensión adicional si es necesario para volver a [8, 224, 224]
                proposal_permuted = proposal_permuted.squeeze(0)
                sumva=0
                for index,singlem in enumerate(proposal_permuted):
                    sumva=sumva+singlem
                    #save_image(singlem, f"./activation_maps/mask{index}_guardada_{img.split('/')[-1].replace('.jpg','')}.png")

                sumva[sumva>1]=1
                final_tensor=final_tensor*sumva
                #final_tensor[final_tensor<0]=1
                activation_map = final_tensor
                activation_map = activation_map.cpu().numpy()  # Convertir el tensor a NumPy

                # Normaliza el mapa de activación entre 0 y 1
                activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
                activation_map_colored = plt.cm.jet(activation_map)  # Jet es un ejemplo de colormap, puedes elegir otro

                # Convertir el mapa de calor a un tensor de PyTorch, descartando el canal alfa
                activation_map_colored = torch.tensor(activation_map_colored[:, :, :3], device=0)#.permute(2, 0, 1)                
                tensor_normalized = tensor_permuted.cpu().numpy() - tensor_permuted.cpu().numpy().min()
                tensor_normalized = tensor_normalized / tensor_normalized.max() 
                #save_image(sum, f"./activation_maps/mask_guardada_{img.split('/')[-1].replace('.jpg','')}.png")                
                # Guardar la imagen
                overlayed_image = 0.5 * torch.tensor(tensor_normalized, device=0) + 0.5 * activation_map_colored
                mask = torch.tensor(tensor_normalized, device=0)*sumva.unsqueeze(-1) 
                #print(overlayed_image.shape)
                #print(sumva.shape)
                # Asegúrate de normalizar la imagen superpuesta
                tensor_normalized=torch.tensor(tensor_normalized, device=0)
                overlayed_image = overlayed_image / overlayed_image.max() 
                tensor_normalized = tensor_normalized.unsqueeze(0)  # Añade una dimensión para el batch
                overlayed_image = overlayed_image.unsqueeze(0)  #    
                mask = mask.unsqueeze(0) 
                        
                concatenated_image = torch.cat((tensor_normalized, overlayed_image,mask), dim=2)  # Dimensión 2 para concatenar verticalmente
                #print(tensor_normalized.shape)              
                
                #print(concatenated_image.shape)
                #concatenated_image = concatenated_image.clamp(0, 1)  # Clampa los valores a [0, 1]
                #concatenated_image = concatenated_image.to(torch.float32)  # Asegúrate de que sea float32

                #save_image(concatenated_image[0].permute(2, 0, 1), f"./activation_maps3/imagen_guardada_{img.split('/')[-1].replace('.jpg','')}.png")


            return features['layer4']['feat'], pids, camids, img_paths

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def write_txt(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(str(line) + '\n')

def get_index(distmat, data_loader, output_dir, num_query):
    indices = np.argsort(distmat, axis=1)
    for idx, (name, pid, camid) in enumerate(data_loader.dataset.dataset[0:num_query]):
        imgname = os.path.basename(name)
        sort_index = indices[idx][0:100]
        save_path = os.path.join(output_dir, imgname.replace('jpg', 'txt'))
        write_txt(save_path, sort_index)

def generate_txt(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(str(line) + '\n')


def inference(
        cfg,
        model,
        val_loader,
        num_query,
):
    device = cfg.MODEL.DEVICE
    log_period = cfg.SOLVER.LOG_PERIOD
    output_dir = cfg.OUTPUT_DIR

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={
            'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, remove_camera=True,
                             extract_feat=True)}, device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={
            'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, remove_camera=True,
                                       extract_feat=True)}, device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_iteration(engine):
        iter = (engine.state.iteration - 1) % len(val_loader) + 1
        if iter % log_period == 0:
            logger.info("Extract Features. Iteration[{}/{}]"
                        .format(iter, len(val_loader)))

    evaluator.run(val_loader)
    distmat, cmc, mAP = evaluator.state.metrics['r1_mAP']

    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    return mAP, cmc[0], cmc[4]