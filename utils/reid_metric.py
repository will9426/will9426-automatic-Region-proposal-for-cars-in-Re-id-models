# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking
import cv2
import matplotlib.pyplot as plt


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', remove_camera=True, extract_feat=False, isvisual='no'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.remove_camera = remove_camera
        self.extract_feat = extract_feat
        self.isvisual = True if isvisual=='yes' else False

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid, _ = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid.cpu())) if isinstance(pid, torch.Tensor) and pid.is_cuda else self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, remove_camera=self.remove_camera, isviaual=self.isvisual)
        if self.extract_feat:
            return distmat, cmc, mAP
        else:
            return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', remove_camera=True, extract_feat=False):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.remove_camera = remove_camera
        self.extract_feat = extract_feat

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.name=[]

    def update(self, output):
        feat, pid, camid, _ = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.name.extend(np.asarray(_))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_name=np.asarray(self.name[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_name=np.asarray(self.name[self.num_query:])

        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        
        for name,cid,dist in zip(q_name,q_pids,distmat):
            query=cv2.imread(name)
            query = cv2.resize(query, [224,224], interpolation=cv2.INTER_LINEAR)

            #print(query.shape)             
            concat_arr = query
            #print(dist.min())
            dict_res={}
            for index,dist_val in enumerate(dist):
                dict_res[dist_val]=g_name[index]   
            diccionario_ordenado = dict(sorted(dict_res.items(), key=lambda item: item[0], reverse=False))
            for k,element in enumerate(diccionario_ordenado):
                #print(element)
                if k<5:
                    result=cv2.imread(diccionario_ordenado[element])
                    #emb=gf[index]
                    #print(emb.shape)
                    # # Definir el color del borde (Rojo en BGR)
                    # color_rojo = (0, 0, 255)  # En formato BGR: Azul, Verde, Rojo
                    # color_verde = (0, 255, 0)  # En formato BGR: Azul, Verde, Rojo
                    # pid=g_pids[index]
                    # if pid==cid:
                    #     cor=color_verde
                    # else:
                    #     cor=color_rojo
                    # # Definir el grosor del borde
                    # grosor_borde = 10  # Puedes ajustar este valor

                    # # Agregar el borde a la imagen (cambia 'color_rojo' a 'color_verde' si quieres borde verde)
                    # result = cv2.copyMakeBorder(
                    #     result,
                    #     grosor_borde, grosor_borde, grosor_borde, grosor_borde,  # Grosor para cada lado (arriba, abajo, izquierda, derecha)
                    #     cv2.BORDER_CONSTANT,  # Tipo de borde: CONSTANTE
                    #     value=color_rojo  # Color del borde
                    # )




                    result = cv2.resize(result, [224,224], interpolation=cv2.INTER_LINEAR)
                    #print(result.shape,query.shape)
                    concat_arr = np.hstack((concat_arr, result))
                    #print(name,q_name[index])
            #print(name.split('/'))
            #cv2.imwrite('./consultas2VRIClargeveri/'+name.split('/')[-1], concat_arr)
                    

            #print(name,dist.shape)
            
        #print(g_pids,g_camids)
        #print(q_name)


        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, remove_camera=self.remove_camera)

        if self.extract_feat:
            return distmat, cmc, mAP
        else:
            return cmc, mAP
        