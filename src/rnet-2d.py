import sys
import os
from os import path
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

sys.path.append(path.join(path.dirname(__file__), '../RibonanzaNet'))
from Network import *

from utils import RNA_Dataset, load_config_from_yaml

USE_GPU = torch.cuda.is_available()

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config):
        config.dropout=0.3
        super(finetuned_RibonanzaNet, self).__init__(config)

        self.dropout=nn.Dropout(0.0)
        self.ct_predictor=nn.Linear(64,1)

    def forward(self,src):
        
        #with torch.no_grad():
        _, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))

        pairwise_features=pairwise_features+pairwise_features.permute(0,2,1,3) #symmetrize

        output=self.ct_predictor(self.dropout(pairwise_features)) #predict

        return output.squeeze(-1)

    def get_embeddings(self, src,src_mask=None,return_aw=False):
        B,L=src.shape
        src = src
        src = self.encoder(src).reshape(B,L,-1)
        
        pairwise_features=self.outer_product_mean(src)
        pairwise_features=pairwise_features+self.pos_encoder(src)

        attention_weights=[]
        for i,layer in enumerate(self.transformer_encoder):
            src,pairwise_features=layer(src, pairwise_features, src_mask,return_aw=return_aw)

        return src, pairwise_features

model = finetuned_RibonanzaNet(load_config_from_yaml(path.join(path.dirname(__file__), '../RibonanzaNet', 'configs/pairwise.yaml')))
if USE_GPU:
    model = model.cuda()
model.load_state_dict(torch.load(path.join(path.dirname(__file__), '../RibonanzaNet-Weights', 'RibonanzaNet-SS.pt'), map_location='cpu'))
model.eval()

if __name__ == '__main__':
    from tqdm import tqdm

    seq = sys.argv[1]
    test_dataset=RNA_Dataset(pd.DataFrame([{'sequence': seq}]))

    test_preds=[]
    for i in tqdm(range(len(test_dataset))):
        example=test_dataset[i]
        sequence=example['sequence']
        if USE_GPU:
            sequence = sequence.cuda()
        sequence = sequence.unsqueeze(0)

        with torch.no_grad():
            pred = model(sequence).sigmoid()
            if USE_GPU:
                pred = pred.cpu()
            test_preds.append(pred.numpy())

    # create dummy arnie config
    os.environ['NUPACKHOME'] = '/tmp'
    from arnie.pk_predictors import _hungarian

    def mask_diagonal(matrix, mask_value=0):
        matrix=matrix.copy()
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if abs(i - j) < 4:
                    matrix[i][j] = mask_value
        return matrix

    test_preds_hungarian=[]
    hungarian_structures=[]
    hungarian_bps=[]
    for i in range(len(test_preds)):
        s,bp=_hungarian(mask_diagonal(test_preds[i][0]),theta=0.5,min_len_helix=1) #best theta based on val is 0.5
        hungarian_bps.append(bp)
        ct_matrix=np.zeros((len(s),len(s)))
        for b in bp:
            ct_matrix[b[0],b[1]]=1
        ct_matrix=ct_matrix+ct_matrix.T
        test_preds_hungarian.append(ct_matrix)
        hungarian_structures.append(s)

    print('structure:' + hungarian_structures[0])