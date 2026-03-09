# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:44:15 2022

@author: ZNDX002
"""

from train import Predict
import yaml
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    config = yaml.load(open(os.path.join(PROJECT_ROOT, 'config', 'config.yaml'), "r"), Loader=yaml.FullLoader)
    config['result_folder'] = os.path.join(PROJECT_ROOT, config['result_folder'])
    dataset = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'ccsbase_4_2.csv'))
    model_path = os.path.join(PROJECT_ROOT, 'model', 'Graphccs_model.pt')
    model_predict = Predict(dataset,model_path,**config)
    y=model_predict.ccs_predict()
    dataset['predicts']=''
    for i in range(len(dataset['SMILES'])):
        dataset.loc[i,'predicts']=y[i]

if __name__ == "__main__":
    main()
