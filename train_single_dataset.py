import argparse
from tqdm import tqdm, trange
from torch.optim import Adam
from torch_geometric.data import Data, HeteroData

from src.baselines import *
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.transforms import ToSparseTensor, VirtualNode, ToUndirected
from torch_geometric.nn import to_hetero
from src.transforms import add_laplace_positional_encoding, add_virtual_node
import yaml
import os
import csv

from train_baselines import *

output_dir = '/data/sam/terrain/'


def format_log_dir(output_dir, 
                   dataset_name, 
                   siamese, 
                   modelname, 
                   vn, 
                   aggr, 
                   loss_func, 
                   layer_type,
                   p,
                   trial):
    log_dir = os.path.join(output_dir, 
                           'models',
                           'single_dataset', 
                           dataset_name, 
                           layer_type,
                           'vn' if vn else 'no-vn',
                           'siamese' if siamese else 'mlp',
                           f'p-{p}')
    if not siamese:
        log_dir = os.path.join(log_dir, aggr)
    log_dir = os.path.join(log_dir, loss_func, modelname, trial)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, help='Needed to construct output directory')
    parser.add_argument('--train-data', type=str)
    parser.add_argument('--test-data', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--config', type=str, default='configs/config-base.yml')
    parser.add_argument('--device', type=str)
    parser.add_argument('--siamese', type=int, default=0)
    parser.add_argument('--vn', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--aggr', type=str, default='max')
    parser.add_argument('--loss', type=str, default='mse_loss')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--layer-type', type=str)
    parser.add_argument('--trial', type=str)
    parser.add_argument('--p', type=int, default=1 )
    parser.add_argument('--finetune', type=int, default=0)
    parser.add_argument('--include-edge-attr', type=int, default=0)

    args = parser.parse_args()
    siamese = True if args.siamese == 1 else False
    vn = True if args.vn == 1 else False 
    aggr = args.aggr
    finetune=True if args.finetune == 1 else False
    # trial = args.trial

    trials = ['1', '2', '3', '4', '5']
    

    # files = ['edge-weight-50k-constrained-0.npz', 
    #          'edge-weight-50k-constrained-1.npz', 
    #          'edge-weight-50k-constrained-2.npz', 
    #          'edge-weight-50k-constrained-3.npz',
    #          'edge-weight-50k-constrained-4.npz']
    # trials = [args.trial]
    for i in range(len(trials)):
        trial = trials[i]
        #data_file =files[i]
    
        # Load data 
        train_file = os.path.join(output_dir, 'data', f'{args.train_data}-{trial}.npz')
        print("Training file", train_file)
        test_file = os.path.join(output_dir, 'data', args.test_data)

        train_data = np.load(train_file, allow_pickle=True)
        test_data = np.load(test_file, allow_pickle=True)

        train_dataset, train_node_features, train_edge_index = npz_to_dataset(train_data)
        train_edge_attr = None 
        if args.include_edge_attr:
            train_edge_attr = train_data['distances']
        print("Number of nodes:", len(train_node_features))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

        test_dataset, test_node_features, test_edge_index = npz_to_dataset(test_data)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)

        # Load model configs
        with open(args.config, 'r') as file:
            model_configs = yaml.safe_load(file)
        
        loss_data = []
        
        for modelname in model_configs:
            # log_dir = format_log_dir(output_dir, 
            #                         args.dataset_name, 
            #                         siamese, 
            #                         modelname, 
            #                         vn, 
            #                         aggr, 
            #                         args.loss, 
            #                         args.layer_type,
            #                         args.p,
            #                         args.trial)
            
            log_dir = format_log_dir(output_dir, 
                                    args.dataset_name, 
                                    siamese, 
                                    modelname, 
                                    vn, 
                                    aggr, 
                                    args.loss, 
                                    args.layer_type,
                                    args.p,
                                    trial)
            config=model_configs[modelname]

            output = train_single_graph_baseline1(train_node_features, train_edge_index, train_dataloader, 
                                                test_node_features, test_edge_index, test_dataloader,layer_type=args.layer_type, 
                                                loss_func=args.loss, model_config = config, epochs=args.epochs, device=args.device,
                                                siamese=siamese, log_dir=log_dir, virtual_node=vn, aggr=aggr, lr=args.lr, p=args.p, 
                                                log=False, finetune=finetune, edge_attr=train_edge_attr)
        #loss_data.append({'modelname':modelname, 'loss':output[-1]})

        # Keep track of validation losses for each configuration
        # if 'base' in args.config:
        #     print("finished training example model")
        #     return 0
        # modeltype = args.layer_type
        # fieldnames = ['modelname', 'loss']
        # csv_file = os.path.join('output', 
        #                         'single_dataset',
        #                         args.dataset_name, 
        #                         args.layer_type, 
        #                         'siamese' if siamese else 'mlp', 
        #                         'vn' if vn else 'no-vn', 
        #                         aggr, 
        #                         args.loss,
        #                         args.trial,
        #                         f'p-{args.p}')
        # if not os.path.exists(csv_file):
        #     os.makedirs(csv_file)
        # csv_file = csv_file + f'/{modeltype}.csv'
        # csvfile = open(csv_file, 'w', newline='')
        # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        # for row in loss_data:
        #     writer.writerow(row)
        # csvfile.close()
        # print(f'Data has been written to {csv_file}')
    
if __name__=='__main__':
    main()