import os
import random
import argparse

import dgl
import torch
from adamp import AdamP
import numpy as np
from box import Box

from MoleculeBench import dataset_info

from molhgcn.hypergraphnetwork.Networks import Net
from molhgcn.utils.chem.data import GraphDataset_Classification, GraphDataLoader_Classification
from molhgcn.utils.chem.stat_utils import compute_auroc
from generate_data import get_classification_dataset
from hyper_params import hyper_params


def get_config(ds: str,
               init_feat: bool,
               fg_cyc: bool,
               num_neurons: list,
               input_norm: str,
               nef_dp: float,
               reg_dp: float,
               hid_dims: int
               ):
    config = Box({
        'model': {
            'num_neurons': num_neurons,
            'input_norm': input_norm,
            'nef_dp': nef_dp,
            'reg_dp': reg_dp,
            'node_hidden_dim': hid_dims,
            'edge_hidden_dim': hid_dims,
            'fg_hidden_dim': hid_dims,
        },
        'data': {
            'init_feat': init_feat,
            'fg_cyc': fg_cyc,
        },
        'train': {
            'ds': ds,
            'n_procs': 3,
            'lr': 1e-3,
            'bs': 128,
            'epochs': 100,
            'T_0': 50,
        },
        'log': {
            'log_every': 50
        },
        'wandb': {
            'group': None
        }
    })
    return config


def main(ds: str,
         data_seed: int,
         run_seed: int,
         init_feat: bool,
         fg_cyc: bool,
         fully_connected_fg: bool,
         num_neurons: list,
         input_norm: str,
         nef_dp: float,
         reg_dp: float,
         hid_dims: int,
         device: str):
    info = dataset_info(ds)
    if info.splitting == 'random':
        result_dir = f'./saved_models/{ds}/data_seed={data_seed}/run_seed={run_seed}'
    elif info.splitting == 'scaffold':
        result_dir = f'./saved_models/{ds}/run_seed={run_seed}'

    os.makedirs(result_dir, exist_ok=True)

    seed_everything(args.run_seed)

    config = get_config(ds, init_feat, fg_cyc, num_neurons, input_norm, nef_dp, reg_dp, hid_dims)
    n_workers = config.train.n_procs

    (train_gs, train_info), (val_gs, val_info), (test_gs, test_info) = \
        get_classification_dataset(ds, init_feat, fg_cyc, fully_connected_fg, n_jobs=0, seed=data_seed)

    train_ds = GraphDataset_Classification(train_gs, train_info['y'], train_info['m'])
    train_dl = GraphDataLoader_Classification(train_ds, num_workers=n_workers, batch_size=config.train.bs,
                               shuffle=True)  # , shuffle=args.shuffle

    val_gs = dgl.batch(val_gs).to(device)
    val_labels = val_info['y'].to(device)
    val_masks = val_info['m'].to(device)

    test_gs = dgl.batch(test_gs).to(device)
    test_labels = test_info['y'].to(device)
    test_masks = test_info['m'].to(device)

    task_pos_weights = train_info['w']

    model = Net(val_labels.shape[1],
                 node_in_dim=2,
                 edge_in_dim=2,
                 fg_in_dim=2,
                 **config.model).to(device)
    opt = AdamP(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=config.train.T_0)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=task_pos_weights.to(device),
                                           reduction='none')

    iters = len(train_dl)
    best_val_auroc = 0
    for epoch in range(config.train.epochs):
        print("-------------------Epoch {}-------------------".format(epoch))

        for i, (gs, labels, masks) in enumerate(train_dl):
            gs = gs.to(device)
            labels = labels.to(device).float()
            masks = masks.to(device)

            af = gs.nodes['atom'].data['feat']
            bf = gs.edges[('atom', 'interacts', 'atom')].data['feat']
            ff = gs.nodes['func_group'].data['feat']

            logits = model(gs, af, bf, ff)
            loss = (criterion(logits, labels) * masks).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step(epoch + i / iters)

        with torch.no_grad():
            model.eval()

            val_af = val_gs.nodes['atom'].data['feat']
            val_bf = val_gs.edges[('atom', 'interacts', 'atom')].data['feat']
            val_ff = val_gs.nodes['func_group'].data['feat']

            val_logits = model(val_gs, val_af, val_bf, val_ff)
            val_auroc = compute_auroc(val_logits, val_labels, masks=val_masks, average='macro')
            print("Val AUROC: {}".format(val_auroc))

            test_af = test_gs.nodes['atom'].data['feat']
            test_bf = test_gs.edges[('atom', 'interacts', 'atom')].data['feat']
            test_ff = test_gs.nodes['func_group'].data['feat']

            test_logits = model(test_gs, test_af, test_bf, test_ff)
            test_auroc = compute_auroc(test_logits, test_labels, masks=test_masks, average='macro')
            print("Test AUROC: {}".format(test_auroc))

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                test_auroc_at_best_val_auroc = test_auroc
                torch.save(model, f'{result_dir}/best_valid_model.pt')

            print("Test AUROC At Best Val AUROC: {}".format(test_auroc_at_best_val_auroc))

            model.train()

    with open(f'{result_dir}/test_auroc_at_best_val_auroc.txt', 'w') as fout:
        fout.write(str(test_auroc_at_best_val_auroc))


def seed_everything(run_seed):
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)

    return None


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-ds', type=str, choices=['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace'], required=True)
    p.add_argument('-data_seed', type=int, default=42)
    p.add_argument('-run_seed', type=int, required=True)
    p.add_argument('-init_feat', type=bool, default=True, help='init FG features')
    p.add_argument('-fg_cyc', type=bool, default=False, help='cycles included in func_group')
    p.add_argument('-num_neurons', type=list, default=[], help='num_neurons in MLP')
    p.add_argument('-input_norm', type=str, default='batch', help='input norm')
    p.add_argument('-nef_dp', type=float, default=0.0, help='node, edge, func_group dropout')
    p.add_argument('-reg_dp', type=float, default=0.1, help='regressor dropout')
    p.add_argument('-hid_dims', type=int, default=32, help='node, edge, fg hidden dims in Net')
    p.add_argument('-device', type=str, default='cuda:0', help='fitting device')

    args = p.parse_args()
    hps = hyper_params[args.ds]

    main(args.ds,
         args.data_seed,
         args.run_seed,
         hps.mean_fg_init,
         hps.use_cycle,
         hps.fully_connected_fg,
         args.num_neurons,
         args.input_norm,
         args.nef_dp,
         args.reg_dp,
         args.hid_dims,
         args.device)
