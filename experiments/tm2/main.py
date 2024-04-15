import os
import argparse
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch

torch.set_num_threads(3)

from src.models.tm2 import GMGNN
from src.engines.tm2_engine import TestEngine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, get_dataset_info, load_adj_from_numpy
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--init_dim', type=int, default=3)
    parser.add_argument('--out_dim', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=64)
    parser.add_argument('--node_num', type=int, default=716),
    # parser.add_argument('--input_dim', type=int, default=32),
    # parser.add_argument('--output_dim', type=int, default=32),
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--end_dim', type=int, default=512)

    parser.add_argument('--num_clusters', type=int, default=96)
    parser.add_argument('--filter_type', type=str, default='doubletransition')

    # parser.add_argument('--seq_len', type=int, default=64)
    # parser.add_argument('--horizon', type=int, default=512)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def main():
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    adj_mx = np.zeros((node_num, node_num), dtype=np.float32)
    for n in range(node_num):
        idx = np.nonzero(adj_mx[n])[0]
        adj_mx[n, idx] = 1
    adj_mx = normalize_adj_mx(adj_mx, 'scalap')[0]
    adj_mx = torch.tensor(adj_mx).to(device)

    logger.info(f'Shape of Adj_Matrix {adj_mx.shape}')

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = GMGNN(node_num=node_num,
                  input_dim=args.input_dim,
                  init_dim=args.init_dim,
                  output_dim=args.output_dim,
                  adj_mx=adj_mx,
                  h_dim=args.h_dim,
                  num_clusters=args.num_clusters,
                  dropout=args.dropout,
                  end_dim=args.end_dim,
                  layer=args.layer,
                  device=args.device,

                  )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = None

    engine = TestEngine(device=device,
                        model=model,
                        dataloader=dataloader,
                        scaler=scaler,
                        sampler=None,
                        loss_fn=loss_fn,
                        lrate=args.lrate,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        clip_grad_value=args.clip_grad_value,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        log_dir=log_dir,
                        logger=logger,
                        seed=args.seed
                        )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
