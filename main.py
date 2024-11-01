import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from time import gmtime, strftime


import wandb
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl

from trainer import OCRTrainer
from data_module import DataModule

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--train_data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--output_dir', type=str, default='./output')


    parser.add_argument("--gpus", type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=150)
    
    args = parser.parse_args()


    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def setup_logger(use_wandb=True, output_dir='./output'):
    my_loggers = []
    csv_logger = CSVLogger(save_dir=output_dir, name="result")
    my_loggers.append(csv_logger)

    if use_wandb:
        wandb.init(
            project="p3_ocr",
            entity="tayoung1005-aitech",
            name=output_dir     #.replace("./result/", ""),
        )
        wandb_logger = WandbLogger(
            save_dir=output_dir,
            name=os.path.basename(output_dir),
            project="p3_ocr",
        )
        my_loggers.append(wandb_logger)

    return my_loggers

def do_training(args):
    
    
    # my_loggers = setup_logger(args.use_wandb, args.output_dir)
    my_loggers = setup_logger(output_dir=args.output_dir)

    checkpoint_callback = []
    checkpoint_callback.append(
        ModelCheckpoint(
            dirpath=args.output_dir,
            save_last=True,
            save_top_k=1,
            monitor='val_loss',
            mode='min',
        )
    )

    trainer = pl.Trainer(
        logger=my_loggers,
        accelerator="cpu" if args.gpus == 0 else "gpu",
        # precision=16 if hparams['gpus'] != 0 else 32,  # CPU에서는 32-bit precision
        devices=None if args.gpus == 0 else args.gpus,
        callbacks=checkpoint_callback,  # 콜백 리스트로 묶는 것이 좋음
        max_epochs=args.epoch,
        # accumulate_grad_batches=(1 if hparams['accumulate_grad_batches'] <= 0
        #     else hparams['accumulate_grad_batches']),
    )


    trainer_mod = OCRTrainer(args)
    data_mod = DataModule(args)
    
    trainer.fit(trainer_mod, data_mod)



def main(args):

    # data split 
    seed = 42
    torch.manual_seed(seed)

    do_training(args)

if __name__ == '__main__':
    args = parse_args()
    args.gpus = [int(i) for i in str(args.gpus).split(",")]

    name_str = "1024_size"

    current_time = strftime("%m-%d_%H:%M:%S", gmtime())

    args.output_dir = os.path.join(args.output_dir, name_str + "_" + current_time)

    main(args)