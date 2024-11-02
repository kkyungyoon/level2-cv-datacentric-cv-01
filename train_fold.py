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

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
import time
import wandb
import uuid
import json

from deteval import calc_deteval_metrics
from utils import get_gt_bboxes, get_pred_bboxes, seed_everything, AverageMeter


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)

    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument("--wandb_name", type=str, default="default_run_name")
    parser.add_argument('--num_folds', type=int, default=5)
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, val_interval, wandb_name, num_folds):

     # wandb init은 함수 외부에서 한 번만 실행
    if wandb_name == "default_run_name":
        wandb_name = f"run_{uuid.uuid4().hex[:8]}"
    
    wandb.init(
        entity="tayoung1005-aitech",
        project="p3_ocr",
        name=wandb_name,
        config={
            "learning_rate": learning_rate,
            "epochs": max_epoch,
            "batch_size": batch_size,
            "input_size": input_size,
            "image_size": image_size,
            "num_folds": num_folds
        }
    )
    global_step = 0

    for fold in range(num_folds):  # fold 전체를 순회
        print(f"Training for Fold {fold+1}/{num_folds}")

        ##### DataLoader ##### 
        train_dataset = SceneTextDataset(
            data_dir,
            split='train',
            image_size=image_size,
            crop_size=input_size,
            num_fold=fold  # fold 변수를 추가
        )
        train_dataset = EASTDataset(train_dataset)
        num_batches = math.ceil(len(train_dataset) / batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        val_dataset = SceneTextDataset(
            data_dir,
            split='valid',
            image_size=image_size,
            crop_size=input_size,
            num_fold=fold  # fold 변수를 추가
        )
        val_dataset = EASTDataset(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = EAST()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

        ##### Train #####
        train_loss = AverageMeter()
        val_loss = AverageMeter() 
        model.train()
        min_f1_score = 2
        for epoch in range(max_epoch):
            epoch_loss, epoch_start = 0, time.time()
            train_loss.reset()
            global_step += 1 
            with tqdm(total=num_batches) as pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                    start_time = time.time()
                    
                
                    pbar.set_description(f'[Fold {fold} | Epoch {epoch + 1}]')

                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss.update(loss.item())

                    pbar.update(1)
                    # train_dict = {
                    #     f'fold_{fold}/train total loss': train_loss.avg,
                    #     f'fold_{fold}/train cls loss': extra_info['cls_loss'],
                    #     f'fold_{fold}/train angle loss': extra_info['angle_loss'],
                    #     f'fold_{fold}/train iou loss': extra_info['iou_loss']
                    # }
                    train_dict = {
                        'train total loss': train_loss.avg,
                        'train cls loss': extra_info['cls_loss'],
                        'train angle loss': extra_info['angle_loss'],
                        'train iou loss': extra_info['iou_loss']
                    }
                    pbar.set_postfix(train_dict)
                    wandb.log(train_dict, step=global_step)

                    #end_time = time.time()
                    #print("Execution Time:", end_time - start_time, "seconds")

            scheduler.step()
            ##### Validation #####
            if (epoch + 1) % val_interval == 0 or epoch >= max_epoch - 5:
                with torch.no_grad():
                    print('Validation Result ....')
                    
                    receipt_dirs = [osp.join('data',d) for d in os.listdir(data_dir) if d.endswith('_receipt')]
                    # receipt_dirs = [osp(d,'ufo') for d in os.listdir(data_dir) if d.endswith('_receipt')]
                    
                    for receipt_dir in receipt_dirs:
                        valid_json_file = osp.join(receipt_dir, 'ufo',f'valid{fold}.json')
                        # valid_json_path = osp.join(data_dir, valid_json_file)
                        print(f'valid_json_file = {valid_json_file}')
                        # print(f'valid_json_path = {valid_json_path}')
                        
                        if not osp.exists(valid_json_file):
                            print(f'{valid_json_file} does not exist, skipping...')
                            continue

                        with open(valid_json_file, 'r', encoding='utf-8') as file:
                            data = json.load(file)
                        valid_images = list(data['images'].keys())

                        pred_bboxes_dict = get_pred_bboxes(model, receipt_dir, valid_images, input_size, batch_size, split='train')
                        gt_bboxes_dict = get_gt_bboxes(receipt_dir, json_file=valid_json_file, valid_images=valid_images)

                        result = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)
                        total_result = result['total']
                        precision, recall = total_result['precision'], total_result['recall']
                        f1_score = 2 * precision * recall / (precision + recall)
                        print(f'Results for {receipt_dir} - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

                        val_dict = {
                            f'{receipt_dir}/val precision': precision,
                            f'{receipt_dir}/val recall': recall,
                            f'{receipt_dir}/val f1_score': f1_score
                        }
                        wandb.log(val_dict, step=global_step)

                    elapsed_time = time.time() - epoch_start
                    estimated_time_left = elapsed_time / (epoch + 1) * (max_epoch - epoch - 1)

                    eta = str(timedelta(seconds=estimated_time_left))
                    print(f'Epoch {epoch + 1} Validation Finished. Elapsed time: {elapsed_time} | Left ETA: {eta}')
                    

            print('Mean loss: {:.4f} | Elapsed time: {}'.format(
                epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

            if (epoch + 1) % save_interval == 0:
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)

                ckpt_fpath = osp.join(model_dir, f'latest_fold{fold}.pth')
                torch.save(model.state_dict(), ckpt_fpath)

    wandb.finish()

def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)