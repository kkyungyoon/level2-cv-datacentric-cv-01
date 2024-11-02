import json
import os.path as osp
from sklearn.model_selection import KFold
import argparse
import glob

def split_and_save_kfold(data, image_list, receipt_dir, folds, seed):
    """KFold split and save JSON files for each receipt directory."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_list)):
        train_images = {k: v for i, (k, v) in enumerate(image_list) if i in train_idx}
        val_images = {k: v for i, (k, v) in enumerate(image_list) if i in val_idx}

        # Save train and validation splits in respective receipt directories
        save_json(train_images, osp.join(receipt_dir, f'ufo/train{fold}.json'))
        save_json(val_images, osp.join(receipt_dir, f'ufo/valid{fold}.json'))

        print(f"Receipt: {receipt_dir}, Fold {fold} - Train images: {len(train_images)}, Validation images: {len(val_images)}")

def save_json(data, file_path):
    """Save dictionary as a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump({'images': data}, file, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="KFold split of image data")
    parser.add_argument('--seed', type=int, default=7, help="Random seed for KFold")
    parser.add_argument('--folds', type=int, default=5, help="Number of folds for KFold")
    parser.add_argument('--root_dir', type=str, default='data', help="Root directory of the dataset")
    parser.add_argument('--json_pattern', type=str, default='ufo/train.json', help="Pattern for JSON file path")

    args = parser.parse_args()

    # Find all directories matching the *_receipt pattern
    receipt_dirs = glob.glob(osp.join(args.root_dir, '*_receipt'))
    if not receipt_dirs:
        raise FileNotFoundError("No receipt directories found matching the pattern")
    
    for receipt_dir in receipt_dirs:
        json_file = osp.join(receipt_dir, args.json_pattern)
        if not osp.exists(json_file):
            print(f"No JSON file found in {receipt_dir}, skipping...")
            continue

        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        image_list = list(data['images'].items())

        split_and_save_kfold(data, image_list, receipt_dir, args.folds, args.seed)

if __name__ == "__main__":
    main()
