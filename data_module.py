from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from dataset import SceneTextDataset
from east_dataset import EASTDataset




class DataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()

        self.train_data_dir = args.train_data_dir
        # self.test_data_dir = args.test_data_dir
        
        self.batch_size = args.batch_size

        self.image_size= args.image_size
        self.crop_size= args.input_size
        
        self.num_workers = args.num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":

            # full_trainset = train_data(self.data_name, self.train_transform, self.train_data_dir,info_df=self.train_info_df) #TODO val,train 분리
            # self.train_dataset, self.val_dataset = random_split(
            #     full_trainset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            # )
            trainval_dataset = train_data(
                self.train_data_dir,
                image_size=self.image_size,
                crop_size=self.crop_size,
            )
            

            train_size = int(0.8 * len(trainval_dataset))
            val_size = len(trainval_dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(trainval_dataset, [train_size, val_size])


        # if stage == "predict":
        #     self.test_dataset = test_data(
        #         self.data_name,
        #         self.test_transform,
        #         self.test_data_dir,
        #         info_df=self.test_info_df,
        #     )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            # prefetch_factor = 4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            # prefetch_factor = 4
        )

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    # def predict_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #     )


def train_data(train_data_dir, image_size, crop_size):
    if True:
        train_dataset = SceneTextDataset(
                train_data_dir,
                split='train',
                image_size=image_size,
                crop_size=crop_size,)
        # train_dataset = 
        return EASTDataset(train_dataset)
        


# TODO
# def val_data()
# def val_data(
#     data_name, transforms, train_data_dir="./", info_df=None, is_inference=False
# ):
#     pass 

# def test_data(
#     data_name, transforms, test_data_dir="./", info_df=None, is_inference=True
# ):
#     if data_name == "base":
#         return CustomDataset(
#             test_data_dir, info_df, transforms, is_inference
#         )
#     elif data_name == "folder":
#         return CustomImageFolderDataset(
#             test_data_dir, transform=transforms
#         )
#     elif data_name == 'swin_data':
#         return SwinCustomDataset(
#             test_data_dir, info_df, transforms, is_inference
#         )
#     else:
#         raise ValueError("not a correct test data name", data_name)