import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torchvision import transforms as T
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from guided_diffusion.flaw_highlighter import FlawHighlighter
from tqdm import tqdm
import wandb
import os

class Dataset(Dataset):
    def __init__(
            self,
            folder,
            image_size,
            exts=['jpg', 'jpeg', 'png', 'tiff'],
            augment_horizontal_flip=False,
            convert_image_to=None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if convert_image_to else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class FHTrainer(object):
    def __init__(self, FH, real_img, gen_img, image_size, FH_ckpt=None, batch_size=16, lr=1e-5, adam_betas=(0.5, 0.999), num_epoch=20):
        super().__init__()

        self.FH = FH
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.adam_betas = adam_betas
        self.num_epoch = num_epoch

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FH = self.FH.to(self.device)
        print('device:', self.device)

        self.real_dataset = Dataset(real_img, self.image_size)
        self.gen_dataset = Dataset(gen_img, self.image_size)

        real_data_size = len(self.real_dataset)
        gen_data_size = len(self.gen_dataset)

        print(f"Real Dataset Size: {real_data_size}")
        print(f"Generated Dataset Size: {gen_data_size}")

        real_test_size = int(real_data_size * 0.1)
        real_train_size = real_data_size - real_test_size
        self.train_real, self.test_real = random_split(self.real_dataset, [real_train_size, real_test_size])

        gen_test_size = int(gen_data_size * 0.1)
        gen_train_size = gen_data_size - gen_test_size
        self.train_gen, self.test_gen = random_split(self.gen_dataset, [gen_train_size, gen_test_size])

        print(f"Train Real: {real_train_size}, Test Real: {real_test_size}")
        print(f"Train Generated: {gen_train_size}, Test Generated: {gen_test_size}")

        self.train_real_dl = DataLoader(self.train_real, batch_size=self.batch_size, shuffle=True)
        self.test_real_dl = DataLoader(self.test_real, batch_size=self.batch_size, shuffle=False)

        self.train_gen_dl = DataLoader(self.train_gen, batch_size=self.batch_size, shuffle=True)
        self.test_gen_dl = DataLoader(self.test_gen, batch_size=self.batch_size, shuffle=False)

        self.opt = AdamW(self.FH.parameters(), lr=self.lr, betas=self.adam_betas)

        self.scheduler = OneCycleLR(
            self.opt,
            max_lr=self.lr,
            epochs=self.num_epoch,
            steps_per_epoch=len(self.train_real_dl),
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1e4
        )

        self.best_acc = 0

        if FH_ckpt:
            self.start_epoch = self.load_checkpoint(FH_ckpt)
        else:
            self.start_epoch = 1

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        self.FH.load_state_dict(checkpoint['model_state_dict'])
        
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"Checkpoint loaded: Starting at epoch {start_epoch}")
        return start_epoch

    def train(self):
        wandb.init(project="guided-diffusion-fh")

        self.FH.train()

        for epoch in range(self.start_epoch, self.num_epoch+1):
            epoch_loss = []
            pbar = tqdm(zip(self.train_real_dl, self.train_gen_dl), total=len(self.train_real_dl), desc=f"Epoch {epoch}/{self.num_epoch}")

            for real_images, generated_images in pbar:
                real_images, generated_images = real_images.to(self.device), generated_images.to(self.device)

                real_labels = torch.ones((len(real_images), 1), device=self.device)
                fake_labels = torch.zeros((len(generated_images), 1), device=self.device)

                real_loss = F.binary_cross_entropy(self.FH(real_images), real_labels)
                fake_loss = F.binary_cross_entropy(self.FH(generated_images), fake_labels)

                total_loss = (real_loss + fake_loss) / 2

                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()
                self.scheduler.step()

                epoch_loss.append(total_loss.item())
                pbar.set_postfix({'loss': total_loss.item()})
                wandb.log({"iteration loss": total_loss})

            avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            wandb.log({"epoch loss": avg_epoch_loss})

            print(f"Epoch [{epoch}/{self.num_epoch}], Loss: {avg_epoch_loss}")

            if not os.path.exists('./fh_ckpt/'):
                os.makedirs('./fh_ckpt/')

            self.FH.eval()
            all_real_preds = []
            all_fake_preds = []
            all_real_labels = []
            all_fake_labels = []

            with torch.no_grad():
                for real_images, generated_images in zip(self.test_real_dl, self.test_gen_dl):
                    real_images, generated_images = real_images.to(self.device), generated_images.to(self.device)

                    real_preds = self.FH(real_images).squeeze().cpu().numpy()
                    fake_preds = self.FH(generated_images).squeeze().cpu().numpy()

                    real_labels = np.ones(len(real_images))
                    fake_labels = np.zeros(len(generated_images))

                    all_real_preds.extend(real_preds)
                    all_fake_preds.extend(fake_preds)
                    all_real_labels.extend(real_labels)
                    all_fake_labels.extend(fake_labels)
                    
            real_preds = np.array(all_real_preds)
            fake_preds = np.array(all_fake_preds)
            real_labels = np.array(all_real_labels)
            fake_labels = np.array(all_fake_labels)

            real_preds_rounded = np.round(real_preds)
            fake_preds_rounded = np.round(fake_preds)

            real_acc = accuracy_score(real_labels, real_preds_rounded)
            fake_acc = accuracy_score(fake_labels, fake_preds_rounded)
            avg_acc = (real_acc + fake_acc) / 2

            avg_f1 = f1_score(np.concatenate([real_labels, fake_labels]), np.concatenate([real_preds_rounded, fake_preds_rounded]))

            avg_roc_auc = roc_auc_score(np.concatenate([real_labels, fake_labels]), np.concatenate([real_preds, fake_preds]))

            if avg_acc > self.best_acc:
                self.best_acc = avg_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.FH.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': avg_epoch_loss,
                    'roc_auc': avg_roc_auc,
                    'accuracy': avg_acc,
                    'f1_score': avg_f1
                }, f'./fh_ckpt/FH_best_{epoch}.pth')
                

            wandb.log({"Accuracy": avg_acc, "F1 Score": avg_f1, "ROC AUC": avg_roc_auc})
            print(f"Epoch [{epoch}/{self.num_epoch}], Accuracy: {avg_acc:.4f}, F1 Score: {avg_f1:.4f}, ROC AUC: {avg_roc_auc:.4f}")

            self.FH.train()

if __name__ == '__main__':
    params = {
        'nc' : 3,
        'ndf' : 32,
        }

    FH = FlawHighlighter(params)

    FH_trainer = FHTrainer(FH=FH,
                        real_img='../data/celeba_hq_256/',
                        gen_img='./image_samples/',
                        image_size=128)

    FH_trainer.train()