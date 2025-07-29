import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from torch.nn.functional import adaptive_avg_pool2d
from einops import rearrange, repeat
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import os
import json


class EvalImageFolder(Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        self.folder = folder
        self.paths = [p for ext in exts for p in Path(folder).glob(f'**/*.{ext}')]
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def get_inception_features(images, model, device, channels=3):
    if channels == 1:
        images = repeat(images, "b 1 ... -> b c ...", c=3)
    model.eval()
    with torch.no_grad():
        features = model(images)[0]
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
    return features


def calculate_precision_recall(real_features, fake_features, k=3):
    real_features = real_features.cpu()
    fake_features = fake_features.cpu()
    real_dists = torch.cdist(real_features, real_features, p=2)
    real_dists.fill_diagonal_(float('inf'))
    kth_dists, _ = real_dists.kthvalue(k, dim=1)
    tau = kth_dists.median().item()

    dists_fake2real = torch.cdist(fake_features, real_features, p=2)
    min_dists_fake, _ = dists_fake2real.min(dim=1)
    precision = (min_dists_fake < tau).float().mean().item()

    dists_real2fake = torch.cdist(real_features, fake_features, p=2)
    min_dists_real, _ = dists_real2fake.min(dim=1)
    recall = (min_dists_real < tau).float().mean().item()

    return precision, recall


def extract_features(folder, model, device, image_size=128, batch_size=50, num_samples=50000):
    dataset = EvalImageFolder(folder, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Extracting features from {folder}'):
            batch = batch.to(device)
            feats = get_inception_features(batch, model, device)
            features.append(feats)
            if len(torch.cat(features)) >= num_samples:
                break
    return torch.cat(features)[:num_samples]


def evaluate_folders(real_folder, fake_folder, image_size=128, batch_size=50, num_samples=50000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)

    real_feats = extract_features(real_folder, inception, device, image_size, batch_size, num_samples)
    fake_feats = extract_features(fake_folder, inception, device, image_size, batch_size, num_samples)

    mu_real = real_feats.cpu().numpy().mean(axis=0)
    sigma_real = np.cov(real_feats.cpu().numpy(), rowvar=False)
    mu_fake = fake_feats.cpu().numpy().mean(axis=0)
    sigma_fake = np.cov(fake_feats.cpu().numpy(), rowvar=False)

    fid = calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
    precision, recall = calculate_precision_recall(real_feats, fake_feats, k=3)

    results = {
        "fid": fid,
        "precision": precision,
        "recall": recall
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_folder', type=str, default='../data/celeba_hq_256', help='Path to folder with real images')
    parser.add_argument('--fake_folder', type=str, default='./image_samples_ours', help='Path to folder with generated images')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=30000)
    parser.add_argument('--output_path', type=str, default='results/ours.json')
    args = parser.parse_args()

    results = evaluate_folders(args.real_folder, args.fake_folder, args.image_size, args.batch_size, args.num_samples)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print("\nEvaluation complete:")
    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    main()