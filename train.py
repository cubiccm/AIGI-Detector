import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from PIL import Image
import os
from tqdm import tqdm

datasets = [
  ("./dataset/train/real", "a real image with no alterations"),
  ("./dataset/train/ADM", "a fake image from ablated diffusion"),
  ("./dataset/train/DDPM", "a fake image from denoising diffusion"),
  ("./dataset/train/PNDM", "a fake image from psuedo numerical diffusion"),
  ("./dataset/train/IDDPM", "a fake image from improved denoising diffusion"),
  ("./dataset/train/LDM", "a fake image from latent diffusion"),
  ("./dataset/train/ProjectedGAN", "a fake image from original ProjectedGAN"),
  ("./dataset/train/StyleGAN", "a fake image from original StyleGan"),
  ("./dataset/train/ProGAN", "a fake image from ProGAN"),
  ("./dataset/train/Diff-ProjectedGAN", "a fake image from Diff-ProjectedGAN"),
  ("./dataset/train/Diff-StyleGAN2", "a fake image from Diff-StyleGAN2"),
]

print("Initializing model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

class image_dataset():
  def __init__(self, datasets):
    self.filepath = []
    self.labels = []
    for path, label in datasets:
      tokenized_label = clip.tokenize(label)[0]
      for root, dirs, files in os.walk(path):
        for file in files:
          if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
            self.filepath.append(os.path.join(root, file))
            self.labels.append(tokenized_label)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    image = preprocess(Image.open(self.filepath[idx]))
    label = self.labels[idx]
    return image, label

def convert_models_to_fp32(model): 
  for p in model.parameters(): 
    p.data = p.data.float() 
    if p.requires_grad:
      p.grad.data = p.grad.data.float() 

print("Loading train data...")

num_epochs = 16
batch_size = 16
train_dataloader = DataLoader(image_dataset(datasets), batch_size=batch_size, shuffle=True)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=1e-4)

for epoch in range(num_epochs):
  pbar = tqdm(train_dataloader, total=len(train_dataloader))
  for batch in pbar:
    optimizer.zero_grad()

    images, labels = batch
    
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    logits_per_image, logits_per_text = model(images, labels)

    # Compute loss
    ground_truth = torch.arange(len(images), dtype=torch.long,device=device)
    total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth)) / 2

    # Backward pass
    total_loss.backward()
    if device == "cpu":
      optimizer.step()
    else: 
      convert_models_to_fp32(model)
      optimizer.step()
      clip.model.convert_weights(model)

    pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': total_loss,
  }, f"model_checkpoint/model.pt")