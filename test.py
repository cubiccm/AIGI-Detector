import torch
from torch.utils.data import DataLoader
import clip
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

datasets = [
  ("./dataset/val/real", "a real image with no alterations"),
  ("./dataset/val/ADM", "a fake image from ablated diffusion"),
  ("./dataset/val/DDPM", "a fake image from denoising diffusion"),
  ("./dataset/val/PNDM", "a fake image from psuedo numerical diffusion"),
  ("./dataset/val/IDDPM", "a fake image from improved denoising diffusion"),
  ("./dataset/val/LDM", "a fake image from latent diffusion"),
  ("./dataset/val/ProjectedGAN", "a fake image from original ProjectedGAN"),
  ("./dataset/val/StyleGAN", "a fake image from original StyleGan"),
  ("./dataset/val/ProGAN", "a fake image from ProGAN"),
  ("./dataset/val/Diff-ProjectedGAN", "a fake image from Diff-ProjectedGAN"),
  ("./dataset/val/Diff-StyleGAN2", "a fake image from Diff-StyleGAN2"),
]

label_map = [
  "a real image with no alterations",
  "a fake image from ablated diffusion",
  "a fake image from denoising diffusion",
  "a fake image from psuedo numerical diffusion",
  "a fake image from improved denoising diffusion",
  "a fake image from latent diffusion",
  "a fake image from original ProjectedGAN",
  "a fake image from original StyleGan",
  "a fake image from ProGAN",
  "a fake image from Diff-ProjectedGAN",
  # "a fake image from Diff-StyleGAN2",
]

print("Initializing model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.load_state_dict(torch.load("model_checkpoint/model.pt")['model_state_dict'])

class image_dataset():
  def __init__(self, datasets):
    self.filepath = []
    self.labels = []
    for path, label in datasets:
      for root, dirs, files in os.walk(path):
        for file in files:
          if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.webp'):
            self.filepath.append(os.path.join(root, file))
            self.labels.append(label)

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

batch_size = 1
test_dataloader = DataLoader(image_dataset(datasets), batch_size=batch_size, shuffle=True)

total_tested = 0
correct_count = 0
real_fake_correct_count = 0

total = 1000

pbar = tqdm(test_dataloader, total=total)
text = clip.tokenize(label_map).to(device)
for batch in pbar:
  images, labels = batch

  images = images.to(device)

  image = images[0].unsqueeze(0)
  label = labels[0]

  with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
  
  ans_index = np.argmax(probs)
  ans_label = label_map[ans_index]
  is_correct = (ans_label == label)
  is_real_fake_correct = ((ans_index == 0) == (label == label_map[0]))

  total_tested += 1
  correct_count += (is_correct == True)
  real_fake_correct_count += (is_real_fake_correct == True)

  if total_tested == total:
    break

print("Correct:", str(correct_count / total_tested))
print("Real/fake correct:", str(real_fake_correct_count / total_tested))
