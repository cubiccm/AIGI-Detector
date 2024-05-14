import torch
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.load_state_dict(torch.load("model_checkpoint/model.pt")['model_state_dict'])

def predict(image):
  image = preprocess(Image.open(image)).unsqueeze(0).to(device)
  labels = [
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
  text = clip.tokenize(labels).to(device)

  with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    answer = labels[np.argmax(probs)]
    real_prob = probs[0]

  # print("Label probs:", probs)
  # print(answer)

  return answer, probs[0][0]


if __name__ == "__main__":
  print(predict("/mnt/c/Users/Limos/Desktop/photo.jpg"))