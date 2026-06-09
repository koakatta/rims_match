import torch
import clip 
from PIL import Image
device="cuda" if torch.cuda.is_available() else "cpu"
model, preprocess=clip.load("ViT-B/32",device=device)
img1=preprocess(Image.open("rims/BBS_CH-R_II.png")).unsqueeze(0).to(device)
img2=preprocess(Image.open("rims/BBS_CC-R.png")).unsqueeze(0).to(device)
with torch.no_grad():
    feat1=model.encode_image(img1)
    feat2=model.encode_image(img2)
feat1=feat1/feat1.norm(dim=-1,keepdim=True)
feat2=feat2/feat2.norm(dim=-1,keepdim=True)
similarity=(feat1@feat2.T).item()
print(similarity)