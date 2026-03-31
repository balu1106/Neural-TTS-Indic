import torch
import os

PATH = "models/A100_POC_epoch_149.pt"

print("Loading checkpoint...")

ckpt = torch.load(PATH, map_location="cpu")

print("\nCheckpoint size:", os.path.getsize(PATH)/1024/1024, "MB")

print("\nType:", type(ckpt))

if isinstance(ckpt, dict):

    print("\nTop-level keys:\n")

    for k in ckpt.keys():
        print("-", k)

    if "model" in ckpt:
        print("\nMODEL PARAMETERS\n")
        for k,v in ckpt["model"].items():
            print(k, v.shape)

    if "proj" in ckpt:
        print("\nPROJECTOR PARAMETERS\n")
        for k,v in ckpt["proj"].items():
            print(k, v.shape)