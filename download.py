import torch

mel = torch.load(r"E:\APPLE\neural_tts_poc\server\mels\LJ001-0001.pt")

print(type(mel))
print(mel.shape)