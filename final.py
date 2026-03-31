import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import time
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------
# CONFIGURATION & CONSTANTS
# ------------------------------------------------
HIFIGAN_PATH = os.path.join(os.getcwd(), "hifi_gan")
sys.path.insert(0, HIFIGAN_PATH)

from models import Generator
from env import AttrDict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct" 

VOCAB_SIZE = 256
NUM_CLASSES = 256 
SAMPLE_RATE = 24000
BASE_EXPAND = 8

# ------------------------------------------------
# AUDIO UTILITIES
# ------------------------------------------------
def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio

def lowpass_filter(audio, sr=24000, cutoff=9000):
    nyq = 0.5 * sr
    norm = cutoff / nyq
    b, a = butter(5, norm, btype="low")
    return filtfilt(b, a, audio)

# ------------------------------------------------
# MODEL CLASSES
# ------------------------------------------------

class QwenTokenPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"Loading Qwen backbone ({MODEL_NAME})...")
        self.backbone = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
        self.dtype = self.backbone.dtype 

        config = self.backbone.config
        hidden = config.text_config.hidden_size if hasattr(config, "text_config") else config.hidden_size

        self.embedding = nn.Embedding(VOCAB_SIZE, hidden).to(self.dtype)
        self.prosody_head = nn.Linear(hidden, NUM_CLASSES).to(self.dtype)
        self.content_head = nn.Linear(hidden, NUM_CLASSES).to(self.dtype)
        self.residual_head = nn.Linear(hidden, NUM_CLASSES).to(self.dtype)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        outputs = self.backbone(inputs_embeds=x)
        hidden = outputs.last_hidden_state
        return self.prosody_head(hidden), self.content_head(hidden), self.residual_head(hidden)

class TokenToMel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.prosody_embed = nn.Embedding(NUM_CLASSES, 256)
        self.content_embed = nn.Embedding(NUM_CLASSES, 256)
        self.residual_embed = nn.Embedding(NUM_CLASSES, 256)
        self.mixer = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.GELU(),
            nn.Linear(512, 80)
        )
        self.to(dtype)

    def forward(self, p_idx, c_idx, r_idx):
        p_vec = self.prosody_embed(p_idx)
        c_vec = self.content_embed(c_idx)
        r_vec = self.residual_embed(r_idx)
        x = torch.cat([p_vec, c_vec, r_vec], dim=-1)
        return self.mixer(x.to(self.dtype))

# ------------------------------------------------
# PIPELINE FUNCTIONS
# ------------------------------------------------

def load_hifigan():
    print("Loading HiFiGAN decoder...")
    with open("hifigan.json") as f:
        config = AttrDict(json.load(f))
    decoder = Generator(config).to(DEVICE)
    ckpt = torch.load("hifigan.bin", map_location=DEVICE)
    decoder.load_state_dict(ckpt["generator"] if "generator" in ckpt else ckpt)
    decoder.eval()
    return decoder

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = QwenTokenPredictor().to(DEVICE).eval()
    projector = TokenToMel(dtype=model.dtype).to(DEVICE).eval()
    decoder = load_hifigan()

    ckpt_path = r"E:\APPLE\neural_tts_poc\models\checkpoint_qwen3_epoch1.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading weights: {ckpt_path}")
        full_ckpt = torch.load(ckpt_path, map_location=DEVICE)
        sd = full_ckpt.get("state_dict", full_ckpt)
        model.load_state_dict(sd, strict=False)
        projector.load_state_dict(sd, strict=False)
        print("Checkpoint loaded.")
    else:
        print("No checkpoint found.")
        return

    time.sleep(0.5)
    text = input("\nEnter text: ").strip()
    if not text: return
    
    tokens = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE) % VOCAB_SIZE

    with torch.no_grad():
        prosody, content, residual = model(tokens)
        p_idx = torch.argmax(prosody, dim=-1)
        c_idx = torch.argmax(content, dim=-1)
        r_idx = torch.argmax(residual, dim=-1)

        # 1. Generate Mel
        mel = projector(p_idx, c_idx, r_idx)
        
        # --- DEBUG MONITOR ---
        # Convert to float for stats to avoid BFloat16 math issues in numpy
        mel_float = mel.float()
        mel_mean = mel_float.mean().item()
        mel_std = mel_float.std().item()
        print(f"\n--- MEL STATS ---")
        print(f"Mean Intensity: {mel_mean:.4f} | Std Dev: {mel_std:.4f}")
        print(f"Indices: P={p_idx[0,0].item()}, C={c_idx[0,0].item()}")
        print("-----------------\n")

        # --- VISUALIZATION ---
        # FIX: .float() added here to prevent ScalarType BFloat16 error
        plt.figure(figsize=(12, 4))
        plt.imshow(mel_float[0].cpu().numpy().T, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"Spectrogram for: {text}")
        plt.xlabel("Time (Frames)")
        plt.ylabel("Mel Bin")
        plt.colorbar()
        plt.savefig("spectrogram.png")
        plt.close()
        print("Visual spectrogram saved to 'spectrogram.png'")

        # 2. Synthesis
        mel_upsampled = mel_float.repeat_interleave(BASE_EXPAND, dim=1).transpose(1, 2)
        wav = decoder(mel_upsampled)

    wav_np = normalize_audio(wav.squeeze().cpu().numpy())
    sf.write("tts_output.wav", lowpass_filter(wav_np), SAMPLE_RATE)
    print("Audio saved to tts_output.wav")

if __name__ == "__main__":
    main()