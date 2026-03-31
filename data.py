import os
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------
WAV_DIR = r"E:\APPLE\neural_tts_poc\input\LJSpeech-1.1\wavs"
METADATA = r"E:\APPLE\neural_tts_poc\input\LJSpeech-1.1\metadata.csv"
SAVE_DIR = r"E:\APPLE\neural_tts_poc\input\mels"

# HiFiGAN Standard Parameters for 24kHz
SR = 24000
N_FFT = 1024
HOP_LENGTH = 256  # This determines your temporal resolution
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 0
FMAX = 8000 # Standard for human speech

os.makedirs(SAVE_DIR, exist_ok=True)

def extract_log_mel(wav_path):
    # 1. Load and resample audio
    y, _ = librosa.load(wav_path, sr=SR)
    
    # 2. Compute Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        win_length=WIN_LENGTH, 
        n_mels=N_MELS, 
        fmin=FMIN, fmax=FMAX
    )
    
    # 3. Convert to Log-scale (Decibels)
    # This matches the dynamic range expected by TTS models
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 4. Transpose and convert to Torch Tensor
    # We transpose to [Time, Channels] for standard PyTorch processing
    return torch.FloatTensor(log_mel).T

# ------------------------------------------------
# EXECUTION
# ------------------------------------------------
def main():
    # Load metadata (Column 0: ID, Column 2: Normalized Text)
    df = pd.read_csv(METADATA, sep='|', header=None, quoting=3)
    print(f"Starting extraction for {len(df)} files...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        wav_id = row[0]
        wav_path = os.path.join(WAV_DIR, f"{wav_id}.wav")
        save_path = os.path.join(SAVE_DIR, f"{wav_id}.pt")
        
        if not os.path.exists(save_path):
            try:
                mel_tensor = extract_log_mel(wav_path)
                # Saving as .pt is significantly faster for PyTorch loaders
                torch.save(mel_tensor, save_path)
            except Exception as e:
                print(f"Skipping {wav_id} due to error: {e}")

    print(f"\nSuccess! 13,100 Mels extracted to {SAVE_DIR}")

if __name__ == "__main__":
    main()