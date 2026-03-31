import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# --- A100 HIGH-PERFORMANCE CONFIG ---
DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
BATCH_SIZE = 32         # A100 handles this easily
MAX_LR = 1e-3          # Increased for fast POC results
EPOCHS = 50            # Aim for 50 for a solid POC
NUM_CLASSES = 256
VOCAB_SIZE = 256

# --- DATASET ---
class TTSDataset(Dataset):
    def __init__(self, csv_path, mel_dir, tokenizer):
        self.metadata = pd.read_csv(csv_path, sep='|', header=None, quoting=3)
        self.mel_dir = mel_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        try:
            wav_id = self.metadata.iloc[idx, 0]
            text = str(self.metadata.iloc[idx, 2]) # Normalized text
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).input_ids.squeeze(0)
            mel = torch.load(os.path.join(self.mel_dir, f"{wav_id}.pt"))
            return tokens % VOCAB_SIZE, mel
        except: return None

# --- MODEL (BF16 Optimized) ---
class QwenTokenPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"Loading {MODEL_NAME} in BF16...")
        # Load backbone in BFloat16 (Native to A100)
        self.backbone = AutoModel.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(DEVICE)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        hidden = 2048
        self.prosody_head = nn.Linear(hidden, NUM_CLASSES).to(torch.bfloat16)
        self.content_head = nn.Linear(hidden, NUM_CLASSES).to(torch.bfloat16)
        self.residual_head = nn.Linear(hidden, NUM_CLASSES).to(torch.bfloat16)

    def forward(self, token_ids):
        outputs = self.backbone(input_ids=token_ids)
        hidden = outputs.last_hidden_state
        return self.prosody_head(hidden), self.content_head(hidden), self.residual_head(hidden)

class TokenToMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Linear(NUM_CLASSES * 3, 512),
            nn.GELU(),
            nn.Linear(512, 80)
        ).to(torch.bfloat16)

    def forward(self, p, c, r):
        # Concatenate softmax probabilities for robust feature mapping
        x = torch.cat([torch.softmax(p, -1), torch.softmax(c, -1), torch.softmax(r, -1)], dim=-1)
        return self.mixer(x)

# --- TRAINER ---
def main():
    os.makedirs("models", exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = QwenTokenPredictor().to(DEVICE)
    projector = TokenToMel().to(DEVICE)
    
    # Optimizer for heads only
    optimizer = torch.optim.AdamW(
        list(model.prosody_head.parameters()) + 
        list(model.content_head.parameters()) + 
        list(projector.parameters()), 
        lr=MAX_LR/25 # Initial LR
    )

    # 1. Padding Collate Function for large batches
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        tokens = nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True)
        max_mel_len = max([b[1].shape[0] for b in batch])
        mels = torch.stack([nn.functional.pad(b[1], (0, 0, 0, max_mel_len - b[1].shape[0])) for b in batch])
        return tokens, mels

    dataset = TTSDataset(r"input/LJSpeech-1.1/metadata.csv", r"input/mels", tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8)

    # 2. OneCycleLR Scheduler (The "Super-Convergence" engine)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=len(loader) * EPOCHS
    )
    
    criterion = nn.MSELoss()

    print(f"A100 Training Started with {len(loader)} steps per epoch.")
    for epoch in range(EPOCHS):
        pbar = tqdm(loader)
        for tokens, target_mel in pbar:
            tokens, target_mel = tokens.to(DEVICE), target_mel.to(DEVICE).to(torch.bfloat16)
            
            optimizer.zero_grad()
            
            # Forward
            prosody, content, residual = model(tokens)
            pred_mel = projector(prosody, content, residual)
            
            # Match lengths
            pred_mel = nn.functional.interpolate(pred_mel.transpose(1, 2), size=target_mel.shape[1]).transpose(1, 2)
            
            loss = criterion(pred_mel, target_mel)
            loss.backward()
            
            # Gradient clipping to prevent "nan" loss on high LR
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_description(f"Ep {epoch} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

        # Save Checkpoint
        torch.save({
            'model': model.state_dict(),
            'proj': projector.state_dict(),
            'epoch': epoch
        }, f"models/A100_checkpoint_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()