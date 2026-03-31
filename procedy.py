import torch
import soundfile as sf
from huggingface_hub import hf_hub_download\
    
from ns3_codec import FACodec


class FAcodecAudioTokenizer:
    """
    Audio → FAcodec encoder latents
    Quantized content/prosody IDs are generated later during training.
    """

    def __init__(self, device="cpu"):
        self.device = device

        # Initialize FAcodec encoder
        self.encoder = FACodecEncoder(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        ).to(device)

        # Load pretrained encoder checkpoint
        ckpt_path = hf_hub_download(
            repo_id="amphion/naturalspeech3_facodec",
            filename="ns3_facodec_encoder.bin"
        )

        self.encoder.load_state_dict(
            torch.load(ckpt_path, map_location=device)
        )
        self.encoder.eval()

    @torch.no_grad()
    def encode(self, wav_path: str):
        """
        wav_path: path to WAV file
        returns LLM-ready dict
        """

        # Load audio safely (Windows compatible)
        wav_np, sr = sf.read(wav_path)
        wav = torch.from_numpy(wav_np).float()

        # Stereo → mono
        if wav.dim() == 2:
            wav = wav.mean(dim=1)

        # Shape: [B, C, T]
        wav = wav.unsqueeze(0).unsqueeze(0)

        # Resample to 16 kHz if needed
        if sr != 16000:
            wav = torch.nn.functional.interpolate(
                wav,
                scale_factor=16000 / sr,
                mode="linear",
                align_corners=False
            )

        wav = wav.to(self.device)

        # Encoder forward pass (latent features)
        latents = self.encoder(wav)

        return {
            # These are produced ONLY after quantization (training stage)
            "content_ids": None,
            "prosody_ids": None,

            # Encoder latent stream (used for quantization later)
            "detail_ids": latents.squeeze().tolist(),
        }


# -------------------------------------------------
# TEST RUN
# -------------------------------------------------
if __name__ == "__main__":

    tokenizer = FAcodecAudioTokenizer(device="cpu")

    wav_path = input("Enter path to wav file: ").strip()

    out = tokenizer.encode(wav_path)

    print("\nFAcodec OUTPUT (Encoder Stage)")

    print("Content IDs :", out["content_ids"])
    print("Prosody IDs :", out["prosody_ids"])
    print("Detail IDs  :", out["detail_ids"][:20], "...")
