# Neural TTS for Indic Languages
## Status
This project is currently under development. Core components such as preprocessing, training, and inference pipelines are implemented, with ongoing improvements in model performance and output quality.
This project implements a FastSpeech2-based neural text-to-speech (TTS) system using PyTorch for generating speech in Indic languages.

## Features
- FastSpeech2-based architecture
- Phoneme-based text processing
- Training and inference pipeline
- Configurable parameters using JSON

## Tech Stack
- Python
- PyTorch
- Deep Learning

## Project Structure
- train.py → Model training
- inference.py → Speech generation
- preprocess.py → Data preprocessing
- data.py → Data loading
- config.json → Model configuration
- phoneme_vocab.json → Phoneme mapping

## Usage

 Train the model
bash
python train.py
Run inference
python inference.py

Dataset

Due to size constraints, datasets are not included. Public datasets such as LJSpeech or IndicTTS can be used.

Author

Balasubramanyam B S


---

# 2. requirements.txt (quick method)

### EASIEST WAY (recommended)

Open terminal in your project folder and run:

```bash
pip freeze > requirements.txt
Or if Manual Needed 
torch
numpy
scipy
librosa
matplotlib
streamlit
