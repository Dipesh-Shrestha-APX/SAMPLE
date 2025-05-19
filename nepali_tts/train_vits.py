import os
import sys
import torch
import numpy as np
import soundfile as sf
import argparse
from pathlib import Path
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import load_tts_samples
from TTS.trainer import Trainer, TrainingArgs
from TTS.utils.io import load_config
import librosa
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch
import unicodedata
import re
import logging
import gradio as gr
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("output/training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def normalize_nepali_text(text):
    """Normalize Nepali text for TTS."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[.,!?;]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    def number_to_nepali(num):
        nepali_digits = "०१२३४५६७८९"
        return "".join(nepali_digits[int(d)] for d in str(num))
    
    text = re.sub(r"\d+", lambda m: number_to_nepali(m.group()), text)
    return text

def preprocess_audio(audio_path, sample_rate=16000):
    """Preprocess audio file: normalize and trim silence."""
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    return audio, sr

def validate_dataset(dataset_path, metadata_file):
    """Validate dataset and preprocess audio."""
    metadata_path = os.path.join(dataset_path, metadata_file)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file {metadata_path} not found.")
    
    audio_dir = os.path.join(dataset_path, "wavs")
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory {audio_dir} not found.")
    
    speakers = set()
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        parts = line.strip().split('|')
        if len(parts) not in [2, 3]:
            raise ValueError(f"Invalid metadata format at line {i}: {line}")
        audio_path, transcript = parts[:2]
        speaker = parts[2] if len(parts) == 3 else "default"
        speakers.add(speaker)
        
        full_audio_path = os.path.join(dataset_path, audio_path)
        if not os.path.exists(full_audio_path):
            raise FileNotFoundError(f"Audio file {full_audio_path} not found at line {i}")
        if not transcript:
            raise ValueError(f"Empty transcript for audio {audio_path} at line {i}")
        
        audio, sr = preprocess_audio(full_audio_path, sample_rate=16000)
        sf.write(full_audio_path, audio, sr)
    
    logger.info(f"Dataset validated: {len(lines)} samples, {len(speakers)} speakers found.")
    return list(speakers)

def create_data_augmentation():
    """Create audio augmentation pipeline."""
    return Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3)
    ])

def create_vits_config(dataset_path, output_path, speakers, pretrained_model=None):
    """Create VITS configuration for Nepali TTS with 16 kHz audio."""
    config = VitsConfig(
        output_path=output_path,
        run_name="nepali_vits_16khz",
        audio={
            "sample_rate": 16000,
            "win_length": 1024,
            "hop_length": 256,
            "num_mels": 80,
            "mel_fmin": 0,
            "mel_fmax": 8000
        },
        use_phonemes=False,
        text_cleaner=None,
        characters={
            "characters": (
                "ँंःअआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञ"
                "ऀऄऺऻ़ऽािीुूृेैोौ्"
                "०१२३४५६७८९"
            ),
            "punctuations": "।,?!;:'\"()",
            "pad": "<PAD>",
            "eos": "<EOS>",
            "bos": "<BOS>",
            "blank": "<BLNK>"
        },
        batch_size=8,
        eval_batch_size=4,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=0,
        epochs=2000,
        save_step=1000,
        save_n_checkpoints=5,
        save_best_after=1000,
        datasets=[{
            "name": "nepali_dataset",
            "meta_file_train": "metadata.csv",
            "path": dataset_path
        }],
        add_blank=True,
        compute_linear_spec=True,
        lr=0.0001,
        lr_scheduler="CosineAnnealingLR",
        lr_scheduler_params={"T_max": 1000, "eta_min": 1e-6},
        warmup_steps=500,
        mixed_precision=True,
        use_speaker_embedding=len(speakers) > 1,
        speaker_embedding_channels=256,
        use_language_embedding=False,
        print_step=50,
        print_eval=True,
        test_sentences=[
            ("यो एक परीक्षण वाक्य हो।", "default" if len(speakers) == 1 else speakers[0]),
            ("नेपाल सुन्दर देश हो।", "default" if len(speakers) == 1 else speakers[0])
        ]
    )
    
    if pretrained_model:
        pretrained_config = load_config(os.path.join(pretrained_model, "config.json"))
        config.model_args = pretrained_config.model_args
        config.load_json(os.path.join(pretrained_model, "config.json"))
        config.audio["sample_rate"] = 16000  # Override to match dataset
        config.audio["mel_fmax"] = 8000
    
    config.save_json(os.path.join(output_path, "config.json"))
    return config

def train_model(config, dataset_path, speakers, pretrained_model=None):
    """Train the VITS model with advanced features."""
    writer = SummaryWriter(os.path.join(config.output_path, "logs"))
    
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_size=0.1
    )
    
    audio_processor = AudioProcessor(**config.audio)
    augment = create_data_augmentation()
    
    model = Vits(config)
    if pretrained_model:
        model.load_checkpoint(config, os.path.join(pretrained_model, "best_model.pth"), strict=False)
        logger.info("Loaded pretrained model for fine-tuning.")
    
    trainer = Trainer(
        TrainingArgs(
            early_stopping_patience=50,
            early_stopping_monitor="val_loss",
            early_stopping_min_delta=0.001
        ),
        config,
        output_path=config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        audio_processor=audio_processor,
        data_augmentation=augment
    )
    
    trainer.fit()
    writer.close()

def synthesize_speech(model_path, text, speaker=None, output_wav="output.wav"):
    """Synthesize speech from text."""
    config = load_config(os.path.join(model_path, "config.json"))
    audio_processor = AudioProcessor(**config.audio)
    model = Vits.init_from_config(config)
    model.load_checkpoint(config, os.path.join(model_path, "best_model.pth"))
    model.eval()
    
    text = normalize_nepali_text(text)
    with torch.no_grad():
        wav = model.tts(text, speaker_id=speaker, language_id=None)
    
    wav = np.array(wav)
    sf.write(output_wav, wav, config.audio["sample_rate"])
    logger.info(f"Synthesized audio saved to { ویت

System: It looks like the code was cut off at the end. I'll complete the `train_vits.py` script, ensuring it includes the full `synthesize_speech` function and the rest of the implementation, while maintaining all the advanced features tailored for your 16 kHz Nepali dataset. The script will remain a single, self-contained file, optimized for robustness, performance, and ease of use, with support for data augmentation, multi-speaker setups, fine-tuning, and a Gradio interface. I’ll also provide detailed setup and usage instructions to ensure you can execute the project from start to finish.

### Completed Code: `train_vits.py`
```python
import os
import sys
import torch
import numpy as np
import soundfile as sf
import argparse
from pathlib import Path
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import load_tts_samples
from TTS.trainer import Trainer, TrainingArgs
from TTS.utils.io import load_config
import librosa
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch
import unicodedata
import re
import logging
import gradio as gr
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("output/training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def normalize_nepali_text(text):
    """Normalize Nepali text for TTS."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[.,!?;]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    def number_to_nepali(num):
        nepali_digits = "०१२३४५६७८९"
        return "".join(nepali_digits[int(d)] for d in str(num))
    
    text = re.sub(r"\d+", lambda m: number_to_nepali(m.group()), text)
    return text

def preprocess_audio(audio_path, sample_rate=16000):
    """Preprocess audio file: normalize and trim silence."""
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        audio = librosa.util.normalize(audio)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        return audio, sr
    except Exception as e:
        logger.error(f"Error preprocessing {audio_path}: {str(e)}")
        raise

def validate_dataset(dataset_path, metadata_file):
    """Validate dataset and preprocess audio."""
    metadata_path = os.path.join(dataset_path, metadata_file)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file {metadata_path} not found.")
    
    audio_dir = os.path.join(dataset_path, "wavs")
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory {audio_dir} not found.")
    
    speakers = set()
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        parts = line.strip().split('|')
        if len(parts) not in [2, 3]:
            raise ValueError(f"Invalid metadata format at line {i}: {line}")
        audio_path, transcript = parts[:2]
        speaker = parts[2] if len(parts) == 3 else "default"
        speakers.add(speaker)
        
        full_audio_path = os.path.join(dataset_path, audio_path)
        if not os.path.exists(full_audio_path):
            raise FileNotFoundError(f"Audio file {full_audio_path} not found at line {i}")
        if not transcript:
            raise ValueError(f"Empty transcript for audio {audio_path} at line {i}")
        
        audio, sr = preprocess_audio(full_audio_path, sample_rate=16000)
        sf.write(full_audio_path, audio, sr)
    
    logger.info(f"Dataset validated: {len(lines)} samples, {len(speakers)} speakers found.")
    return list(speakers)

def create_data_augmentation():
    """Create audio augmentation pipeline."""
    return Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3)
    ])

def create_vits_config(dataset_path, output_path, speakers,_ROBUSTNESS=0.9, pretrained_model=None):
    """Create VITS configuration for Nepali TTS with 16 kHz audio."""
    config = VitsConfig(
        output_path=output_path,
        run_name="nepali_vits_16khz",
        audio={
            "sample_rate": 16000,
            "win_length": 1024,
            "hop_length": 256,
            "num_mels": 80,
            "mel_fmin": 0,
            "mel_fmax": 8000
        },
        use_phonemes=False,
        text_cleaner=None,
        characters={
            "characters": (
                "ँंःअआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञ"
                "ऀऄऺऻ़ऽािीुूृेैोौ्"
                "०१२३४५६७८९"
            ),
            "punctuations": "।,?!;:'\"()",
            "pad": "<PAD>",
            "eos": "<EOS>",
            "bos": "<BOS>",
            "blank": "<BLNK>"
        },
        batch_size=8,
        eval_batch_size=4,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=0,
        epochs=2000,
        save_step=1000,
        save_n_checkpoints=5,
        save_best_after=1000,
        datasets=[{
            "name": "nepali_dataset",
            "meta_file_train": "metadata.csv",
            "path": dataset_path
        }],
        add_blank=True,
        compute_linear_spec=True,
        lr=0.0001,
        lr_scheduler="CosineAnnealingLR",
        lr_scheduler_params={"T_max": 1000, "eta_min": 1e-6},
        warmup_steps=500,
        mixed_precision=True,
        use_speaker_embedding=len(speakers) > 1,
        speaker_embedding_channels=256,
        use_language_embedding=False,
        print_step=50,
        print_eval=True,
        test_sentences=[
            ("यो एक परीक्षण वाक्य हो।", "default" if len(speakers) == 1 else speakers[0]),
            ("नेपाल सुन्दर देश हो।", "default" if len(speakers) == 1 else speakers[0])
        ]
    )
    
    if pretrained_model:
        pretrained_config = load_config(os.path.join(pretrained_model, "config.json"))
        config.model_args = pretrained_config.model_args
        config.load_json(os.path.join(pretrained_model, "config.json"))
        config.audio["sample_rate"] = 16000
        config.audio["mel_fmax"] = 8000
    
    config.save_json(os.path.join(output_path, "config.json"))
    return config

def train_model(config, dataset_path, speakers, pretrained_model=None):
    """Train the VITS model with advanced features."""
    try:
        writer = SummaryWriter(os.path.join(config.output_path, "logs"))
        train_samples, eval_samples = load_tts_samples(
            config.datasets,
            eval_split=True,
            eval_split_size=0.1
        )
        
        audio_processor = AudioProcessor(**config.audio)
        augment = create_data_augmentation()
        
        model = Vits(config)
        if pretrained_model:
            model.load_checkpoint(config, os.path.join(pretrained_model, "best_model.pth"), strict=False)
            logger.info("Loaded pretrained model for fine-tuning.")
        
        trainer = Trainer(
            TrainingArgs(
                early_stopping_patience=50,
                early_stopping_monitor="val_loss",
                early_stopping_min_delta=0.001
            ),
            config,
            output_path=config.output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
            audio_processor=audio_processor,
            data_augmentation=augment
        )
        
        trainer.fit()
        writer.close()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def synthesize_speech(model_path, text, speaker=None, output_wav="output.wav"):
    """Synthesize speech from text."""
    try:
        config = load_config(os.path.join(model_path, "config.json"))
        audio_processor = AudioProcessor(**config.audio)
        model = Vits.init_from_config(config)
        model.load_checkpoint(config, os.path.join(model_path, "best_model.pth"))
        model.eval()
        
        text = normalize_nepali_text(text)
        with torch.no_grad():
            wav = model.tts(text, speaker_id=speaker, language_id=None)
        
        wav = np.array(wav)
        sf.write(output_wav, wav, config.audio["sample_rate"])
        logger.info(f"Synthesized audio saved to {output_wav}")
        return output_wav
    except Exception as e:
        logger.error(f"Synthesis failed: {str(e)}")
        raise

def launch_gradio_interface(model_path, speakers):
    """Launch Gradio interface for real-time synthesis."""
    try:
        def synthesize_gradio(text, speaker):
            output_wav = f"output/temp_{np.random.randint(10000)}.wav"
            return synthesize_speech(model_path, text, speaker, output_wav)
        
        with gr.Blocks() as demo:
            gr.Markdown("# Nepali VITS TTS (16 kHz)")
            text_input = gr.Textbox(label="Enter Nepali Text", value="यो एक परीक्षण वाक्य हो।")
            speaker_input = gr.Dropdown(choices=speakers, label="Select Speaker", value=speakers[0])
            synthesize_btn = gr.Button("Synthesize")
            audio_output = gr.Audio(label="Synthesized Audio")
            
            synthesize_btn.click(
                fn=synthesize_gradio,
                inputs=[text_input, speaker_input],
                outputs=audio_output
            )
        
        demo.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        logger.error(f"Gradio interface failed: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Advanced Nepali VITS TTS for 16 kHz audio")
    parser.add_argument("--mode", choices=["train", "synthesize", "interface"], default="train",
                        help="Mode: train, synthesize, or launch Gradio interface")
    parser.add_argument("--dataset_path", default="dataset",
                        help="Path to dataset folder")
    parser.add_argument("--output_path", default="output",
                        help="Path to save model outputs")
    parser.add_argument("--pretrained_model", default=None,
                        help="Path to pretrained model for fine-tuning")
    parser.add_argument("--model_path", default=None,
                        help="Path to trained model for synthesis/interface")
    parser.add_argument("--text", default="यो एक परीक्षण वाक्य हो।",
                        help="Text to synthesize")
    parser.add_argument("--speaker", default=None,
                        help="Speaker ID for synthesis")
    parser.add_argument("--output_wav", default="output.wav",
                        help="Output WAV file path")
    
    args = parser.parse_args()
    
    try:
        os.makedirs(args.output_path, exist_ok=True)
        os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, "logs"), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, "visualizations"), exist_ok=True)
        
        speakers = validate_dataset(args.dataset_path, "metadata.csv")
        
        if args.mode == "train":
            config = create_vits_config(args.dataset_path, args.output_path, speakers, args.pretrained_model)
            train_model(config, args.dataset_path, speakers, args.pretrained_model)
        
        elif args.mode == "synthesize":
            if not args.model_path or not os.path.exists(args.model_path):
                raise ValueError("Please provide a valid model_path")
            synthesize_speech(args.model_path, args.text, args.speaker, args.output_wav)
        
        elif args.mode == "interface":
            if not args.model_path or not os.path.exists(args.model_path):
                raise ValueError("Please provide a valid model_path")
            launch_gradio_interface(args.model_path, speakers)
    
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()