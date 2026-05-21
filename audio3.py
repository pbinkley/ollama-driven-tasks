import os
import sys
import platform
import numpy as np
import torch
import torchaudio
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

def get_transcription_provider(model_size="small"):
    """
    Detects operating system and initializes the absolute fastest hardware acceleration layer.
    Both paths normalize output to return uniform phrase objects with clean text + timestamp metadata.
    """
    system_platform = platform.system()
    machine_arch = platform.processor()
    
    # --- PATHWAY A: APPLE SILICON (M1/M2/M3/M4 MACS) ---
    if system_platform == "Darwin" and machine_arch == "arm":
        try:
            import mlx_whisper
            print("🚀 M-Series Mac Detected! Initializing Apple MLX Framework (Neural Engine Acceleration)...")
            
            def mlx_provider(audio_path):
                hf_model = f"mlx-community/whisper-{model_size}-mlx"
                result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=hf_model, word_timestamps=True)
                
                normalized_segments = []
                for segment in result.get("segments", []):
                    # We process phrases rather than single words to give the embedding model stable context
                    normalized_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"]
                    })
                return normalized_segments
            return mlx_provider
        except ImportError:
            print("⚠️ Mac architecture detected but 'mlx-whisper' package missing. Using CPU fallback...")

    # --- PATHWAY B: WINDOWS / LINUX (NVIDIA CUDA & INTEL/AMD CPU) ---
    try:
        from faster_whisper import WhisperModel
        
        if torch.cuda.is_available():
            print("🔥 NVIDIA GPU Detected! Initializing faster-whisper [CUDA/float16]...")
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
        else:
            print("💻 Standard Linux/Windows Architecture. Initializing thread-optimized [CPU/int8]...")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            
        def faster_whisper_provider(audio_path):
            segments, _ = model.transcribe(audio_path, vad_filter=True)
            normalized_segments = []
            for segment in segments:
                normalized_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                })
            return normalized_segments
        return faster_whisper_provider

    except ImportError:
        print("❌ Core framework modules not found. Ensure dependencies are satisfied.")
        sys.exit(1)


def process_audio_pipeline(audio_path, num_speakers=2):
    # Automatically switch speech-to-text engines based on machine profile
    transcribe_engine = get_transcription_provider(model_size="small")
    
# Check execution runtime for the Neural Voiceprint Extractor
    if torch.backends.mps.is_available():
        voice_device = "mps"
    elif torch.cuda.is_available():
        voice_device = "cuda"
    else:
        voice_device = "cpu"
        
    print(f"🎙️ Initializing SpeechBrain Neural Voiceprint Extractor...")
    
    if voice_device == "cuda":
        # Linux / Windows NVIDIA pathways use standard configuration
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": "cuda"}
        )
    elif voice_device == "mps":
        # Mac M-Series fix: Initialize safely on CPU, then cast to Metal
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )
        classifier.to("mps")
        classifier.device = torch.device("mps") # Force internal parameter tag matching
    else:
        # standard fallback
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )

    print("-> Step 1: Running optimal Whisper model extraction...")
    segments = transcribe_engine(audio_path)
    
    if not segments:
        print("❌ Failure: No distinct spoken dialogue localized in target file.")
        return

    print("-> Step 2: Loading structural audio for waveform fingerprint extraction...")
    signal, fs = torchaudio.load(audio_path)
    # Downmix multi-channel stereo down to an algorithmic mono channel if needed
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    embeddings = []
    valid_segments = []

    print("-> Step 3: Isolation and feature map extraction of speaker vocal profiles...")
    for seg in segments:
        start_sample = int(seg["start"] * fs)
        end_sample = int(seg["end"] * fs)
        
        # Audio chunks must span at least 0.1 seconds to draw a stable vector profile
        if (end_sample - start_sample) < (fs * 0.1):
            continue
            
        audio_chunk = signal[:, start_sample:end_sample]
        
        with torch.no_grad():
            # Neural network scans vocal formants to register an absolute identifier embedding
            embedding = classifier.encode_batch(audio_chunk)
            embedding = embedding.squeeze().cpu().numpy()
            
        embeddings.append(embedding)
        valid_segments.append(seg)

    if not embeddings:
        print("❌ Failure: Couldn't extract uniform vocal clusters. File may contain only noise.")
        return

    print("-> Step 4: Clustering acoustic voiceprints (Cosine Distance matching)...")
    embeddings = np.array(embeddings)
    
    # Mathematical linkage tracking uses Cosine Similarity (voice tone analysis over time position)
    clusterer = AgglomerativeClustering(n_clusters=num_speakers, metric="cosine", linkage="average")
    speaker_labels = clusterer.fit_predict(embeddings)

    # --- Print Structured Transcript Output ---
    print("\n" + "="*60 + "\n          TRUE ACOUSTICALLY DIARIZED TRANSCRIPT\n" + "="*60 + "\n")
    
    for seg, speaker_id in zip(valid_segments, speaker_labels):
        print(f"[Speaker {speaker_id}] ({seg['start']:.2f}s - {seg['end']:.2f}s): {seg['text'].strip()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python acoustic_unified_transcribe.py <path_to_audio_file> [num_speakers]")
        sys.exit(1)
        
    target_audio = sys.argv[1]
    speaker_count = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    if not os.path.exists(target_audio):
        print(f"❌ Target path '{target_audio}' points to an invalid local directory.")
        sys.exit(1)
        
    process_audio_pipeline(target_audio, num_speakers=speaker_count)
