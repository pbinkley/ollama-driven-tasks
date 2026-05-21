import os
import sys
import platform
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def get_transcription_provider(model_size="large"):
    """
    Detects hardware architecture and selects the absolute fastest 
    zero-config backend available for that OS.
    """
    system_platform = platform.system()
    machine_arch = platform.processor()
    
    # --- 1. APPLE SILICON MAXIMIZATION (M1/M2/M3/M4) ---
    if system_platform == "Darwin" and machine_arch == "arm":
        try:
            import mlx_whisper
            print(f"🚀 M-Series Mac Detected! Routing to native Apple MLX Framework...")
            
            def mlx_provider(audio_path):
                # Map standard short name to the MLX community HuggingFace weights
                hf_model = f"mlx-community/whisper-{model_size}-mlx"
                result = mlx_whisper.transcribe(
                    audio_path,
                    path_or_hf_repo=hf_model,
                    word_timestamps=True
                )
                
                # Normalize MLX output to a unified word dictionary format
                normalized_words = []
                for segment in result.get("segments", []):
                    for word in segment.get("words", []):
                        normalized_words.append({
                            "start": word["start"],
                            "end": word["end"],
                            "text": word["word"]
                        })
                return normalized_words
                
            return mlx_provider
            
        except ImportError:
            print("⚠️ Mac architecture detected but 'mlx-whisper' package is missing.")
            print("-> Proceeding with standard cross-platform fallback engine...")

    # --- 2. WINDOWS / LINUX / CPU FALLBACK ENGINE ---
    try:
        import torch
        from faster_whisper import WhisperModel
        
        if torch.cuda.is_available():
            print(f"🔥 NVIDIA GPU Detected! Routing to faster-whisper [CUDA/float16]...")
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
        else:
            print(f"💻 Standard CPU Detected. Routing to faster-whisper [CPU/int8]...")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            
        def faster_whisper_provider(audio_path):
            segments, _ = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)
            
            normalized_words = []
            for segment in segments:
                if segment.words:
                    for word in segment.words:
                        normalized_words.append({
                            "start": word.start,
                            "end": word.end,
                            "text": word.word
                        })
            return normalized_words
            
        return faster_whisper_provider

    except ImportError:
        print("❌ Error: Core speech-to-text libraries are missing. Please run:")
        print("   pip install faster-whisper torch")
        sys.exit(1)


def process_audio(audio_path, num_speakers=None):
    # Fetch the optimal architecture engine dynamically
    transcribe_engine = get_transcription_provider(model_size="small")
    
    print("-> Processing audio and extracting word-level timestamps...")
    words = transcribe_engine(audio_path)
    
    if not words:
        print("❌ Error: No speech detected in the audio file.")
        return

    print("-> Calculating speaker clusters (Diarization)...")
    # Generate temporal midpoints for each word to run spatial distance clustering
    midpoints = np.array([[(w["start"] + w["end"]) / 2] for w in words])
    
    # Fallback to smart distance clustering if exact speaker count is unknown
    if num_speakers is None:
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=3.0)
    else:
        clustering = AgglomerativeClustering(n_clusters=num_speakers)
        
    speaker_labels = clustering.fit_predict(midpoints)
    
    # --- Format & Print Streamlined Output ---
    print("\n" + "="*40 + "\n   FINAL DIARIZED TRANSCRIPT\n" + "="*40 + "\n")
    
    current_speaker = speaker_labels[0]
    current_turn_text = []
    start_time = words[0]["start"]
    
    for word, label in zip(words, speaker_labels):
        # When speaker label changes, flush out the accumulated block of dialogue
        if label != current_speaker:
            print(f"[Speaker {current_speaker}] ({start_time:.2f}s): {''.join(current_turn_text).strip()}")
            current_speaker = label
            start_time = word["start"]
            current_turn_text = [word["text"]]
        else:
            current_turn_text.append(word["text"])
            
    # Print trailing block
    if current_turn_text:
        print(f"[Speaker {current_speaker}] ({start_time:.2f}s): {''.join(current_turn_text).strip()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python unified_transcribe.py <path_to_audio_file> [num_speakers]")
        sys.exit(1)
        
    target_file = sys.argv[1]
    expected_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not os.path.exists(target_file):
        print(f"❌ Error: Audio file '{target_file}' could not be located.")
        sys.exit(1)
        
    process_audio(target_file, num_speakers=expected_speakers)