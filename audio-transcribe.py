import os
import sys
from faster_whisper import WhisperModel
import ollama

def transcribe_and_process(audio_path, ollama_model="llama3.2"):
    """
    Cross-platform function to transcribe a WAV file and refine it with Ollama.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return None

    print(f"--- Phase 1: Transcribing Audio using Whisper ---")
    # 'base' strikes an exceptional cross-platform balance between speed and accuracy.
    # It dynamically utilizes system hardware (Apple Silicon, CUDA, or basic CPU)
    try:
        #model_size = "base"
        #model_size = "large-v3"
        model_size = "turbo"
        # model = WhisperModel(model_size, device="auto", compute_type="float32")
        model = WhisperModel(model_size, device="auto", compute_type="int8")
        
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        # Combine transcribed fragments into a single cohesive string
        raw_text = ""
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            raw_text += segment.text + " "
            
    except Exception as e:
        print(f"Transcription failed: {e}")
        return None

    print(f"\n--- Phase 2: Refining Transcript with Ollama ({ollama_model}) ---")
    try:
#        system_prompt = (
#            "You are an expert editor. Clean up the following raw audio transcript. "
#            "Fix grammar errors, repair spelling mistakes, and remove verbal fillers "
#            "like 'um', 'uh', or 'like', but preserve the original meaning completely."
#        )

        system_prompt = (
            "You are an expert transcriber. Clean up the following raw audio transcript. "
            "Remove verbal fillers like 'um', 'uh', or 'like', and add punctuation, "
            "but otherwise preserve the original text exactly and completely."
        )
        
        response = ollama.chat(
            model=ollama_model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': raw_text}
            ]
        )
        return response['message']['content']
        
    except Exception as e:
        print(f"Ollama processing failed: {e}")
        print("Returning raw transcript instead.")
        return raw_text

if __name__ == "__main__":
    # Test file path (Replace with your actual .wav file location)
    # audio_file = "/Users/pbinkley/Documents/Projects/llm/gradio/paul-vs-george.mp3" 
    # audio_file = "/Users/pbinkley/Downloads/Entretien recherche sur le Patrimoine - 2026_05_12 09_58 MDT - Recording.mp3"
    audio_file = "optimal-output.mp3"

    refined_transcript = transcribe_and_process(audio_file, ollama_model="llama3.2")
    
    if refined_transcript:
        print("\n--- Final Polished Transcript ---")
        print(refined_transcript)
