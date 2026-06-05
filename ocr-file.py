import os, sys, base64, argparse
import ollama
import ollama_config
import filename_suffix
import fastwer
import pdb

image = sys.argv[1]
print(image)

def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

#def normalize_line_endings(input_text):

def perform_ocr_raw_api(image_path):
    parser = argparse.ArgumentParser(description="Ollama-driven task")
    config, groundtruth = ollama_config.build_config(parser, image_path)

    image_path = config['images']
    print(f"Encoding image {image_path}")
    config['images'] = [encode_image_to_base64(image_path)]

    # 2. Pass the dictionary keys as arguments using ** unpacking
    print(f"Transcribing image {image_path}")
    response = ollama.generate(**config)

    # 3. Print the result
    print(response)
    print(f"groundtruth: {groundtruth}")
    
    return response, groundtruth

ocr_response, groundtruth = perform_ocr_raw_api(image)

# show sample output
print("  OCR Sample: " + " ".join(ocr_response['response'][:40].splitlines()) + "...")

if groundtruth:
    print("do ground truth")

    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir, groundtruth['groundtruth'])

    with open(abs_file_path, "r") as f:
        ground = f.read()
        print(f"ground: {repr(ground)}")

    print(f"ocr: {repr(ocr_response['response'])}")
    
    cer = fastwer.score_sent(ocr_response['response'], ground, char_level=True)
    wer = fastwer.score_sent(ocr_response['response'], ground, char_level=False)

    print(f"CER: {cer}; WER: {wer}")

# create output directory
if not os.path.exists("output"):
    os.makedirs("output")

output_file = filename_suffix.filename_suffix(image)

# save OCR text file
with open(output_file, "w") as f:
    f.write(ocr_response['response'])
print(f"Saved as {output_file}")