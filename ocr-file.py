import os, sys, base64, argparse
import ollama
import ollama_config # from ollama_config.py
import filename_suffix
import fastwer
import pdb

image = sys.argv[1]
print(image)

def encode_image_to_base64(image_path):
    # Encodes an image file to a base64 string.
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def perform_ocr_raw_api(config):
    pdb.set_trace()

    image_path = config[0]['images']
    print(f"Encoding image {image_path}")
    # TODO this is ending up with "-g"
    config[0]['images'] = [encode_image_to_base64(image_path)]

    # pass config keys as arguments using ** unpacking
    print(f"Transcribing image {image_path}")
    response = ollama.generate(**config)

    # show config
    print(response)
    print(f"groundtruth: {groundtruth}")
    
    return response, groundtruth

# parse config options
parser = argparse.ArgumentParser(description="Ollama-driven task")
config = ollama_config.build_config(parser, image)

if 'verbose' in config[0]:
    print(config)

ocr_response, groundtruth = perform_ocr_raw_api(config)

# show sample output
print("  OCR Sample: " + " ".join(ocr_response['response'][:40].splitlines()) + "...")

if groundtruth:
    print("do ground truth")

    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir, groundtruth['groundtruth'])

    with open(abs_file_path, "r") as f:
        ground = f.read()

    if config['v']:
        print(f"ground: {repr(ground)}")
        print(f"ocr: {repr(ocr_response['response'])}")
    
    cer = fastwer.score_sent(ocr_response['response'], ground, char_level=True)
    wer = fastwer.score_sent(ocr_response['response'], ground, char_level=False)

    print(f"CER: {cer}; WER: {wer}")

# create output directory
if not os.path.exists("output"):
    os.makedirs("output")

# save OCR text file
output_file = filename_suffix.filename_suffix(image)
with open(output_file, "w") as f:
    f.write(ocr_response['response'])
print(f"Saved as {output_file}")