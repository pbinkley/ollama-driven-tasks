import os, sys, base64, argparse
import ollama
import ollama_config
import filename_suffix

image = sys.argv[1]

def perform_ocr_raw_api(image_path):
    parser = argparse.ArgumentParser(description="Ollama-driven task")
    config = ollama_config.build_config(parser, image_path)

    # 2. Pass the dictionary keys as arguments using ** unpacking
    response = ollama.generate(**config)

    # 3. Print the result
    # print(response['response'])
    
    return response

ocr_response = perform_ocr_raw_api(image)

# show sample output
print("  OCR Sample: " + " ".join(ocr_response['response'][:40].splitlines()) + "...")

# create output directory
if not os.path.exists("output"):
    os.makedirs("output")

output_file = filename_suffix.filename_suffix(image)

# save OCR text file
with open(output_file, "w") as f:
    f.write(ocr_response['response'])
print(f"Saved as {output_file}")