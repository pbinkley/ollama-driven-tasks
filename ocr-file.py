import os, sys, base64, argparse
import ollama
import ollama_config
import filename_suffix
import json
from langchain_core.utils.json import parse_json_markdown

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

output = ocr_response['response']
filetype = 'md'
if output.startswith('```json'):
    filetype = 'json'
    #output = )
#    import pdb; pdb.set_trace()
#    output = re.sub(r"```json(.*?)```", r"\1", output)



# show sample output
print("  OCR Sample: " + " ".join(ocr_response['response'][:40].splitlines()) + "...")

# create output directory
if not os.path.exists("output"):
    os.makedirs("output")

output_file = filename_suffix.filename_suffix(image)
#import pdb; pdb.set_trace()

# save OCR text file
with open(output_file, "w") as f:
    #f.write(ocr_response['response'])
    json.dump(parse_json_markdown(output), f, indent=4)
    #f.write(output)
print(f"Saved as {output_file}")