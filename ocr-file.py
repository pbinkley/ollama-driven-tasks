import os, sys, base64
from datetime import datetime
from pathlib import Path
import ollama

image = sys.argv[1]

VISION_MODEL = "qwen3-vl:30b-a3b-instruct"
DEFAULT_PROMPT = "Transcribe this text. You must output it in basic Markdown format. Do not translate it; transcribe it exactly. Keep the original line breaks within paragraphs. Render lists as Markdown lists. Ignore the running heads but include the page numbers."
DEFAULT_SYSTEM_PROMPT = 'You are a careful reader. When transcribing you always generate output in Markdown format, with careful attention to formatting.'

def perform_ocr_raw_api(image_path, model_name=VISION_MODEL):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    response = ollama.chat(
        model = model_name,
        messages=[
            { 
              'role': 'system',
              'content': DEFAULT_SYSTEM_PROMPT
            },
            {
              'role': 'user',
              'content': DEFAULT_PROMPT,
              'images': [base64_image]
          }]
        ) 
    
    return response

ocr_response = perform_ocr_raw_api(
    image, 
    model_name=VISION_MODEL
)

# show sample output
print("  " + " ".join(ocr_response.message.content[:40].splitlines()) + "...")

if not os.path.exists("output"):
    os.makedirs("output")

image_file = Path(image).name

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

output_file = f"output/{image_file}_{timestamp}.markdown"
with open(output_file, "w") as f:
  f.write(ocr_response.message.content)
print(f"Saved as {output_file}")