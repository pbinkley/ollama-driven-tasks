import sys, glob
from datetime import datetime
from ollama import chat

# Pass in the path to the images
dir = sys.argv[1]

PATH = f"{dir}/pngs/*.png"
VISION_MODEL = "qwen3-vl:30b-a3b-instruct"
DEFAULT_PROMPT = 'Transcribe this Portuguese text. You must output it in basic Markdown format, with footnotes. Do not translate it; transcribe it exactly.'
DEFAULT_SYSTEM_PROMPT = 'You are a careful reader. When transcribing you always generate output in Markdown format, with careful attention to formatting, including footnotes, which use the square bracket format: [^1], in the text and in the footnote. Ignore the running heads but include the page numbers.'
CONTEXT_WINDOW = 16384 

output = ""

for file in sorted(glob.iglob(PATH, recursive=True)):
  print(file)
  response = chat(
    model = VISION_MODEL,
    messages = [
      { 
        'role': 'system',
        'content': DEFAULT_SYSTEM_PROMPT
      },
      {
        'role': 'user',
        'content': DEFAULT_PROMPT,
        'images': [file],
      },
    ],
    options = {
      'num_ctx': CONTEXT_WINDOW
    }
  )

  output += f"========= {file}\n\n"
  output += response.message.content + "\n\n"
  print("  " + " ".join(response.message.content[:40].splitlines()) + "...")

#  print(output)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

with open(f"{dir}/transcription_{timestamp}.markdown", "w") as f:
  f.write(output)

print(output)