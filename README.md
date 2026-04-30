# ollama-driven-tasks
Python scripts to run OCR and other tasks on local Ollama instance

The idea is to enable standardized LLM-driven tasks to run on a variety of hardware platforms without forcing users to configure them for different hardware, so that the same script can run on Macs with unified memory or Windows or Linux boxes with CUDA GPUs. This is achieved by letting Ollama handle the hardware, and writing scripts that use Ollama's API.

The expectation is that this will be useful for:

- experimenting with LLM configuration (choosing different LLMs, tweaking the prompts and other configuration options)
- automating tasks in a way that can be shared with users of different hardware platforms
- and of course: running everything locally, so that research data is never shared with online services

This approach introduces a significant limitation: you can only use models that can run on Ollama. TODO document process of adapting LLMs to run in Ollama, and limitations of this approach.

## Installation

- install [Ollama](https://ollama.com/)
- clone this repository into a convenient directory (the installation directory)
- set up a Python virtual environment with venv or conda in the installation directory
- activate the virtual environment
- install dependencies (currently just the Ollama API)

```
pip install -r requirements.txt
```

## Basic OCR Task

- create a project directory, which contains a directory called ```pngs```, which contains a set of page images in png format, whose filenames sort in the appropriate order
- open a terminal in the installation directory and activate the ollama-driven-tasks virtual environment
- run with ```python run-folder.py {directory}```, giving a full or relative path of the project directory
- the output will list the image files as they are processed, and will show a sample of the OCR text:

```
(venv) (base) Nekomata:nachbin pbinkley$ python run-folder.py demo
demo/pngs/Nachman-front-matter 0007.png
  # Como se fosse um Prefácio¹  O que é hi...
demo/pngs/Nachman-front-matter 0008.png
  12  JACOB NACHBIN: OS PRIMÓDIOS DA HISTO...
demo/pngs/Nachman-front-matter 0009.png
  Pois Nachbin, se teve falhas como histor...
```

- when it finishes, it will write the output to the terminal, and will also save it in the project directory. The file name includes a timestamp, so you can do multiple runs without overwriting any output:

```
$ ls -l ~/demo
total 56
drwxr-xr-x  13 pbinkley  staff    416 Apr 30 12:33 pngs
-rw-r--r--   1 pbinkley  staff  24630 Apr 30 14:13 transcription_20260430-141342.markdown
```

## Configuration

- currently hardcoded
- TODO replace prompts etc. with reasonable defaults; extract the current working configuration into a project-level configuration.json, which is automatically detected and which overrides the default

## TODO

- add the ability to OCR pdfs as well as image files
- add a translation task: break the OCR output into Markdown chunks, process and assemble them, without losing any structural details
- add easy control of the full range of LLM configuration options
- think about ChainForge-like evaluation pipelines, for automatically testing prompts and documenting the results by using ground truth files
