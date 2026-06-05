#!/usr/bin/env python

# set up configuration for a task

# sources:
# - command line value
# - config file specified on command line
# - default config file in directory
# - default value in this file

# configurable fields:
# - system prompt
# - user prompt
# - model
# - context window
# - temperature
# - output file
# - filename format (including timestamp, git commit id, etc.)

# use safe default values
# other parameters: https://docs.ollama.com/modelfile#parameter

# note: will have to handle multistep processes
# e.g. faster-whisper's use of an audio model for transcription and a text model for clean-up

import argparse
import json
import pdb

DEFAULT_MODEL = 'qwen3-vl:30b-a3b-instruct'

def build_config(parser, input_file):

    # 1. load specified values

    # parser = argparse.ArgumentParser(description="Ollama-driven task")

    # Use '?' for optional positional arguments, or start with '-' for flags
    parser.add_argument('-p', '--prompt', nargs='?', help='Prompt')
    parser.add_argument('-m', '--model', nargs='?', default=DEFAULT_MODEL, help='Model')
    parser.add_argument('-x', '--ctx', nargs='?', type=int, help='Context window (num_ctx)')
    parser.add_argument('-t', '--temp', nargs='?', type=float, help='Temperature')
    parser.add_argument('-c', '--conf', nargs='?', default='config.json', help='Configuration file')
    parser.add_argument('-i', '--ignore', nargs='?', const='', help='Ignore default config.json')
    parser.add_argument('-v', '--verbose', nargs='?', help='Verbose output')
    parser.add_argument('-g', '--groundtruth', nargs='?', help='Report error rates (ground truth)')
    parser.add_argument('input_file', nargs='?', default=input_file)

    parsed_args = parser.parse_args()
    args = vars(parsed_args) # convert to dict

    # 3. Separate parameters into different dictionaries
    script_dict = {}
    if 'groundtruth' in args: 
        script_dict['groundtruth'] = args['groundtruth']
    else:
        script_dict['groundtruth'] = False






    print(args)
    print(f"g: {script_dict['groundtruth']}")

    # pdb.set_trace()

    # select specified config file, if any
    if args['conf']:
        conf_file = args['conf']         
    # otherwise, select default config file if present and not ignored
    elif(not(args['ignore'])):           
        conf_file = './config.json'

    # load the selected config file into config
    if (conf_file):                   
        with open(conf_file, 'r') as file:
            config = json.load(file)
    else:
        config = {}                  # last resort: create empty config

    # add input_file (path to image) to config
    # pdb.set_trace()
    if args['prompt']:
        config['prompt'] = args['prompt']
    if args['model']:
        config['model'] = args['model']
    if args['ctx']:
        # TODO make sure config['options'] exists
        config['options']['num_ctx'] = args['ctx']
    if args['temp']:
        config['options']['temperature'] = args['temp']
    if args['input_file']:
        config['images'] = args['input_file']
    return config, script_dict

# config = build_config(parser, input_file)
# print(config)
