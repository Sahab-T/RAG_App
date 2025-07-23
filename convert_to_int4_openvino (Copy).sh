#!/bin/bash
MODEL_DIR="llama3-8b-instruct"
OUTPUT_DIR="llama3-8b-instruct-int4"
python3 -m openvino.tools.mo \
  --input_model $MODEL_DIR \
  --output_dir $OUTPUT_DIR \
  --data_type INT4