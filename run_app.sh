#!/bin/bash
# Script to run the LLMmemory Streamlit app

export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export MODEL_ID="/Users/jaskiratsingh/Desktop/humo-app/Models/Qwen3-VL_8B_(4bit)"
export MODEL_KIND="vlm"
streamlit run src/llmmemory/apps/streamlit_app.py
