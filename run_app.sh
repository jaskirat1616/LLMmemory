#!/bin/bash
# Script to run the LLMmemory Streamlit app

export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
streamlit run src/llmmemory/apps/streamlit_app.py
