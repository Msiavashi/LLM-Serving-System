#!/usr/bin/env bash
# Simple setup script for LLM Serving System
set -e
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

