#!/bin/bash
# Docker entrypoint for Task 1: Linear Probing
# Calls predict.py which handles both ROI and non-ROI modes
set -e
cd /workspace/code
python -m mae_st.linprobe.predict
