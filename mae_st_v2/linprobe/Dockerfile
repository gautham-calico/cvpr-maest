FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install dependencies
RUN pip install --no-cache-dir nibabel h5py timm iopath simplejson

# Unbuffered output for logging
ENV PYTHONUNBUFFERED=1

# Set up working directory
WORKDIR /workspace/code

# Copy mae_st package
COPY mae_st/ /workspace/code/mae_st/

# Copy stripped encoder checkpoint
COPY encoder_checkpoint.pth /workspace/code/mae_st/linprobe/encoder_checkpoint.pth

# Place extract_feat_LP.sh in WORKDIR so "sh extract_feat_LP.sh" works
RUN cp /workspace/code/mae_st/linprobe/extract_feat_LP.sh /workspace/code/extract_feat_LP.sh && \
    chmod +x /workspace/code/extract_feat_LP.sh

# Create input/output directories
RUN mkdir -p /workspace/inputs /workspace/outputs

ENTRYPOINT ["/bin/bash", "-c", "sh extract_feat_LP.sh"]
