# Start with AWS PyTorch Base Image (Pre-installed with drivers)
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.8.1-gpu-py36-cu111-ubuntu18.04

# Install python libraries
COPY code/requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

# Copy the TTSR Source Code (The "Architecture")
# We copy it to /opt/ml/code/TTSR so imports work
COPY ../model/TTSR /opt/ml/code/TTSR

# Copy the Inference Script
COPY code/inference.py /opt/ml/code/inference.py

# Define entrypoint
ENV SAGEMAKER_PROGRAM inference.py