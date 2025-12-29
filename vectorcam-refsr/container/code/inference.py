import sys
import os
import json
import logging
import torch
import io
import numpy as np
import boto3
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

sys.path.append('/opt/ml/code/TTSR')

try: 
    from model.TTSR import TTSR
except ImportError as e:
    logger.error("Could not import TTSR. Check PYTHONPATH.")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
s3 = boto3.client('s3')

# Transform pipeline for input images
trans = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Args class required by TTSR model
class Args:
    def __init__(self):
        self.num_res_blocks = '16+16+8+4+4'
        self.n_feats = 64
        self.res_scale = 1.0

def model_fn(model_dir):
    logger.info("Loading model...")
    model = TTSR(Args())
    
    # In SageMaker, model artifacts are unzipped to 'model_dir'
    # We expect 'TTSR-rec.pt' to be there.
    path = os.path.join(model_dir, 'TTSR-rec.pt')

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    checkpoint = torch.load(path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully!")
    return model


def input_fn(request_body, request_content_type):
    """
    Expected Input JSON:
    {
        "bucket": "my-bucket",
        "input_key": "incoming/mosquito_blur.jpg",
        "ref_key": "refs/mosquito_good.jpg"
    }
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        bucket = data['bucket']

        logger.info(f"Downloading images from bucket: {bucket}")

        f_input  = io.BytesIO()
        s3.download_fileobj(bucket, data['input_key'], f_input)
        f_input.seek(0)  # Reset to beginning before reading
        input_img = Image.open(f_input).convert('RGB')


        f_ref = io.BytesIO()
        s3.download_fileobj(bucket, data['ref_key'], f_ref)
        f_ref.seek(0)  # Reset to beginning before reading
        ref_img = Image.open(f_ref).convert('RGB')

        input_tensor = trans(input_img).unsqueeze(0)
        ref_tensor = trans(ref_img).unsqueeze(0)

        return input_tensor, ref_tensor
    else:
        raise ValueError(f"Content type must be application/json")
    
def predict_fn(input_data, model):
    lr_tensor, ref_tensor = input_data
    lr_tensor = lr_tensor.to(device)
    ref_tensor = ref_tensor.to(device)

    with torch.no_grad():
        sr_tensor, _, _, _ = model(lr_tensor, ref_tensor)

    return sr_tensor

def output_fn(prediction, response_content_type):
    res_img = transforms.ToPILImage()(prediction.squeeze(0).cpu())
    
    buf = io.BytesIO()
    res_img.save(buf, format="PNG")
    return buf.getvalue()
