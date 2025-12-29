import torch
from PIL import Image
from torchvision import transforms
import sys
import os

# --- PATH HACK ---
# The TTSR code was written to be run from inside its own folder.
# We need to tell Python where to look for it.
sys.path.append(os.path.join(os.getcwd(), 'model/TTSR'))

try:
    # Import the TTSR class from the model subfolder
    from model.TTSR import TTSR
except ImportError as e:
    print("❌ Error: Could not find the TTSR code.")
    print("Did you run 'git clone' inside the model folder?")
    print(f"Details: {e}")
    sys.exit(1)

def run_test():
    print("1. Loading Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")

    # Create a simple args object with required parameters
    class Args:
        def __init__(self):
            self.num_res_blocks = '16+16+8+4+4'  # Default architecture
            self.n_feats = 64
            self.res_scale = 1.0
    
    # Initialize the architecture
    model = TTSR(Args())
    
    # Load the weights you verified earlier
    checkpoint = torch.load('model/checkpoints/TTSR-rec.pt', map_location=device)
    
    # Handle the 'state_dict' structure if present
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print("   Model loaded!")

    print("2. Preparing Images...")
    # Resize images to be small (160px) so your laptop doesn't freeze
    # (TTSR is heavy!)
    t_list = [transforms.Resize((160,160)), transforms.ToTensor()] 
    trans = transforms.Compose(t_list)

    try:
        lr_img = Image.open('mosquito_blur.jpg').convert('RGB')
        ref_img = Image.open('mosquito_ref.jpg').convert('RGB')
    except FileNotFoundError:
        print("❌ Error: Missing images.")
        print("Please ensure 'mosquito_blur.jpg' and 'mosquito_ref.jpg' are in this folder.")
        return

    # Turn images into math (Tensors)
    lr_tensor = trans(lr_img).unsqueeze(0).to(device)
    ref_tensor = trans(ref_img).unsqueeze(0).to(device)

    print("3. Running AI Inference (This might take 10-20 seconds)...")
    with torch.no_grad():
        # The Magic Step: Input + Reference = Super Resolution
        sr_tensor, _, _, _ = model(lr_tensor, ref_tensor)

    print("4. Saving Result...")
    # Convert math back to image
    output = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())
    output.save("output_enhanced.png")
    print("✅ Success! Check 'output_enhanced.png' to see the result.")

if __name__ == "__main__":
    run_test()