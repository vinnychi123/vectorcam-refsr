import torch

# Point this to your 26MB file
model_path = "model/checkpoints/TTSR-rec (1).pt"

try:
    # Load the dictionary
    state_dict = torch.load(model_path, map_location='cpu')
    
    print(f"✅ Success! File loaded. Size in memory matches file.")
    print(f"Root keys: {state_dict.keys()}")
    
    # Check if it is a 'state_dict' (just weights) or a full checkpoint (with epoch, optimizer)
    if 'state_dict' in state_dict:
        # It's a full checkpoint, the weights are inside this key
        print("Structure: Full Checkpoint (contains 'state_dict')")
        weights = state_dict['state_dict']
    else:
        # It's likely just the weights directly
        print("Structure: Weights Only (Direct Key-Value pairs)")
        weights = state_dict
        
    # Print the first layer to verify it's not empty
    first_layer_name = list(weights.keys())[0]
    print(f"First Layer: {first_layer_name}")
    print(f"Shape: {weights[first_layer_name].shape}")

except Exception as e:
    print(f"❌ Error: The file is likely corrupted or not a PyTorch model.\nDetails: {e}")