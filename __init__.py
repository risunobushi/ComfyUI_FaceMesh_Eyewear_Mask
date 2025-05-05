# __init__.py

# Import the module containing the node class AND the mappings
from . import facemesh_eyewear_mask

# Access the mappings defined at the module level in facemesh_eyewear_mask.py
NODE_CLASS_MAPPINGS = {
    **facemesh_eyewear_mask.NODE_CLASS_MAPPINGS,
    # Add other nodes here if you have more in this directory
}

NODE_DISPLAY_NAME_MAPPINGS = {
   **facemesh_eyewear_mask.NODE_DISPLAY_NAME_MAPPINGS,
   # Add other nodes here if you have more in this directory
}

# Export the combined mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("Loaded Face Mesh Eyewear Mask Custom Node") # Optional: print statement to confirm loading
