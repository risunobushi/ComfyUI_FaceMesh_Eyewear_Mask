# Import the node class
from .facemesh_eyewear_mask import FaceMeshEyewearMask

# A dictionary that routes node lists and mappings to the appropriate resource lists
NODE_CLASS_MAPPINGS = {
    **FaceMeshEyewearMask.NODE_CLASS_MAPPINGS,
    # Add other nodes here if you have more in this directory
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
   **FaceMeshEyewearMask.NODE_DISPLAY_NAME_MAPPINGS,
   # Add other nodes here if you have more in this directory
}

# Export the mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("Loaded Face Mesh Eyewear Mask Custom Node") # Optional: print statement to confirm loading