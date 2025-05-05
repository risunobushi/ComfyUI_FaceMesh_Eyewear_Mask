# Import node modules
from . import facemesh_eyewear_mask # Assuming you keep the previous node
from . import openpose_eyewear_mask

# Combine mappings from all modules in this directory
NODE_CLASS_MAPPINGS = {
    **facemesh_eyewear_mask.NODE_CLASS_MAPPINGS,
    **openpose_eyewear_mask.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
   **facemesh_eyewear_mask.NODE_DISPLAY_NAME_MAPPINGS,
   **openpose_eyewear_mask.NODE_DISPLAY_NAME_MAPPINGS,
}

# Export the combined mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Print confirmation (optional)
print("Loaded Face Mesh Eyewear Mask Custom Node")
if openpose_eyewear_mask.DWPOSE_INSTALLED:
    print("Loaded OpenPose Eyewear Mask Custom Node")
else:
    print("Warning: OpenPose Eyewear Mask node loaded BUT requires 'comfyui_controlnet_aux' to be installed.")
