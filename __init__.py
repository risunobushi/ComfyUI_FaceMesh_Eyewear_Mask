# Import node modules
from . import facemesh_eyewear_mask
# from . import openpose_eyewear_mask # Comment out or remove if replacing
from . import pose_to_eyewear_mask # Import the new node
from . import mediapipe_face_to_mask # Import the MediaPipe face to mask node

# Combine mappings from all modules in this directory
NODE_CLASS_MAPPINGS = {
    **facemesh_eyewear_mask.NODE_CLASS_MAPPINGS,
    # **openpose_eyewear_mask.NODE_CLASS_MAPPINGS, # Comment out or remove if replacing
    **pose_to_eyewear_mask.NODE_CLASS_MAPPINGS, # Add the new node's mapping
    **mediapipe_face_to_mask.NODE_CLASS_MAPPINGS, # Add the MediaPipe face to mask node's mapping
}

NODE_DISPLAY_NAME_MAPPINGS = {
   **facemesh_eyewear_mask.NODE_DISPLAY_NAME_MAPPINGS,
   # **openpose_eyewear_mask.NODE_DISPLAY_NAME_MAPPINGS, # Comment out or remove if replacing
   **pose_to_eyewear_mask.NODE_DISPLAY_NAME_MAPPINGS, # Add the new node's display name
   **mediapipe_face_to_mask.NODE_DISPLAY_NAME_MAPPINGS, # Add the MediaPipe face to mask node's display name
}

# Export the combined mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Print confirmation (optional)
print("Loaded Face Mesh Eyewear Mask Custom Node")
# print("Loaded OpenPose Eyewear Mask Custom Node") # Comment out or remove if replacing
print("Loaded Mask From Facial Keypoints Custom Node")
print("Loaded MediaPipe Face to Mask Custom Node")
