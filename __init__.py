# Import node modules
from . import facemesh_eyewear_mask
# from . import openpose_eyewear_mask # Comment out or remove if replacing
from . import pose_to_eyewear_mask # Import the new node

# Import MediaPipe face to mask node with error handling
try:
    from . import mediapipe_face_to_mask
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe Face to Mask module imported successfully")
except Exception as e:
    print(f"Failed to import MediaPipe Face to Mask module: {e}")
    MEDIAPIPE_AVAILABLE = False
    mediapipe_face_to_mask = None

# Combine mappings from all modules in this directory
NODE_CLASS_MAPPINGS = {
    **facemesh_eyewear_mask.NODE_CLASS_MAPPINGS,
    # **openpose_eyewear_mask.NODE_CLASS_MAPPINGS, # Comment out or remove if replacing
    **pose_to_eyewear_mask.NODE_CLASS_MAPPINGS, # Add the new node's mapping
}

# Add MediaPipe mappings if available
if MEDIAPIPE_AVAILABLE and mediapipe_face_to_mask:
    NODE_CLASS_MAPPINGS.update(mediapipe_face_to_mask.NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
   **facemesh_eyewear_mask.NODE_DISPLAY_NAME_MAPPINGS,
   # **openpose_eyewear_mask.NODE_DISPLAY_NAME_MAPPINGS, # Comment out or remove if replacing
   **pose_to_eyewear_mask.NODE_DISPLAY_NAME_MAPPINGS, # Add the new node's display name
}

# Add MediaPipe display names if available
if MEDIAPIPE_AVAILABLE and mediapipe_face_to_mask:
    NODE_DISPLAY_NAME_MAPPINGS.update(mediapipe_face_to_mask.NODE_DISPLAY_NAME_MAPPINGS)

# Export the combined mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Print confirmation (optional)
print("Loaded Face Mesh Eyewear Mask Custom Node")
# print("Loaded OpenPose Eyewear Mask Custom Node") # Comment out or remove if replacing
print("Loaded Mask From Facial Keypoints Custom Node")

if MEDIAPIPE_AVAILABLE:
    print("Loaded MediaPipe Face to Mask Custom Node")
else:
    print("MediaPipe Face to Mask Custom Node NOT loaded due to import error")
