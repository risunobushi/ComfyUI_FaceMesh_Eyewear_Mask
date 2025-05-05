import torch
import numpy as np
from PIL import Image
import cv2 # OpenCV for drawing and dilation
import mediapipe as mp

# Initialize MediaPipe Face Mesh - it's better to initialize solutions outside the main function
# if the node might be called repeatedly, but initializing inside is simpler for standalone nodes.
# We will use a 'with' block inside the function for better resource management per execution.
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils # Not strictly needed for mask, but good to have access
mp_drawing_styles = mp.solutions.drawing_styles # Not strictly needed for mask

class FaceMeshEyewearMask:
    # --- MediaPipe Landmark Indices for Eyewear Region ---
    # These define the polygon boundary. Adjust if needed.
    # Refined indices based on the visual target (yellow area)
    # Tracing the perimeter clockwise:
    EYEWEAR_REGION_INDICES = [
        # == Top Edge (Brows and Upper Nose Bridge) ==
        # Start near Left Temple / Outer Brow
        103, 67, 109, 10,  # Outer-upper brow left -> moving inwards
        151, # Center top of nose bridge (slightly lower than absolute top brow)
        338, 297, 332, # Inner -> Outer-upper brow right
    
        # == Right Side Edge (Temple) ==
        286, # Outer corner area right
        348, # Mid-temple right
        411, # Lower temple / upper cheek right side
    
        # == Bottom Edge (Upper Cheeks / Lower Nose Bridge) ==
        374, # Right cheek below eye, outer
        381, # Right cheek below eye, inner-mid
        367, # Right cheek near nose wing
        # 364, # Alternate near nose wing right
        # Crossing nose bridge area below eyes
        397, # Center point below nose bridge, above tip
        # 172, # Alternate center point
        138, # Left cheek near nose wing
        # 135, # Alternate near nose wing left
        154, # Left cheek below eye, inner-mid
        145, # Left cheek below eye, outer
    
        # == Left Side Edge (Temple) ==
        187, # Lower temple / upper cheek left side
        119, # Mid-temple left
        56, # Outer corner area left
    
        # Connecting back towards the start (103)
        # (cv2.fillPoly usually closes it, but being explicit can help visualize)
        # 103 # Already listed as start
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), # Input image from ComfyUI
                "refine_landmarks": ("BOOLEAN", {"default": True}), # MediaPipe option
                "min_detection_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}), # Detection threshold
                "dilation_kernel_size": ("INT", {"default": 5, "min": 0, "max": 51, "step": 2}), # Kernel size for dilation (0 to disable)
                "dilation_iterations": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}), # Number of dilation iterations
            }
        }

    RETURN_TYPES = ("MASK",) # Output a single-channel mask
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask"
    CATEGORY = "image/masking" # Or choose a custom category like "landmarks"
    OUTPUT_NODE = False

    def create_mask(self, image, refine_landmarks, min_detection_confidence, dilation_kernel_size, dilation_iterations):
        batch_size, img_h, img_w, channels = image.shape
        output_masks = []

        # Process each image in the batch
        for i in range(batch_size):
            img_tensor = image[i] # Get single image tensor (H, W, C)

            # Convert ComfyUI image tensor (usually CHW or HWC, float 0-1) to NumPy array (HWC, uint8, 0-255) for OpenCV/MediaPipe
            img_np = img_tensor.cpu().numpy()
            if img_np.max() <= 1.0: # Check if image is normalized
                 img_np = (img_np * 255).astype(np.uint8)
            else:
                 img_np = img_np.astype(np.uint8) # Assume already 0-255 if max > 1

            # Ensure image is HWC (if it came in as CHW) - less common in ComfyUI output usually
            if img_np.shape[0] == channels and img_np.shape[2] != channels:
                 img_np = np.transpose(img_np, (1, 2, 0))

            # Ensure 3 channels (RGB) for MediaPipe
            if img_np.shape[-1] == 1: # Grayscale
                img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[-1] == 4: # RGBA
                img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            else: # Assume RGB
                img_np_rgb = img_np

            # Create a blank mask for this image
            current_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            # Initialize MediaPipe Face Mesh within a 'with' block for resource management
            try:
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True, # Process static images
                    max_num_faces=1,        # We only care about one face for this mask
                    refine_landmarks=refine_landmarks,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=0.5) as face_mesh: # tracking confidence less relevant for static images

                    # Process the image with MediaPipe
                    results = face_mesh.process(img_np_rgb) # MediaPipe expects RGB

                    # If landmarks are found, create the polygon mask
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark # Get landmarks for the first face

                        polygon_points = []
                        valid_landmarks = True
                        for idx in self.EYEWEAR_REGION_INDICES:
                            if idx < 0 or idx >= len(landmarks):
                                print(f"Warning: Landmark index {idx} out of bounds ({len(landmarks)} landmarks found). Skipping mask generation for this image.")
                                valid_landmarks = False
                                break
                            lm = landmarks[idx]
                            # Convert normalized coordinates to pixel coordinates
                            cx, cy = int(lm.x * img_w), int(lm.y * img_h)
                            polygon_points.append([cx, cy])

                        if valid_landmarks and polygon_points:
                            # Convert points to NumPy array of shape (N, 1, 2) for cv2.fillPoly
                            polygon_np = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))

                            # Draw the filled polygon on the mask
                            cv2.fillPoly(current_mask, [polygon_np], 255) # 255 for white color

                            # Optional: Dilate the mask slightly
                            if dilation_kernel_size > 0 and dilation_iterations > 0:
                                # Ensure kernel size is odd, otherwise OpenCV might error or behave unexpectedly
                                k_size = dilation_kernel_size if dilation_kernel_size % 2 != 0 else dilation_kernel_size + 1
                                kernel = np.ones((k_size, k_size), np.uint8)
                                current_mask = cv2.dilate(current_mask, kernel, iterations=dilation_iterations)

                    # If no landmarks detected, the mask remains black (zeros) - which is desired behavior
                    # else:
                    #     print(f"No face landmarks detected in image {i+1}/{batch_size}.") # Optional logging

            except Exception as e:
                 print(f"Error during MediaPipe processing or mask creation for image {i+1}/{batch_size}: {e}")
                 # Keep the mask black on error

            # Convert the NumPy mask (H, W, uint8 0-255) back to a Torch tensor (H, W, float32 0-1)
            # ComfyUI expects MASK type to be (Batch, H, W)
            mask_tensor = torch.from_numpy(current_mask.astype(np.float32) / 255.0)
            output_masks.append(mask_tensor) # Append (H, W) tensor

        # Stack all masks in the batch along the first dimension
        final_mask_batch = torch.stack(output_masks, dim=0) # Shape: (B, H, W)

        return (final_mask_batch,) # Return as a tuple

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FaceMeshEyewearMask": FaceMeshEyewearMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceMeshEyewearMask": "Face Mesh Eyewear Mask"
}
