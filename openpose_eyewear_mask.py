import torch
import numpy as np
from PIL import Image
import cv2 # OpenCV for drawing and dilation
import folder_paths # ComfyUI utility

# --- Attempt to import DWPoseDetector ---
# This node REQUIRES the user to have comfyui_controlnet_aux installed
try:
    from comfyui_controlnet_aux.dwpose import DWPoseDetector
    print("DWPoseDetector imported successfully.")
    DWPOSE_INSTALLED = True
except ImportError:
    print("Warning: DWPoseDetector could not be imported. comfyui_controlnet_aux is required for the OpenPose Eyewear Mask node.")
    DWPOSE_INSTALLED = False
    # Define a dummy class if import fails, so ComfyUI can at least load the structure
    class DWPoseDetector:
        def __init__(self, *args, **kwargs):
            raise ImportError("DWPoseDetector failed to import. Please install comfyui_controlnet_aux.")
        def __call__(self, *args, **kwargs):
            raise ImportError("DWPoseDetector failed to import. Please install comfyui_controlnet_aux.")


class OpenPoseEyewearMask:
    # --- OpenPose Keypoint Indices (Body_25 convention used by DWPose) ---
    # We need Nose, Eyes, and Ears
    NOSE_IDX = 0
    LEYE_IDX = 15
    REYE_IDX = 16
    LEAR_IDX = 17
    REAR_IDX = 18

    # Define the order to create the polygon mask
    # We try to go: Left Ear -> Left Eye -> Nose -> Right Eye -> Right Ear
    # If ears are missing, we'll just use Eye -> Nose -> Eye
    POLYGON_ORDER_INDICES = [LEAR_IDX, LEYE_IDX, NOSE_IDX, REYE_IDX, REAR_IDX]

    # Minimum required keypoints for a basic mask (Eyes and Nose)
    REQUIRED_INDICES = [LEYE_IDX, NOSE_IDX, REYE_IDX]

    # Class attribute to hold the detector instance
    detector_instance = None

    @classmethod
    def INPUT_TYPES(cls):
        # Check if the dependency is met before defining inputs
        if not DWPOSE_INSTALLED:
             # If DWPose isn't installed, provide minimal inputs to avoid crashing ComfyUI load
            return {"required": {"image": ("IMAGE",)}}

        # Define inputs if DWPose is installed
        # Get available DWPose model paths
        model_dir = folder_paths.get_folder_paths("dwpose") # From comfyui_controlnet_aux integration
        if not model_dir:
            print("Warning: DWPose model directory not found. Check comfyui_controlnet_aux setup.")
            dwpose_yolo_options = ["NONE"]
            dwpose_options = ["NONE"]
        else:
            model_dir = model_dir[0] # Get the first path if multiple exist
            dwpose_yolo_options = ["yolox_l.onnx"] # Default or add more if needed/detected
            dwpose_options = ["dw-ll_ucoco_384.onnx", "dw-ll_ucoco.onnx", "dw-l_ucoco.onnx", "dw-m_ucoco.onnx", "dw-s_ucoco.onnx"] # Common options

        return {
            "required": {
                "image": ("IMAGE",), # Input image from ComfyUI
                "detect_hand": ("BOOLEAN", {"default": False}), # DWPose option
                "detect_face": ("BOOLEAN", {"default": False}), # DWPose option - Not face mesh, just basic detection
                "dwpose_model": (dwpose_options, ),
                "yolo_model": (dwpose_yolo_options, ),
                "confidence_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}), # Min confidence for keypoints
                "dilation_kernel_size": ("INT", {"default": 5, "min": 0, "max": 51, "step": 2}), # Kernel size for dilation (0 to disable)
                "dilation_iterations": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}), # Number of dilation iterations
            }
        }

    RETURN_TYPES = ("MASK",) # Output a single-channel mask
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask"
    CATEGORY = "image/masking" # Or choose a custom category like "landmarks"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        # Basic check: If image hash changes, need to re-run
        return image.shape[0] if isinstance(image, torch.Tensor) else hash(image.tobytes())

    def initialize_detector(self, dwpose_model, yolo_model):
        """Initializes the DWPoseDetector if not already done or if models changed."""
        # Construct full paths using folder_paths
        model_dir = folder_paths.get_folder_paths("dwpose")[0]
        dwpose_path = folder_paths.get_full_path("dwpose", dwpose_model)
        yolo_path = folder_paths.get_full_path("dwpose", yolo_model)

        if not dwpose_path or not yolo_path:
             raise FileNotFoundError("Could not find DWPose or YOLO models. Check comfyui_controlnet_aux setup and model names.")

        # Check if instance exists and uses the same models
        if OpenPoseEyewearMask.detector_instance:
            # Crude check: Compare model names. A more robust check might involve hashes or full paths.
            current_dw_model = getattr(OpenPoseEyewearMask.detector_instance, 'model_name', None) # Assuming detector stores its model name
            current_yolo_model = getattr(OpenPoseEyewearMask.detector_instance, 'yolo_model_name', None)
            # Need to see how DWPoseDetector stores its model names or implement a way to track them.
            # For now, let's re-initialize if models might have changed, or just keep one instance.
            # A simpler approach for now: always keep the first initialized detector.
            # If model switching is needed frequently, this needs more complex state management.
            pass # Keep existing instance for now
        else:
            print(f"Initializing DWPoseDetector with DWPose: {dwpose_model}, YOLO: {yolo_model}")
            OpenPoseEyewearMask.detector_instance = DWPoseDetector(dwpose_path, yolo_path)
            # Store model names if possible for future checks (depends on DWPoseDetector implementation)
            # setattr(OpenPoseEyewearMask.detector_instance, 'model_name', dwpose_model)
            # setattr(OpenPoseEyewearMask.detector_instance, 'yolo_model_name', yolo_model)

        return OpenPoseEyewearMask.detector_instance


    def create_mask(self, image, detect_hand, detect_face, dwpose_model, yolo_model, confidence_threshold, dilation_kernel_size, dilation_iterations):

        if not DWPOSE_INSTALLED:
             print("Error: DWPose is not installed (comfyui_controlnet_aux). Cannot create OpenPose mask.")
             # Return a black mask matching input dimensions
             batch_size, img_h, img_w, _ = image.shape
             black_mask = torch.zeros((batch_size, img_h, img_w), dtype=torch.float32, device=image.device)
             return (black_mask,)

        try:
            detector = self.initialize_detector(dwpose_model, yolo_model)
        except Exception as e:
            print(f"Error initializing DWPose detector: {e}")
            # Return black mask on initialization error
            batch_size, img_h, img_w, _ = image.shape
            black_mask = torch.zeros((batch_size, img_h, img_w), dtype=torch.float32, device=image.device)
            return (black_mask,)


        batch_size, img_h, img_w, channels = image.shape
        output_masks = []

        # Process each image in the batch
        for i in range(batch_size):
            img_tensor = image[i] # Get single image tensor (H, W, C)

            # Convert ComfyUI image tensor (0-1 float) to NumPy array (HWC, uint8, 0-255) for DWPose
            img_np = img_tensor.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # DWPose detector expects BGR format
            if img_np.shape[-1] == 3: # Assume RGB if 3 channels
                img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif img_np.shape[-1] == 4: # RGBA
                img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            elif img_np.shape[-1] == 1: # Grayscale
                 img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            else:
                 print(f"Warning: Unexpected number of channels ({img_np.shape[-1]}) in input image {i+1}. Attempting to proceed.")
                 img_np_bgr = img_np # Hope for the best


            # Create a blank mask for this image
            current_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            try:
                # Run DWPose detection
                # Note: The call signature might vary slightly depending on the comfyui_controlnet_aux version
                pose_results = detector(img_np_bgr, detect_hand=detect_hand, detect_face=detect_face)
                # Expected result structure (may vary): often a dict or list containing poses,
                # each pose having 'keypoints' which is a list of [x, y, conf]

                # Check if any poses were detected (result structure might differ)
                # Assuming pose_results is a list of detected bodies/poses
                # Or if it's directly the keypoints list/array for the primary person
                poses = []
                if isinstance(pose_results, dict) and 'bodies' in pose_results:
                    poses = pose_results['bodies'] # Common structure
                elif isinstance(pose_results, list) and len(pose_results) > 0 and isinstance(pose_results[0], dict) and 'keypoints' in pose_results[0]:
                     poses = pose_results # List of pose dicts
                # Add more checks if the structure is different

                if poses:
                    # Get keypoints for the first detected person
                    keypoints = poses[0]['keypoints'] # Assuming list of [x, y, conf]

                    polygon_points_indices = [] # Store indices of valid points in order
                    valid_keypoints = {} # Store coordinates of valid points by index

                    # Extract valid keypoints based on confidence
                    for idx in self.POLYGON_ORDER_INDICES:
                        if idx < len(keypoints):
                            kpt = keypoints[idx]
                            x, y, conf = kpt[0], kpt[1], kpt[2]
                            if conf >= confidence_threshold and x > 0 and y > 0: # OpenPose uses (0,0) for undetected points
                                valid_keypoints[idx] = (int(x), int(y))
                                polygon_points_indices.append(idx)

                    # Check if minimum required points are present
                    has_required = all(idx in valid_keypoints for idx in self.REQUIRED_INDICES)

                    if has_required and len(polygon_points_indices) >= 3: # Need at least 3 points for a polygon
                        # Build the polygon point list in the defined order, using only valid points
                        polygon_coords = [valid_keypoints[idx] for idx in polygon_points_indices]

                        # Convert points to NumPy array for cv2.fillPoly
                        polygon_np = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))

                        # Draw the filled polygon on the mask
                        cv2.fillPoly(current_mask, [polygon_np], 255) # White

                        # Optional: Dilate the mask
                        if dilation_kernel_size > 0 and dilation_iterations > 0:
                            k_size = dilation_kernel_size if dilation_kernel_size % 2 != 0 else dilation_kernel_size + 1
                            kernel = np.ones((k_size, k_size), np.uint8)
                            current_mask = cv2.dilate(current_mask, kernel, iterations=dilation_iterations)
                    # else:
                    #     print(f"Not enough valid keypoints found for image {i+1} (Required: {self.REQUIRED_INDICES}, Found Valid: {list(valid_keypoints.keys())})")

                # else:
                #     print(f"No poses detected in image {i+1}.")


            except Exception as e:
                 print(f"Error during OpenPose processing or mask creation for image {i+1}: {e}")
                 # Keep the mask black on error

            # Convert the NumPy mask back to a Torch tensor (H, W, float32 0-1)
            mask_tensor = torch.from_numpy(current_mask.astype(np.float32) / 255.0)
            output_masks.append(mask_tensor)

        # Stack all masks in the batch
        final_mask_batch = torch.stack(output_masks, dim=0) # Shape: (B, H, W)

        return (final_mask_batch,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "OpenPoseEyewearMask": OpenPoseEyewearMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPoseEyewearMask": "OpenPose Eyewear Mask"
}
