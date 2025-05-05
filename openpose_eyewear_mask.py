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
    # And node can fail gracefully during execution instead of loading
    class DWPoseDetector:
        def __init__(self, *args, **kwargs):
            # This shouldn't be called if DWPOSE_INSTALLED is False and checked correctly
            print("Error: Dummy DWPoseDetector initialized - check install.")
        def __call__(self, *args, **kwargs):
             # This shouldn't be called
            print("Error: Dummy DWPoseDetector called - check install.")
            return None # Or raise error

class OpenPoseEyewearMask:
    # --- OpenPose Keypoint Indices (Body_25 convention used by DWPose) ---
    NOSE_IDX = 0
    LEYE_IDX = 15
    REYE_IDX = 16
    LEAR_IDX = 17
    REAR_IDX = 18
    POLYGON_ORDER_INDICES = [LEAR_IDX, LEYE_IDX, NOSE_IDX, REYE_IDX, REAR_IDX]
    REQUIRED_INDICES = [LEYE_IDX, NOSE_IDX, REYE_IDX]

    detector_instance = None
    current_dwpose_model = None # Track initialized models
    current_yolo_model = None

    @classmethod
    def INPUT_TYPES(cls):
        # Input types depend on whether the dependency is met
        if not DWPOSE_INSTALLED:
            # Provide minimal inputs and maybe an error message display if possible
            # Returning just image prevents crash but UI won't show full options
            return {
                "required": {
                     "image": ("IMAGE",),
                     # Maybe add a text widget indicating error if ComfyUI supports it easily?
                     # "error_message": ("STRING", {"default": "DWPose (comfyui_controlnet_aux) not installed!", "multiline": True}),
                }
            }

        # --- Define full inputs ONLY if DWPose is installed ---
        dwpose_options = ["NONE"]
        dwpose_yolo_options = ["NONE"]
        try:
            # Get available DWPose model paths
            model_dir_paths = folder_paths.get_folder_paths("dwpose") # From comfyui_controlnet_aux integration
            if model_dir_paths:
                # Use folder_paths.get_filename_list which is safer
                dwpose_options = folder_paths.get_filename_list("dwpose")
                dwpose_yolo_options = folder_paths.get_filename_list("dwpose") # Assuming YOLO models are in the same dir

                # Filter common names if needed (optional)
                dwpose_options = [f for f in dwpose_options if "dw-" in f and ".onnx" in f] or ["NONE"]
                dwpose_yolo_options = [f for f in dwpose_yolo_options if "yolox_" in f and ".onnx" in f] or ["NONE"]

            else:
                 print("Warning: DWPose model directory not found via folder_paths. Check comfyui_controlnet_aux setup.")

        except Exception as e:
            print(f"Warning: Could not list DWPose models - {e}")


        return {
            "required": {
                "image": ("IMAGE",),
                "detect_hand": ("BOOLEAN", {"default": False}),
                "detect_face": ("BOOLEAN", {"default": False}),
                "dwpose_model": (dwpose_options, ),
                "yolo_model": (dwpose_yolo_options, ),
                "confidence_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "dilation_kernel_size": ("INT", {"default": 5, "min": 0, "max": 51, "step": 2}),
                "dilation_iterations": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask"
    CATEGORY = "image/masking"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        # Using image tensor hash is generally okay for change detection
        return folder_paths.get_hash(image.cpu().numpy()) if isinstance(image, torch.Tensor) else hash(str(image))

    def initialize_detector(self, dwpose_model, yolo_model):
        """Initializes/Reinitializes the DWPoseDetector if needed."""
        # Check if models changed since last init
        if (OpenPoseEyewearMask.detector_instance is None or
            OpenPoseEyewearMask.current_dwpose_model != dwpose_model or
            OpenPoseEyewearMask.current_yolo_model != yolo_model):

            print(f"Initializing DWPoseDetector. DWPose: {dwpose_model}, YOLO: {yolo_model}")
            try:
                # Construct full paths using folder_paths
                dwpose_path = folder_paths.get_full_path("dwpose", dwpose_model)
                yolo_path = folder_paths.get_full_path("dwpose", yolo_model)

                if not dwpose_path:
                    raise FileNotFoundError(f"Could not find DWPose model: {dwpose_model}")
                if not yolo_path:
                     raise FileNotFoundError(f"Could not find YOLO model: {yolo_model}")

                # Instantiate detector (requires comfyui_controlnet_aux to be installed)
                OpenPoseEyewearMask.detector_instance = DWPoseDetector(dwpose_path, yolo_path)
                OpenPoseEyewearMask.current_dwpose_model = dwpose_model
                OpenPoseEyewearMask.current_yolo_model = yolo_model
                print("DWPoseDetector initialized successfully.")

            except FileNotFoundError as e:
                 print(f"Error: {e}. Check comfyui_controlnet_aux model paths.")
                 # Set instance to None so it doesn't try to use a bad/old one
                 OpenPoseEyewearMask.detector_instance = None
                 raise # Re-raise the error to stop execution cleanly
            except Exception as e:
                 print(f"Error initializing DWPose detector: {e}")
                 OpenPoseEyewearMask.detector_instance = None
                 raise # Re-raise

        return OpenPoseEyewearMask.detector_instance

    # Corrected function signature - REMOVE the arguments if DWPOSE_INSTALLED is False
    # But the calling mechanism means we MUST define them all if DWPOSE_INSTALLED is True
    # So the check needs to be INSIDE the function
    def create_mask(self, image, detect_hand=False, detect_face=False, dwpose_model=None, yolo_model=None, confidence_threshold=0.1, dilation_kernel_size=5, dilation_iterations=2):

        # --- !! CRITICAL CHECK !! ---
        if not DWPOSE_INSTALLED:
            print("Error: OpenPoseEyewearMask cannot run because DWPose (comfyui_controlnet_aux) is not installed or failed to import.")
            # Return a black mask matching input dimensions
            batch_size, img_h, img_w, _ = image.shape
            black_mask = torch.zeros((batch_size, img_h, img_w), dtype=torch.float32, device=image.device)
            # You might want to also signal an error in the UI if possible, but returning black mask is a safe fallback.
            return (black_mask,)
        # --- End Critical Check ---

        # Proceed only if DWPose is installed
        try:
            # Ensure models are selected if DWPose is installed
            if dwpose_model is None or dwpose_model == "NONE" or yolo_model is None or yolo_model == "NONE":
                 print("Error: DWPose and YOLO models must be selected in the node properties.")
                 raise ValueError("DWPose/YOLO model not selected.")

            detector = self.initialize_detector(dwpose_model, yolo_model)
            if detector is None: # Initialization failed
                 raise RuntimeError("DWPose detector initialization failed.")

        except Exception as e:
            print(f"Error during detector setup: {e}")
            # Return black mask on setup error
            batch_size, img_h, img_w, _ = image.shape
            black_mask = torch.zeros((batch_size, img_h, img_w), dtype=torch.float32, device=image.device)
            return (black_mask,)

        # --- Rest of the processing logic (remains largely the same) ---
        batch_size, img_h, img_w, channels = image.shape
        output_masks = []

        for i in range(batch_size):
            # ... (image conversion to numpy BGR - same as before) ...
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            if img_np.shape[-1] == 3: img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif img_np.shape[-1] == 4: img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            elif img_np.shape[-1] == 1: img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            else: img_np_bgr = img_np # Fallback

            current_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            try:
                pose_results = detector(img_np_bgr, detect_hand=detect_hand, detect_face=detect_face)
                # ... (rest of the keypoint extraction, polygon creation, dilation logic - same as before) ...
                poses = []
                if isinstance(pose_results, dict) and 'bodies' in pose_results:
                    poses = pose_results['bodies']
                elif isinstance(pose_results, list) and len(pose_results) > 0 and isinstance(pose_results[0], dict) and 'keypoints' in pose_results[0]:
                     poses = pose_results

                if poses:
                    keypoints = poses[0]['keypoints']
                    polygon_points_indices = []
                    valid_keypoints = {}
                    for idx in self.POLYGON_ORDER_INDICES:
                        if idx < len(keypoints):
                            kpt = keypoints[idx]
                            x, y, conf = kpt[0], kpt[1], kpt[2]
                            if conf >= confidence_threshold and x > 0 and y > 0:
                                valid_keypoints[idx] = (int(x), int(y))
                                polygon_points_indices.append(idx)

                    has_required = all(idx in valid_keypoints for idx in self.REQUIRED_INDICES)
                    if has_required and len(polygon_points_indices) >= 3:
                        polygon_coords = [valid_keypoints[idx] for idx in polygon_points_indices]
                        polygon_np = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(current_mask, [polygon_np], 255)

                        if dilation_kernel_size > 0 and dilation_iterations > 0:
                            k_size = dilation_kernel_size if dilation_kernel_size % 2 != 0 else dilation_kernel_size + 1
                            kernel = np.ones((k_size, k_size), np.uint8)
                            current_mask = cv2.dilate(current_mask, kernel, iterations=dilation_iterations)

            except Exception as e:
                 print(f"Error during OpenPose processing or mask creation for image {i+1}: {e}")

            # ... (conversion back to tensor - same as before) ...
            mask_tensor = torch.from_numpy(current_mask.astype(np.float32) / 255.0)
            output_masks.append(mask_tensor)

        final_mask_batch = torch.stack(output_masks, dim=0)
        return (final_mask_batch,)

# --- Node Mappings (remain the same) ---
NODE_CLASS_MAPPINGS = {
    "OpenPoseEyewearMask": OpenPoseEyewearMask
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPoseEyewearMask": "OpenPose Eyewear Mask"
}
