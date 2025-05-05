import torch
import numpy as np
from PIL import Image
import cv2 # OpenCV for drawing and dilation
import folder_paths # ComfyUI utility
import comfy.model_management as model_management # For device selection
import json # For handling potential output if needed

# --- Attempt to import DWPoseDetector ---
# Use the standard external import path
DWPOSE_INSTALLED = False
DWPOSE_IMPORT_ERROR = ""
try:
    from comfyui_controlnet_aux.dwpose import DwposeDetector # Standard external import
    print("DWPoseDetector class imported successfully from comfyui_controlnet_aux.dwpose")
    DWPOSE_INSTALLED = True
except ImportError as e:
    DWPOSE_IMPORT_ERROR = f"Failed to import DWPoseDetector: {e}. Check comfyui_controlnet_aux install and its dependencies (like onnxruntime)."
    print(f"Warning: {DWPOSE_IMPORT_ERROR}")
    # Define dummy class
    class DwposeDetector: pass # Minimal dummy
except Exception as e:
    DWPOSE_IMPORT_ERROR = f"An unexpected error occurred during DWPoseDetector import: {e}"
    print(f"Warning: {DWPOSE_IMPORT_ERROR}")
    class DwposeDetector: pass # Minimal dummy


# --- Define constants based on DWPose_Preprocessor ---
# These repo names are used by from_pretrained internally
DWPOSE_MODEL_REPO = "yzd-v/DWPose"
HF_YOLOX_REPO = "hr16/yolox-onnx"
HF_YOLONAS_REPO = "hr16/yolo-nas-fp16"
HF_DWPOSE_REPO = "hr16/UnJIT-DWPose"
HF_DWPOSE_TORCHSCRIPT_REPO = "hr16/DWPose-TorchScript-BatchSize5"
HF_RTMP_REPO = "hr16/RTMPose" # Assuming, adjust if AnimalPose used different ones

class OpenPoseEyewearMask:
    # OpenPose Indices (same as before)
    NOSE_IDX, LEYE_IDX, REYE_IDX, LEAR_IDX, REAR_IDX = 0, 15, 16, 17, 18
    POLYGON_ORDER_INDICES = [LEAR_IDX, LEYE_IDX, NOSE_IDX, REYE_IDX, REAR_IDX]
    REQUIRED_INDICES = [LEYE_IDX, NOSE_IDX, REYE_IDX]

    detector_instance = None
    current_dwpose_filename = None # Track initialized models by filename
    current_yolo_filename = None

    @classmethod
    def INPUT_TYPES(cls):
        if not DWPOSE_INSTALLED:
            # Minimal inputs + error display
            return {
                "required": {"image": ("IMAGE",)},
                "hidden": {"error_msg": ("STRING", {"default": f"NODE ERROR: {DWPOSE_IMPORT_ERROR}"})} # Hidden input to show error
            }

        # --- Define full inputs ONLY if DWPose is installed ---
        # Use filenames consistent with DWPose_Preprocessor defaults/options
        # Note: TorchScript models might require specific device handling not fully implemented here yet
        yolo_options = [
            "yolox_l.onnx", "yolox_l.torchscript.pt", # From DWPOSE_MODEL_REPO or HF_YOLOX_REPO
            "yolo_nas_l_fp16.onnx", "yolo_nas_m_fp16.onnx", "yolo_nas_s_fp16.onnx" # From HF_YOLONAS_REPO
        ]
        dwpose_options = [
             "dw-ll_ucoco_384.onnx", # From DWPOSE_MODEL_REPO
             "dw-ll_ucoco.onnx", # From HF_DWPOSE_REPO
             "dw-ll_ucoco_384_bs5.torchscript.pt" # From HF_DWPOSE_TORCHSCRIPT_REPO
        ]

        # Attempt to list available models using folder_paths as fallback/augmentation
        try:
            # List ONNX models from the primary DWPose location first
            primary_dwpose_path = folder_paths.get_folder_paths("dwpose")
            if primary_dwpose_path:
                 onnx_files = folder_paths.get_filename_list("dwpose")
                 # Add discovered models if they match expected patterns and aren't duplicates
                 for f in onnx_files:
                     if f.endswith(".onnx"):
                         if ("yolox" in f or "yolo_nas" in f) and f not in yolo_options: yolo_options.append(f)
                         if "dw-" in f and f not in dwpose_options: dwpose_options.append(f)
            else:
                 print("OpenPoseEyewearMask: DWPose model directory not found via folder_paths. Using hardcoded list.")
        except Exception as e:
            print(f"OpenPoseEyewearMask: Warning - Could not dynamically list DWPose models: {e}")


        return {
            "required": {
                "image": ("IMAGE",),
                "detect_hand": ("BOOLEAN", {"default": False}),
                "detect_face": ("BOOLEAN", {"default": False}),
                # Use FILENAMES for models now
                "dwpose_filename": (dwpose_options, {"default": "dw-ll_ucoco_384.onnx"}),
                "yolo_filename": (yolo_options, {"default": "yolox_l.onnx"}),
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
        # Simple hash check is usually sufficient
        return folder_paths.get_hash(image.cpu().numpy()) if isinstance(image, torch.Tensor) else hash(str(image))

    def initialize_detector(self, dwpose_filename, yolo_filename):
        """Initializes/Reinitializes the DWPoseDetector using from_pretrained."""

        if (OpenPoseEyewearMask.detector_instance is None or
            OpenPoseEyewearMask.current_dwpose_filename != dwpose_filename or
            OpenPoseEyewearMask.current_yolo_filename != yolo_filename):

            print(f"OpenPoseEyewearMask: Initializing DWPoseDetector. DWPose file: {dwpose_filename}, YOLO file: {yolo_filename}")
            try:
                # Determine repo based on filename (mirroring DWPose_Preprocessor logic)
                if yolo_filename == "yolox_l.onnx": yolo_repo = DWPOSE_MODEL_REPO
                elif "yolox" in yolo_filename: yolo_repo = HF_YOLOX_REPO
                elif "yolo_nas" in yolo_filename: yolo_repo = HF_YOLONAS_REPO
                else: raise ValueError(f"Cannot determine repository for YOLO model: {yolo_filename}")

                if dwpose_filename == "dw-ll_ucoco_384.onnx": pose_repo = DWPOSE_MODEL_REPO
                elif dwpose_filename.endswith(".onnx"): pose_repo = HF_DWPOSE_REPO
                elif dwpose_filename.endswith(".torchscript.pt"): pose_repo = HF_DWPOSE_TORCHSCRIPT_REPO
                else: raise ValueError(f"Cannot determine repository for DWPose model: {dwpose_filename}")

                print(f"Using YOLO Repo: {yolo_repo}, Pose Repo: {pose_repo}")

                # Instantiate detector using from_pretrained
                # This requires comfyui_controlnet_aux to be installed AND ITS dependencies (like onnxruntime)
                OpenPoseEyewearMask.detector_instance = DwposeDetector.from_pretrained(
                    pose_estimator_repo_id=pose_repo, # Parameter names might differ slightly, adjust if needed
                    pose_filename=dwpose_filename,
                    det_repo_id=yolo_repo,
                    det_filename=yolo_filename,
                    device=model_management.get_torch_device() # Pass device, important for TorchScript
                )
                OpenPoseEyewearMask.current_dwpose_filename = dwpose_filename
                OpenPoseEyewearMask.current_yolo_filename = yolo_filename
                print("OpenPoseEyewearMask: DWPoseDetector initialized successfully via from_pretrained.")

            except Exception as e:
                 print(f"ERROR during DwposeDetector.from_pretrained: {e}")
                 import traceback
                 traceback.print_exc() # Print detailed traceback
                 # Set instance to None so it doesn't try to use a bad/old one
                 OpenPoseEyewearMask.detector_instance = None
                 OpenPoseEyewearMask.current_dwpose_filename = None
                 OpenPoseEyewearMask.current_yolo_filename = None
                 raise # Re-raise the error to stop execution cleanly

        return OpenPoseEyewearMask.detector_instance

    def create_mask(self, image, detect_hand=False, detect_face=False, dwpose_filename="dw-ll_ucoco_384.onnx", yolo_filename="yolox_l.onnx", confidence_threshold=0.1, dilation_kernel_size=5, dilation_iterations=2):

        if not DWPOSE_INSTALLED:
            print(f"Error: OpenPoseEyewearMask cannot run. {DWPOSE_IMPORT_ERROR}")
            batch_size, img_h, img_w, _ = image.shape
            black_mask = torch.zeros((batch_size, img_h, img_w), dtype=torch.float32, device=image.device)
            return (black_mask,)

        try:
            detector = self.initialize_detector(dwpose_filename, yolo_filename)
            if detector is None:
                 raise RuntimeError("DWPose detector could not be initialized.")

        except Exception as e:
            print(f"Error during detector setup: {e}")
            batch_size, img_h, img_w, _ = image.shape
            black_mask = torch.zeros((batch_size, img_h, img_w), dtype=torch.float32, device=image.device)
            return (black_mask,)

        # --- Processing Logic ---
        batch_size, img_h, img_w, channels = image.shape
        output_masks = []

        for i in range(batch_size):
            img_tensor = image[i]
            img_np = img_tensor.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # Ensure BGR format if needed (DWPose often expects BGR)
            # Check DWPoseDetector documentation/source if unsure, assume BGR for now
            if img_np.shape[-1] == 3: img_np_input = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif img_np.shape[-1] == 4: img_np_input = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            elif img_np.shape[-1] == 1: img_np_input = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            else: img_np_input = img_np

            current_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            try:
                # Call the detector instance
                # Adapt arguments based on DwposeDetector's actual __call__ signature
                # The preprocessor used include_hand, include_face, include_body
                # And returned image + dict when image_and_json=True
                # Let's assume we want the dict output primarily
                # The exact call might need tweaking based on comfyui_controlnet_aux version
                results_img, openpose_dict = detector(
                    img_np_input,
                    include_hand=detect_hand,
                    include_face=detect_face,
                    include_body=True # Assuming body is always needed for eyes/nose/ears
                    # image_and_json=True # Might be implicit or default? Check aux source if needed.
                )

                # Extract keypoints from the returned dictionary
                # The structure might be slightly different than assumed before
                # Check the 'openpose_dict' structure
                poses = []
                if isinstance(openpose_dict, dict) and 'candidate' in openpose_dict and 'subset' in openpose_dict:
                    # This looks like the standard OpenPose JSON output structure
                    # We need to parse 'candidate' (all detected points [x,y,conf,...])
                    # and 'subset' (which points belong to which person)
                    candidates = np.array(openpose_dict['candidate'])
                    subsets = np.array(openpose_dict['subset'])

                    if subsets.shape[0] > 0: # If any person detected
                        person_indices = subsets[0, 0:19].astype(int) # Get point indices for the first person (body_25 has 19 used points in subset)
                        keypoints = np.zeros((19, 3)) # Max 19 points needed for body_25 subset indices
                        valid_point_count = 0
                        for kpt_idx in range(19): # Iterate through possible keypoint types (0-18)
                            candidate_idx = person_indices[kpt_idx]
                            if candidate_idx != -1: # If this keypoint type exists for this person
                                keypoints[kpt_idx] = candidates[candidate_idx, 0:3] # Get x, y, conf
                                valid_point_count += 1

                        if valid_point_count > 0:
                            poses.append({'keypoints': keypoints}) # Format similar to previous assumption

                elif isinstance(openpose_dict, dict) and 'bodies' in openpose_dict:
                     # Fallback for the structure assumed previously
                     poses = openpose_dict['bodies']


                if poses:
                    # Using the first detected person's keypoints
                    keypoints = poses[0]['keypoints'] # Should be shape (N, 3) where N >= 19

                    polygon_points_indices = []
                    valid_keypoints = {}
                    for idx in self.POLYGON_ORDER_INDICES:
                        if idx < keypoints.shape[0]: # Check index bounds
                            kpt = keypoints[idx]
                            # DWPose might use BGR, coordinates are usually fine
                            x, y, conf = kpt[0], kpt[1], kpt[2]
                            # Check for valid coordinates (OpenPose often uses 0 for undetected)
                            if conf >= confidence_threshold and x > 0 and y > 0 and x < img_w and y < img_h:
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
                 print(f"Error during DWPose processing or mask creation for image {i+1}: {e}")
                 import traceback
                 traceback.print_exc()

            mask_tensor = torch.from_numpy(current_mask.astype(np.float32) / 255.0)
            output_masks.append(mask_tensor)

        final_mask_batch = torch.stack(output_masks, dim=0)
        return (final_mask_batch,)


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "OpenPoseEyewearMask": OpenPoseEyewearMask
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPoseEyewearMask": "OpenPose Eyewear Mask (DWPose)" # Renamed slightly
}
