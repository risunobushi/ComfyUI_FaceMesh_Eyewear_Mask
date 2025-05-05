import torch
import numpy as np
from PIL import Image
import cv2 # OpenCV for drawing and dilation

class MaskFromPoseKeypoints:
    # --- OpenPose Keypoint Indices (Body_25 convention often used) ---
    # We need Nose, Eyes, and Ears
    NOSE_IDX = 0
    LEYE_IDX = 15
    REYE_IDX = 16
    LEAR_IDX = 17
    REAR_IDX = 18

    # Define the order to create the polygon mask
    # Left Ear -> Left Eye -> Nose -> Right Eye -> Right Ear -> (Implicit close)
    POLYGON_ORDER_INDICES = [LEAR_IDX, LEYE_IDX, NOSE_IDX, REYE_IDX, REAR_IDX]

    # Minimum required keypoints for a basic mask (Eyes and Nose)
    REQUIRED_INDICES = [LEYE_IDX, NOSE_IDX, REYE_IDX]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Input from DWPose Estimator or similar node
                "pose_keypoints": ("POSE_KEYPOINT", ),
                # Used to get the height/width for the output mask
                "reference_image": ("IMAGE", ),
                "confidence_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "dilation_kernel_size": ("INT", {"default": 5, "min": 0, "max": 51, "step": 2}),
                "dilation_iterations": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_mask"
    CATEGORY = "image/masking/landmarks" # New category perhaps?
    OUTPUT_NODE = False

    def generate_mask(self, pose_keypoints, reference_image, confidence_threshold, dilation_kernel_size, dilation_iterations):
        # Get dimensions from the reference image
        batch_size, img_h, img_w, _ = reference_image.shape
        output_masks = []

        # Check if pose_keypoints is a list and matches batch size (or handle mismatch)
        if not isinstance(pose_keypoints, list):
             print("Warning: pose_keypoints input is not a list. Assuming single item.")
             pose_keypoints = [pose_keypoints] # Wrap in list

        num_poses = len(pose_keypoints)
        print(f"Received {num_poses} pose keypoint sets for a batch of {batch_size} images.")

        for i in range(batch_size):
            # Create a blank mask for this image
            current_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            # Get the corresponding pose data for the current image
            # Handle potential mismatch between image batch size and pose data list size
            if i < num_poses:
                pose_data = pose_keypoints[i]
            elif num_poses > 0:
                 print(f"Warning: Image batch index {i} exceeds pose data list size {num_poses}. Reusing last pose data.")
                 pose_data = pose_keypoints[-1] # Reuse the last available pose data
            else:
                 print(f"Warning: No pose data available for image index {i}. Generating black mask.")
                 pose_data = None # No data available

            keypoints = None
            if isinstance(pose_data, dict):
                # Attempt to extract keypoints, handling different structures
                if 'keypoints' in pose_data: # Structure like {'keypoints': [[x,y,c],...]}
                    keypoints = np.array(pose_data['keypoints'])
                elif 'candidate' in pose_data and 'subset' in pose_data: # Standard OpenPose JSON output
                    try:
                        candidates = np.array(pose_data['candidate'])
                        subsets = np.array(pose_data['subset'])
                        if subsets.shape[0] > 0:
                            person_indices = subsets[0, 0:19].astype(int) # First person, body_25 indices
                            temp_kpts = np.zeros((19, 3))
                            valid_count = 0
                            for kpt_idx in range(19):
                                candidate_idx = person_indices[kpt_idx]
                                if candidate_idx != -1:
                                    if candidate_idx < len(candidates):
                                        temp_kpts[kpt_idx] = candidates[candidate_idx, 0:3]
                                        valid_count += 1
                                    else:
                                         print(f"Warning: Candidate index {candidate_idx} out of bounds for candidates list (len={len(candidates)})")
                            if valid_count > 0:
                                keypoints = temp_kpts # Assign if we parsed successfully
                    except Exception as e:
                        print(f"Error parsing 'candidate'/'subset' structure: {e}")
            elif pose_data is None:
                 pass # No data, keypoints remain None
            else:
                 print(f"Warning: Unexpected type for pose_data element at index {i}: {type(pose_data)}. Expected dict.")


            if keypoints is not None and keypoints.shape[0] > max(self.POLYGON_ORDER_INDICES):
                polygon_points_indices = []
                valid_keypoints = {}

                # Extract valid keypoints based on confidence
                for idx in self.POLYGON_ORDER_INDICES:
                    # No need to check bounds again here if checked above
                    kpt = keypoints[idx]
                    x, y, conf = kpt[0], kpt[1], kpt[2]
                    # Check confidence and basic validity (non-zero coords)
                    if conf >= confidence_threshold and x > 0 and y > 0:
                         # Optional: Add boundary checks (x < img_w, y < img_h) if coords aren't guaranteed valid
                         valid_keypoints[idx] = (int(x), int(y))
                         polygon_points_indices.append(idx) # Store index if valid point found

                # Check if minimum required points are present
                has_required = all(idx in valid_keypoints for idx in self.REQUIRED_INDICES)

                if has_required and len(polygon_points_indices) >= 3:
                    # Build the polygon coordinate list using only the valid points found, in order
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
                # else: # Optional logging if needed
                #     if not has_required: print(f"MaskFromPoseKeypoints: Missing required keypoints (Eyes/Nose) for image {i}")
                #     elif len(polygon_points_indices) < 3: print(f"MaskFromPoseKeypoints: Not enough valid polygon points ({len(polygon_points_indices)}) for image {i}")

            # else: # Optional logging if needed
            #      if keypoints is None: print(f"MaskFromPoseKeypoints: No valid keypoints extracted for image {i}")
            #      else: print(f"MaskFromPoseKeypoints: Keypoints array shape {keypoints.shape} too small for required indices (max index: {max(self.POLYGON_ORDER_INDICES)}) for image {i}")


            # Convert the NumPy mask back to a Torch tensor (H, W, float32 0-1)
            mask_tensor = torch.from_numpy(current_mask.astype(np.float32) / 255.0)
            output_masks.append(mask_tensor)

        # Stack all masks in the batch
        final_mask_batch = torch.stack(output_masks, dim=0) # Shape: (B, H, W)

        return (final_mask_batch,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MaskFromPoseKeypoints": MaskFromPoseKeypoints
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromPoseKeypoints": "Mask From Pose Keypoints"
}
