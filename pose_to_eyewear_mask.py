import torch
import numpy as np
from PIL import Image
import cv2 # OpenCV for drawing and dilation
from einops import rearrange # Need einops for reshaping flat list

class MaskFromPoseKeypoints:
    # OpenPose Keypoint Indices (Body_25 standard uses 25 points, 0-24)
    # DWPose output based on COCO has 18 points (0-17), matching the indices below.
    NOSE_IDX = 0
    LEYE_IDX = 15 # Index for Left Eye in COCO/DWPose output
    REYE_IDX = 16 # Index for Right Eye in COCO/DWPose output
    LEAR_IDX = 17 # Index for Left Ear in COCO/DWPose output
    REAR_IDX = 18 # Index for Right Ear - DWPose uses 18 keypoints (0-17), so this index is OUT OF BOUNDS for standard DWPose output.

    # Revised polygon order using ONLY valid DWPose indices (0-17)
    # Left Ear (17) -> Left Eye (15) -> Nose (0) -> Right Eye (16) -> (Close)
    # We cannot include Right Ear (18) as it doesn't exist in the 18-point format.
    POLYGON_ORDER_INDICES = [LEAR_IDX, LEYE_IDX, NOSE_IDX, REYE_IDX]

    # Minimum required keypoints for a basic mask (Eyes and Nose)
    REQUIRED_INDICES = [LEYE_IDX, NOSE_IDX, REYE_IDX] # Indices 15, 0, 16

    # Maximum valid index for DWPose/COCO 18-point format
    MAX_DWPOSE_INDEX = 17


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT", ),
                "reference_image": ("IMAGE", ), # Used for dimensions
                "confidence_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "dilation_kernel_size": ("INT", {"default": 5, "min": 0, "max": 51, "step": 2}),
                "dilation_iterations": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_mask"
    CATEGORY = "image/masking/landmarks"
    OUTPUT_NODE = False

    def generate_mask(self, pose_keypoints, reference_image, confidence_threshold, dilation_kernel_size, dilation_iterations):
        batch_size, img_h, img_w, _ = reference_image.shape
        output_masks = []

        if not isinstance(pose_keypoints, list):
             print("MaskFromPoseKeypoints: Warning - pose_keypoints input is not a list. Wrapping it.")
             pose_keypoints = [pose_keypoints]

        num_poses = len(pose_keypoints)
        print(f"MaskFromPoseKeypoints: Received {num_poses} pose keypoint sets for a batch of {batch_size} images.")

        for i in range(batch_size):
            current_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            keypoints_array = None # Will hold the Nx3 numpy array

            # --- Start Revised Parsing ---
            pose_frame = None
            if i < num_poses:
                pose_frame = pose_keypoints[i]
            elif num_poses > 0:
                 print(f"MaskFromPoseKeypoints: Warning - Image batch index {i} exceeds pose data list size {num_poses}. Reusing last pose data.")
                 pose_frame = pose_keypoints[-1]

            if isinstance(pose_frame, dict) and "people" in pose_frame:
                people = pose_frame["people"]
                if people: # Check if list is not empty
                    person_data = people[0] # Get the first person
                    if isinstance(person_data, dict) and 'pose_keypoints_2d' in person_data:
                        raw_kps_list = person_data['pose_keypoints_2d']
                        # Check if it's a flat list [x,y,c, x,y,c,...]
                        if isinstance(raw_kps_list, list) and len(raw_kps_list) > 0 and len(raw_kps_list) % 3 == 0:
                            try:
                                # Reshape the flat list into (N, 3) array
                                num_keypoints = len(raw_kps_list) // 3
                                keypoints_array = np.array(raw_kps_list).reshape(num_keypoints, 3)
                                print(f"MaskFromPoseKeypoints: Image {i} - Successfully parsed {num_keypoints} keypoints for person 0.")
                            except Exception as e:
                                print(f"MaskFromPoseKeypoints: Image {i} - Error reshaping keypoints list: {e}")
                        # Handle case where it might already be [[x,y,c], ...] (less likely based on aux code)
                        elif isinstance(raw_kps_list, list) and len(raw_kps_list) > 0 and isinstance(raw_kps_list[0], list) and len(raw_kps_list[0]) == 3:
                             keypoints_array = np.array(raw_kps_list)
                             print(f"MaskFromPoseKeypoints: Image {i} - Parsed keypoints as list of lists for person 0.")
                        else:
                             print(f"MaskFromPoseKeypoints: Image {i} - 'pose_keypoints_2d' has unexpected format or length: {raw_kps_list}")
                    else:
                        print(f"MaskFromPoseKeypoints: Image {i} - First person dict missing 'pose_keypoints_2d' key.")
                else:
                    print(f"MaskFromPoseKeypoints: Image {i} - 'people' list is empty.")
            else:
                print(f"MaskFromPoseKeypoints: Image {i} - pose_frame is not a dict or missing 'people' key.")
            # --- End Revised Parsing ---


            # --- Polygon Drawing Logic (using keypoints_array) ---
            if keypoints_array is not None and keypoints_array.shape[0] > self.MAX_DWPOSE_INDEX: # Ensure enough points for indices used
                polygon_points_indices = []
                valid_keypoints = {}

                for idx in self.POLYGON_ORDER_INDICES: # Use the VALID indices [17, 15, 0, 16]
                    # We already checked shape[0] > MAX_DWPOSE_INDEX (17), so idx 17 is safe if check passes
                    kpt = keypoints_array[idx]
                    x, y, conf = kpt[0], kpt[1], kpt[2]

                    if conf >= confidence_threshold and x > 0 and y > 0:
                        valid_keypoints[idx] = (int(x), int(y))
                        polygon_points_indices.append(idx)

                has_required = all(idx in valid_keypoints for idx in self.REQUIRED_INDICES) # Check for 15, 0, 16

                # We now only need 3 points minimum (L_Ear, L_Eye, Nose, R_Eye)
                if has_required and len(polygon_points_indices) >= 3:
                    polygon_coords = [valid_keypoints[idx] for idx in polygon_points_indices] # Get coords in order [17, 15, 0, 16] (if all valid)
                    polygon_np = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))

                    print(f"MaskFromPoseKeypoints: Image {i} - Drawing polygon with {len(polygon_coords)} valid points.")
                    cv2.fillPoly(current_mask, [polygon_np], 255)

                    if dilation_kernel_size > 0 and dilation_iterations > 0:
                        k_size = dilation_kernel_size if dilation_kernel_size % 2 != 0 else dilation_kernel_size + 1
                        kernel = np.ones((k_size, k_size), np.uint8)
                        current_mask = cv2.dilate(current_mask, kernel, iterations=dilation_iterations)
                else:
                     missing_req = [idx for idx in self.REQUIRED_INDICES if idx not in valid_keypoints]
                     print(f"MaskFromPoseKeypoints: Image {i} - Failed to draw polygon. HasRequired={has_required} (Missing: {missing_req}), NumValidPoints={len(polygon_points_indices)}")

            else:
                kpt_count = keypoints_array.shape[0] if keypoints_array is not None else 0
                print(f"MaskFromPoseKeypoints: Image {i} - Skipping polygon draw. Keypoints array is None or too small (Count: {kpt_count}, Max Index Needed: {self.MAX_DWPOSE_INDEX}).")


            mask_tensor = torch.from_numpy(current_mask.astype(np.float32) / 255.0)
            output_masks.append(mask_tensor)

        final_mask_batch = torch.stack(output_masks, dim=0)
        return (final_mask_batch,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "MaskFromPoseKeypoints": MaskFromPoseKeypoints
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromPoseKeypoints": "Mask From Pose Keypoints"
}
