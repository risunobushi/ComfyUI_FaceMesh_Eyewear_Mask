import torch
import numpy as np
from PIL import Image
import cv2 # OpenCV for drawing and dilation
try:
    from einops import rearrange # Need einops for reshaping flat list
    EINOPS_AVAILABLE = True
except ImportError:
    print("MaskFromPoseKeypoints: Warning - 'einops' library not found. Keypoint reshaping might fail if input is flat list.")
    EINOPS_AVAILABLE = False

class MaskFromFacialKeypoints: # Renamed class slightly
    # --- Keypoint Indices (Facial 68 Only) ---
    # Brows (Upper Edge)
    FACE_L_BROW_OUT = 17
    FACE_L_BROW_1 = 18
    FACE_L_BROW_2 = 19
    FACE_L_BROW_3 = 20
    FACE_L_BROW_IN = 21
    FACE_R_BROW_IN = 22
    FACE_R_BROW_1 = 23
    FACE_R_BROW_2 = 24
    FACE_R_BROW_3 = 25
    FACE_R_BROW_OUT = 26
    FACE_NOSE_BRIDGE_TOP = 27
    # Jawline/Cheeks (Sides and Bottom Edge)
    FACE_L_JAW_1 = 1
    FACE_L_JAW_2 = 2
    FACE_L_JAW_3 = 3
    FACE_L_JAW_4 = 4
    FACE_L_JAW_5 = 5
    FACE_CHIN = 8
    FACE_R_JAW_5 = 11
    FACE_R_JAW_4 = 12
    FACE_R_JAW_3 = 13
    FACE_R_JAW_2 = 14
    FACE_R_JAW_1 = 15
    FACE_MAX_INDEX = 67

    # Define the polygon order using only Facial 68 points
    POLYGON_ORDER_INDICES = [
        # Start near bottom-left jaw, go up
        FACE_L_JAW_3,           # 3
        FACE_L_JAW_2,           # 2
        FACE_L_JAW_1,           # 1
        # Upper Edge (Eyebrows)
        FACE_L_BROW_OUT,        # 17
        FACE_L_BROW_1,          # 18
        FACE_L_BROW_2,          # 19
        FACE_L_BROW_3,          # 20
        FACE_L_BROW_IN,         # 21
        FACE_NOSE_BRIDGE_TOP,   # 27
        FACE_R_BROW_IN,         # 22
        FACE_R_BROW_1,          # 23
        FACE_R_BROW_2,          # 24
        FACE_R_BROW_3,          # 25
        FACE_R_BROW_OUT,        # 26
        # Go down right jaw
        FACE_R_JAW_1,           # 15
        FACE_R_JAW_2,           # 14
        FACE_R_JAW_3,           # 13
        # Bottom Edge (Jawline)
        FACE_R_JAW_4,           # 12
        FACE_R_JAW_5,           # 11
        FACE_CHIN,              # 8
        FACE_L_JAW_5,           # 5
        FACE_L_JAW_4,           # 4
        # Implicit close back to FACE_L_JAW_3
    ]

    # Basic check points (e.g., ensure brows and nose bridge are detected)
    REQUIRED_FACE_INDICES = [
        FACE_L_BROW_OUT, FACE_R_BROW_OUT, FACE_NOSE_BRIDGE_TOP, FACE_CHIN
    ]


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT", ), # Still takes full pose data
                "reference_image": ("IMAGE", ),
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

    def reshape_keypoints(self, kps_list, source_name):
        """Helper to reshape flat list [x,y,c,...] to Nx3 numpy array."""
        if isinstance(kps_list, list) and len(kps_list) > 0 and len(kps_list) % 3 == 0:
            try:
                num_keypoints = len(kps_list) // 3
                if EINOPS_AVAILABLE:
                    return rearrange(np.array(kps_list), "(n c) -> n c", n=num_keypoints, c=3)
                else:
                    return np.array(kps_list).reshape(num_keypoints, 3)
            except Exception as e:
                print(f"{self.__class__.__name__}: Error reshaping {source_name} keypoints list: {e}")
                return None
        elif isinstance(kps_list, list) and len(kps_list) > 0 and isinstance(kps_list[0], list) and len(kps_list[0]) == 3:
             return np.array(kps_list)
        else:
             print(f"{self.__class__.__name__}: {source_name} keypoints have unexpected format or length.")
             return None

    def generate_mask(self, pose_keypoints, reference_image, confidence_threshold, dilation_kernel_size, dilation_iterations):
        batch_size, img_h, img_w, _ = reference_image.shape
        output_masks = []

        if not isinstance(pose_keypoints, list):
             pose_keypoints = [pose_keypoints]

        num_poses = len(pose_keypoints)
        # print(f"{self.__class__.__name__}: Received {num_poses} pose sets for batch of {batch_size}.") # Less verbose

        for i in range(batch_size):
            current_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            face_kps_array = None # Only need facial points now

            # --- Parsing ---
            pose_frame = None
            if i < num_poses: pose_frame = pose_keypoints[i]
            elif num_poses > 0: pose_frame = pose_keypoints[-1] # Reuse last

            if isinstance(pose_frame, dict) and "people" in pose_frame and pose_frame["people"]:
                person_data = pose_frame["people"][0]
                if isinstance(person_data, dict):
                    if 'face_keypoints_2d' in person_data:
                        face_kps_array = self.reshape_keypoints(person_data['face_keypoints_2d'], "Facial")
                    # else: print(f"{self.__class__.__name__}: Image {i} - Person 0 missing 'face_keypoints_2d'.") # Can omit logs if needed
                # else: print(f"{self.__class__.__name__}: Image {i} - Person 0 data is not a dict.")
            # else: print(f"{self.__class__.__name__}: Image {i} - pose_frame structure invalid or 'people' list empty.")
            # --- End Parsing ---

            # --- Polygon Construction ---
            polygon_coords = []
            can_draw = True # Assume valid

            if face_kps_array is None or face_kps_array.shape[0] <= self.FACE_MAX_INDEX:
                print(f"{self.__class__.__name__}: Image {i} - Facial keypoints missing or insufficient (Shape: {face_kps_array.shape if face_kps_array is not None else 'None'}).")
                can_draw = False
            else:
                # Optional: Check if a few basic required points exist first
                for idx in self.REQUIRED_FACE_INDICES:
                    if face_kps_array[idx, 2] < confidence_threshold:
                        print(f"{self.__class__.__name__}: Image {i} - Required facial keypoint {idx} below threshold.")
                        can_draw = False; break

            if can_draw:
                # print(f"{self.__class__.__name__}: Image {i} - Attempting to build polygon using facial points.") # Less verbose
                points_ok = True
                for index in self.POLYGON_ORDER_INDICES: # Iterate through the facial-only indices
                    # Bounds check already done essentially
                    x, y, conf = face_kps_array[index]

                    if conf >= confidence_threshold and x > 0 and y > 0:
                        polygon_coords.append((int(x), int(y)))
                    else:
                        print(f"{self.__class__.__name__}: Image {i} - Point face_{index} below threshold ({conf:.2f}) or invalid coords. Cannot form complete polygon.")
                        points_ok = False
                        break
                can_draw = points_ok

            # --- Drawing ---
            if can_draw and len(polygon_coords) >= 3:
                polygon_np = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
                # print(f"{self.__class__.__name__}: Image {i} - Drawing facial polygon with {len(polygon_coords)} points.") # Less verbose
                cv2.fillPoly(current_mask, [polygon_np], 255) # White

                if dilation_kernel_size > 0 and dilation_iterations > 0:
                    k_size = dilation_kernel_size if dilation_kernel_size % 2 != 0 else dilation_kernel_size + 1
                    kernel = np.ones((k_size, k_size), np.uint8)
                    current_mask = cv2.dilate(current_mask, kernel, iterations=dilation_iterations)
            else:
                # Log only if expected to draw but couldn't
                if face_kps_array is not None and face_kps_array.shape[0] > self.FACE_MAX_INDEX:
                     if not can_draw:
                         print(f"{self.__class__.__name__}: Image {i} - Skipped drawing facial polygon due to missing/invalid points in defined path.")
                     elif len(polygon_coords) < 3:
                         print(f"{self.__class__.__name__}: Image {i} - Skipped drawing facial polygon, not enough valid points gathered ({len(polygon_coords)} < 3).")


            mask_tensor = torch.from_numpy(current_mask.astype(np.float32) / 255.0)
            output_masks.append(mask_tensor)

        final_mask_batch = torch.stack(output_masks, dim=0)
        return (final_mask_batch,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "MaskFromFacialKeypoints": MaskFromFacialKeypoints # Updated class name
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromFacialKeypoints": "Mask From Facial Keypoints" # Updated display name
}
