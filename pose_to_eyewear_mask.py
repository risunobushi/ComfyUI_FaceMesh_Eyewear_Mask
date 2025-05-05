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

class MaskFromPoseKeypoints:
    # --- Keypoint Indices ---
    # COCO 18 format (from 'pose_keypoints_2d')
    COCO_NOSE = 0
    COCO_LEYE = 15
    COCO_REYE = 14
    COCO_LEAR = 17
    COCO_REAR = 16
    COCO_MAX_INDEX = 17

    # Facial 68 format (from 'face_keypoints_2d') - Selected points
    FACE_L_BROW_OUT = 17
    FACE_L_BROW_IN = 21
    FACE_R_BROW_IN = 22
    FACE_R_BROW_OUT = 26
    FACE_NOSE_BRIDGE_TOP = 27
    FACE_NOSE_BOTTOM_MID = 33
    FACE_L_NOSE_WING_OUT = 31 # Near lower boundary
    FACE_R_NOSE_WING_OUT = 35 # Near lower boundary
    FACE_L_CHEEK_SIDE = 1  # Approx side/lower boundary start
    FACE_R_CHEEK_SIDE = 15 # Approx side/lower boundary start
    FACE_MAX_INDEX = 67

    # Define the polygon order using a mix of points
    # Goal: Trace L_Ear -> L_Brow -> NoseBridge -> R_Brow -> R_Ear -> R_Cheek -> NoseBottom -> L_Cheek -> L_Ear
    POLYGON_ORDER = [
        ("coco", COCO_LEAR),         # Start Left Ear
        ("face", FACE_L_BROW_OUT),   # Up to Left Outer Brow
        #("face", FACE_L_BROW_IN),   # Across Left Brow (optional refinement)
        ("face", FACE_NOSE_BRIDGE_TOP),# Across Nose Bridge Top
        #("face", FACE_R_BROW_IN),   # Across Right Brow (optional refinement)
        ("face", FACE_R_BROW_OUT),   # To Right Outer Brow
        ("coco", COCO_REAR),         # Down to Right Ear
        ("face", FACE_R_CHEEK_SIDE), # Down/In along Right Cheek/Jaw start
        ("face", FACE_R_NOSE_WING_OUT),# Inward under eye/nose wing right
        ("face", FACE_NOSE_BOTTOM_MID),# Across bottom of nose
        ("face", FACE_L_NOSE_WING_OUT),# Outward under eye/nose wing left
        ("face", FACE_L_CHEEK_SIDE), # Out/Up along Left Cheek/Jaw start
        #("coco", COCO_LEAR)         # Implicitly closed by fillPoly
    ]

    # Minimum required keypoints for basic function (still Eyes/Nose from COCO)
    REQUIRED_COCO_INDICES = [COCO_LEYE, COCO_NOSE, COCO_REYE] # 15, 0, 14

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT", ),
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
                    # Use einops if available
                    return rearrange(np.array(kps_list), "(n c) -> n c", n=num_keypoints, c=3)
                else:
                    # Fallback to numpy reshape
                    return np.array(kps_list).reshape(num_keypoints, 3)
            except Exception as e:
                print(f"MaskFromPoseKeypoints: Error reshaping {source_name} keypoints list: {e}")
                return None
        # Handle case where it might already be [[x,y,c], ...]
        elif isinstance(kps_list, list) and len(kps_list) > 0 and isinstance(kps_list[0], list) and len(kps_list[0]) == 3:
             return np.array(kps_list)
        else:
             print(f"MaskFromPoseKeypoints: {source_name} keypoints have unexpected format or length.")
             return None

    def generate_mask(self, pose_keypoints, reference_image, confidence_threshold, dilation_kernel_size, dilation_iterations):
        batch_size, img_h, img_w, _ = reference_image.shape
        output_masks = []

        if not isinstance(pose_keypoints, list):
             pose_keypoints = [pose_keypoints]

        num_poses = len(pose_keypoints)
        print(f"MaskFromPoseKeypoints: Received {num_poses} pose sets for batch of {batch_size}.")

        for i in range(batch_size):
            current_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            coco_kps_array = None
            face_kps_array = None

            # --- Parsing ---
            pose_frame = None
            if i < num_poses: pose_frame = pose_keypoints[i]
            elif num_poses > 0: pose_frame = pose_keypoints[-1]; print(f"MaskFromPoseKeypoints: Warning - Reusing last pose data for image {i}.")

            if isinstance(pose_frame, dict) and "people" in pose_frame and pose_frame["people"]:
                person_data = pose_frame["people"][0] # First person
                if isinstance(person_data, dict):
                    if 'pose_keypoints_2d' in person_data:
                        coco_kps_array = self.reshape_keypoints(person_data['pose_keypoints_2d'], "COCO")
                    else: print(f"MaskFromPoseKeypoints: Image {i} - Person 0 missing 'pose_keypoints_2d'.")

                    if 'face_keypoints_2d' in person_data:
                        face_kps_array = self.reshape_keypoints(person_data['face_keypoints_2d'], "Facial")
                    else: print(f"MaskFromPoseKeypoints: Image {i} - Person 0 missing 'face_keypoints_2d'.")
                else: print(f"MaskFromPoseKeypoints: Image {i} - Person 0 data is not a dict.")
            else: print(f"MaskFromPoseKeypoints: Image {i} - pose_frame structure invalid or 'people' list empty.")
            # --- End Parsing ---


            # --- Polygon Construction ---
            polygon_coords = []
            all_points_valid = True # Assume valid until proven otherwise

            # Check basic requirements first
            if coco_kps_array is None or coco_kps_array.shape[0] <= self.COCO_MAX_INDEX:
                print(f"MaskFromPoseKeypoints: Image {i} - COCO keypoints missing or insufficient.")
                all_points_valid = False
            else:
                # Check if required COCO points (Eyes, Nose) meet confidence
                required_coco_valid = True
                for idx in self.REQUIRED_COCO_INDICES:
                    if coco_kps_array[idx, 2] < confidence_threshold:
                        print(f"MaskFromPoseKeypoints: Image {i} - Required COCO keypoint {idx} below threshold ({coco_kps_array[idx, 2]:.2f} < {confidence_threshold})")
                        required_coco_valid = False
                        break
                if not required_coco_valid:
                    all_points_valid = False


            if all_points_valid and (face_kps_array is None or face_kps_array.shape[0] <= self.FACE_MAX_INDEX):
                print(f"MaskFromPoseKeypoints: Image {i} - Facial keypoints missing or insufficient for refined mask.")
                # Decide: Fallback to COCO-only mask or fail completely? Let's fail for now if face points requested implicitly.
                # To fallback, you'd construct a polygon using only COCO points here.
                all_points_valid = False


            if all_points_valid:
                print(f"MaskFromPoseKeypoints: Image {i} - Attempting to build polygon using combined points.")
                point_validity_checks = {} # For logging
                for point_type, index in self.POLYGON_ORDER:
                    kps_array = coco_kps_array if point_type == "coco" else face_kps_array
                    point_name = f"{point_type}_{index}"

                    # Check index bounds (redundant if initial check passed, but safe)
                    if index >= kps_array.shape[0]:
                        print(f"MaskFromPoseKeypoints: Image {i} - Error: Index {index} out of bounds for {point_type} (Shape: {kps_array.shape})")
                        all_points_valid = False; break

                    x, y, conf = kps_array[index]

                    if conf >= confidence_threshold and x > 0 and y > 0:
                        polygon_coords.append((int(x), int(y)))
                        point_validity_checks[point_name] = True
                    else:
                        print(f"MaskFromPoseKeypoints: Image {i} - Point {point_name} below threshold ({conf:.2f}) or invalid coords (x={x}, y={y}). Cannot form complete polygon.")
                        point_validity_checks[point_name] = False
                        all_points_valid = False # Need all points in the defined path
                        break # Stop trying to build this polygon

            # --- Drawing ---
            if all_points_valid and len(polygon_coords) >= 3:
                polygon_np = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
                print(f"MaskFromPoseKeypoints: Image {i} - Drawing refined polygon with {len(polygon_coords)} points.")
                cv2.fillPoly(current_mask, [polygon_np], 255)

                if dilation_kernel_size > 0 and dilation_iterations > 0:
                    k_size = dilation_kernel_size if dilation_kernel_size % 2 != 0 else dilation_kernel_size + 1
                    kernel = np.ones((k_size, k_size), np.uint8)
                    current_mask = cv2.dilate(current_mask, kernel, iterations=dilation_iterations)
            else:
                 if not all_points_valid:
                     print(f"MaskFromPoseKeypoints: Image {i} - Skipped drawing refined polygon due to missing/invalid points.")
                 elif len(polygon_coords) < 3:
                     print(f"MaskFromPoseKeypoints: Image {i} - Skipped drawing refined polygon, not enough valid points ({len(polygon_coords)} < 3).")


            mask_tensor = torch.from_numpy(current_mask.astype(np.float32) / 255.0)
            output_masks.append(mask_tensor)

        final_mask_batch = torch.stack(output_masks, dim=0)
        return (final_mask_batch,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "MaskFromPoseKeypoints": MaskFromPoseKeypoints
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromPoseKeypoints": "Mask From Pose Keypoints (Refined)" # Updated name
}
