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
    # COCO 18 format (from 'pose_keypoints_2d') - Indices 0-17
    COCO_NOSE = 0
    COCO_LEYE = 15
    COCO_REYE = 14 # Corrected
    COCO_LEAR = 17
    COCO_REAR = 16 # Corrected
    COCO_MAX_INDEX = 17

    # Facial 68 format (from 'face_keypoints_2d') - Selected points for outline
    FACE_L_BROW_OUT = 17
    FACE_L_BROW_1 = 18
    FACE_L_BROW_2 = 19
    FACE_L_BROW_3 = 20
    FACE_L_BROW_IN = 21 # Inner
    FACE_R_BROW_IN = 22 # Inner
    FACE_R_BROW_1 = 23
    FACE_R_BROW_2 = 24
    FACE_R_BROW_3 = 25
    FACE_R_BROW_OUT = 26
    FACE_NOSE_BRIDGE_TOP = 27
    FACE_L_NOSE_WING_OUT = 31 # Near lower boundary
    FACE_NOSE_BOTTOM_MID = 33
    FACE_R_NOSE_WING_OUT = 35 # Near lower boundary
    FACE_L_CHEEK_SIDE = 1  # Approx side/lower boundary start (Jawline start)
    FACE_R_CHEEK_SIDE = 15 # Approx side/lower boundary start (Jawline start)
    FACE_MAX_INDEX = 67

    # Define the polygon order using a mix of points (simpler bottom edge)
    POLYGON_ORDER = [
        ("coco", COCO_LEAR),            # 17: L Ear
        ("face", FACE_L_BROW_OUT),      # 17: L Outer Brow Top
        ("face", FACE_L_BROW_1),        # 18
        ("face", FACE_L_BROW_2),        # 19
        ("face", FACE_L_BROW_3),        # 20
        ("face", FACE_L_BROW_IN),       # 21: L Inner Brow Top
        ("face", FACE_NOSE_BRIDGE_TOP), # 27: Nose Bridge Top
        ("face", FACE_R_BROW_IN),       # 22: R Inner Brow Top
        ("face", FACE_R_BROW_1),        # 23
        ("face", FACE_R_BROW_2),        # 24
        ("face", FACE_R_BROW_3),        # 25
        ("face", FACE_R_BROW_OUT),      # 26: R Outer Brow Top
        ("coco", COCO_REAR),            # 16: R Ear
        ("face", FACE_R_CHEEK_SIDE),    # 15: R Cheek Side (approx under ear)
        ("face", FACE_R_NOSE_WING_OUT), # 35: R Nose Wing/Under Eye
        ("face", FACE_NOSE_BOTTOM_MID), # 33: Nose Bottom Mid
        ("face", FACE_L_NOSE_WING_OUT), # 31: L Nose Wing/Under Eye
        ("face", FACE_L_CHEEK_SIDE)     # 1: L Cheek Side (approx under ear)
        # Implicit close back to coco 17
    ]

    # Minimum required keypoints for basic function (Eyes/Nose from COCO)
    REQUIRED_COCO_INDICES = [COCO_LEYE, COCO_NOSE, COCO_REYE] # 15, 0, 14

    @classmethod
    def INPUT_TYPES(cls):
        # (INPUT_TYPES remains the same)
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

    # (reshape_keypoints helper function remains the same)
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
                print(f"MaskFromPoseKeypoints: Error reshaping {source_name} keypoints list: {e}")
                return None
        elif isinstance(kps_list, list) and len(kps_list) > 0 and isinstance(kps_list[0], list) and len(kps_list[0]) == 3:
             return np.array(kps_list)
        else:
             print(f"MaskFromPoseKeypoints: {source_name} keypoints have unexpected format or length.")
             return None

    def generate_mask(self, pose_keypoints, reference_image, confidence_threshold, dilation_kernel_size, dilation_iterations):
        # (Initial setup, batch handling, parsing logic remains the same)
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
            can_draw = True # Assume valid until proven otherwise

            # Check if required arrays are available
            if coco_kps_array is None or face_kps_array is None:
                 print(f"MaskFromPoseKeypoints: Image {i} - Missing COCO or Facial keypoints array.")
                 can_draw = False
            elif coco_kps_array.shape[0] <= self.COCO_MAX_INDEX or face_kps_array.shape[0] <= self.FACE_MAX_INDEX:
                 print(f"MaskFromPoseKeypoints: Image {i} - COCO ({coco_kps_array.shape[0]}) or Facial ({face_kps_array.shape[0]}) keypoints array too small.")
                 can_draw = False
            else:
                # Check basic COCO requirements
                for idx in self.REQUIRED_COCO_INDICES:
                    if coco_kps_array[idx, 2] < confidence_threshold:
                        print(f"MaskFromPoseKeypoints: Image {i} - Required COCO keypoint {idx} below threshold.")
                        can_draw = False; break


            if can_draw:
                print(f"MaskFromPoseKeypoints: Image {i} - Attempting to build polygon using combined points.")
                points_ok = True
                for point_type, index in self.POLYGON_ORDER:
                    kps_array = coco_kps_array if point_type == "coco" else face_kps_array
                    point_name = f"{point_type}_{index}"

                    # Bounds check already done essentially, but safe to leave
                    if index >= kps_array.shape[0]:
                        print(f"MaskFromPoseKeypoints: Image {i} - Error: Index {index} out of bounds for {point_type} (Shape: {kps_array.shape})")
                        points_ok = False; break

                    x, y, conf = kps_array[index]

                    if conf >= confidence_threshold and x > 0 and y > 0:
                        polygon_coords.append((int(x), int(y)))
                    else:
                        print(f"MaskFromPoseKeypoints: Image {i} - Point {point_name} below threshold ({conf:.2f}) or invalid coords (x={x}, y={y}). Cannot form complete polygon.")
                        points_ok = False
                        break
                can_draw = points_ok # Update can_draw based on individual point checks

            # --- Drawing ---
            if can_draw and len(polygon_coords) >= 3:
                polygon_np = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
                print(f"MaskFromPoseKeypoints: Image {i} - Drawing refined polygon with {len(polygon_coords)} points.")
                cv2.fillPoly(current_mask, [polygon_np], 255) # Fill with white

                if dilation_kernel_size > 0 and dilation_iterations > 0:
                    k_size = dilation_kernel_size if dilation_kernel_size % 2 != 0 else dilation_kernel_size + 1
                    kernel = np.ones((k_size, k_size), np.uint8)
                    current_mask = cv2.dilate(current_mask, kernel, iterations=dilation_iterations)
            else:
                 if not can_draw:
                     print(f"MaskFromPoseKeypoints: Image {i} - Skipped drawing polygon due to missing/invalid points in defined path.")
                 elif len(polygon_coords) < 3:
                     print(f"MaskFromPoseKeypoints: Image {i} - Skipped drawing polygon, not enough valid points gathered ({len(polygon_coords)} < 3).")


            mask_tensor = torch.from_numpy(current_mask.astype(np.float32) / 255.0)
            output_masks.append(mask_tensor)

        final_mask_batch = torch.stack(output_masks, dim=0)
        return (final_mask_batch,)


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "MaskFromPoseKeypoints": MaskFromPoseKeypoints
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromPoseKeypoints": "Mask From Pose Keypoints (Refined)"
}
