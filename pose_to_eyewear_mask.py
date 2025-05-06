import torch
import numpy as np
from PIL import Image
import cv2 # OpenCV for drawing and dilation
try:
    from einops import rearrange # Need einops for reshaping flat list
    EINOPS_AVAILABLE = True
except ImportError:
    print("MaskFromFacialKeypoints: Warning - 'einops' library not found. Keypoint reshaping might fail if input is flat list.")
    EINOPS_AVAILABLE = False

class MaskFromFacialKeypoints:
    # No need to define all constants if using a direct list
    FACE_MAX_INDEX = 67 # Standard for 68 points (0-67)

    # User-provided polygon order (immediate consecutive duplicates removed)
    # Original: 1, 2, 3, 32, 31, 36, 16, 16, 17, 27, 26, 26, 25, 23, 22, 21, 20, 18, 18, 1
    POLYGON_ORDER_INDICES = [
        1, 2, 3, 32, 31, 36, 16, 17, 27, 26, 25, 23, 22, 21, 20, 18
        # Point 1 will be implicitly connected back by cv2.fillPoly
    ]

    # Define a minimal set of required points from the user's sequence
    REQUIRED_FACE_INDICES = [
        1, 16, 17, 27, 26, 18 # Check points from start, sides, top
    ]


    @classmethod
    def INPUT_TYPES(cls):
        # (INPUT_TYPES remains the same)
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT", ),
                "reference_image": ("IMAGE", ),
                "confidence_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "dilation_kernel_size": ("INT", {"default": 0, "min": 0, "max": 51, "step": 2}), # Default dilation 0
                "dilation_iterations": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1}),  # Default dilation 0
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_mask"
    CATEGORY = "image/masking/landmarks"
    OUTPUT_NODE = False

    # (reshape_keypoints helper function remains the same)
    def reshape_keypoints(self, kps_list, source_name):
        if isinstance(kps_list, list) and len(kps_list) > 0 and len(kps_list) % 3 == 0:
            try:
                num_keypoints = len(kps_list) // 3
                if EINOPS_AVAILABLE: return rearrange(np.array(kps_list), "(n c) -> n c", n=num_keypoints, c=3)
                else: return np.array(kps_list).reshape(num_keypoints, 3)
            except Exception as e: print(f"{self.__class__.__name__}: Error reshaping {source_name}: {e}"); return None
        elif isinstance(kps_list, list) and len(kps_list) > 0 and isinstance(kps_list[0], list) and len(kps_list[0]) == 3: return np.array(kps_list)
        else: print(f"{self.__class__.__name__}: {source_name} unexpected format."); return None

    def generate_mask(self, pose_keypoints, reference_image, confidence_threshold, dilation_kernel_size, dilation_iterations):
        # (Initial setup, batch handling, parsing logic remains the same)
        batch_size, img_h, img_w, _ = reference_image.shape
        output_masks = []
        if not isinstance(pose_keypoints, list): pose_keypoints = [pose_keypoints]
        num_poses = len(pose_keypoints)

        for i in range(batch_size):
            current_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            face_kps_array = None
            pose_frame = pose_keypoints[i] if i < num_poses else (pose_keypoints[-1] if num_poses > 0 else None)

            if isinstance(pose_frame, dict) and "people" in pose_frame and pose_frame["people"]:
                person_data = pose_frame["people"][0]
                if isinstance(person_data, dict) and 'face_keypoints_2d' in person_data:
                    face_kps_array = self.reshape_keypoints(person_data['face_keypoints_2d'], "Facial")

            # --- Polygon Construction ---
            polygon_coords = []
            can_draw = True
            if face_kps_array is None or face_kps_array.shape[0] <= self.FACE_MAX_INDEX:
                print(f"{self.__class__.__name__}: Image {i} - Facial keypoints insufficient or missing.")
                can_draw = False
            else:
                # Check if required points (subset of the path) meet threshold
                for idx in self.REQUIRED_FACE_INDICES:
                     # Check bounds just in case, although indices should be < 68
                     if idx >= face_kps_array.shape[0] or face_kps_array[idx, 2] < confidence_threshold:
                         print(f"{self.__class__.__name__}: Image {i} - Required facial keypoint {idx} invalid or below threshold.")
                         can_draw = False; break

                if can_draw:
                    points_ok = True
                    # Use the user-defined sequence directly
                    for index in self.POLYGON_ORDER_INDICES:
                        # Bounds check
                        if index >= face_kps_array.shape[0]:
                             print(f"{self.__class__.__name__}: Image {i} - Error: Index {index} out of bounds for Facial (Shape: {face_kps_array.shape})")
                             points_ok = False; break

                        x, y, conf = face_kps_array[index]
                        if conf >= confidence_threshold and x > 0 and y > 0:
                            polygon_coords.append((int(x), int(y)))
                        else:
                            print(f"{self.__class__.__name__}: Img {i} - Point face_{index} invalid (conf={conf:.2f},x={x},y={y}).")
                            points_ok = False; break
                    can_draw = points_ok

            # --- Drawing ---
            if can_draw and len(polygon_coords) >= 3:
                polygon_np = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
                print(f"{self.__class__.__name__}: Image {i} - Drawing user-defined polygon with {len(polygon_coords)} points.")
                cv2.fillPoly(current_mask, [polygon_np], 255) # Fill with white

                # Apply dilation if requested
                if dilation_kernel_size > 0 and dilation_iterations > 0:
                    # Ensure kernel size is odd
                    k_size = dilation_kernel_size if dilation_kernel_size % 2 != 0 else dilation_kernel_size + 1
                    kernel = np.ones((k_size, k_size), np.uint8)
                    current_mask = cv2.dilate(current_mask, kernel, iterations=dilation_iterations)
            else:
                if face_kps_array is not None: # Log only if keypoints were actually available
                    print(f"{self.__class__.__name__}: Img {i} - Skipped drawing polygon. can_draw={can_draw}, #poly_coords={len(polygon_coords)}")

            mask_tensor = torch.from_numpy(current_mask.astype(np.float32) / 255.0)
            output_masks.append(mask_tensor)

        final_mask_batch = torch.stack(output_masks, dim=0)
        return (final_mask_batch,)


# --- Node Mappings ---
# Keep the same class name used in the file for mapping
NODE_CLASS_MAPPINGS = {
    "MaskFromFacialKeypoints": MaskFromFacialKeypoints
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromFacialKeypoints": "Mask From Facial Keypoints" # Modified name
}
