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
    # --- Constants (Facial 68 Only) ---
    FACE_L_BROW_OUTER = 17; FACE_L_BROW_MID1 = 18; FACE_L_BROW_MID2 = 19; FACE_L_BROW_MID3 = 20; FACE_L_BROW_INNER = 21
    FACE_NOSE_BRIDGE_TOP = 27
    FACE_R_BROW_INNER = 22; FACE_R_BROW_MID1 = 23; FACE_R_BROW_MID2 = 24; FACE_R_BROW_MID3 = 25; FACE_R_BROW_OUTER = 26
    FACE_L_TEMPLE_APPROX = 0; FACE_R_TEMPLE_APPROX = 16
    FACE_L_NOSE_WING = 31; FACE_NOSE_BOTTOM_CENTER = 33; FACE_R_NOSE_WING = 35
    FACE_L_UPPER_CHEEK_SIDE = 1; FACE_R_UPPER_CHEEK_SIDE = 15
    FACE_MAX_INDEX = 67

    POLYGON_ORDER_INDICES = [
        FACE_L_BROW_OUTER, FACE_L_BROW_MID1, FACE_L_BROW_MID2, FACE_L_BROW_MID3, FACE_L_BROW_INNER,
        FACE_NOSE_BRIDGE_TOP,
        FACE_R_BROW_INNER, FACE_R_BROW_MID1, FACE_R_BROW_MID2, FACE_R_BROW_MID3, FACE_R_BROW_OUTER,
        FACE_R_TEMPLE_APPROX, FACE_R_UPPER_CHEEK_SIDE, FACE_R_NOSE_WING,
        FACE_NOSE_BOTTOM_CENTER,
        FACE_L_NOSE_WING, FACE_L_UPPER_CHEEK_SIDE, FACE_L_TEMPLE_APPROX,
    ]

    REQUIRED_FACE_INDICES = [
        FACE_L_BROW_OUTER, FACE_R_BROW_OUTER, FACE_NOSE_BRIDGE_TOP,
        FACE_L_TEMPLE_APPROX, FACE_R_TEMPLE_APPROX,
        FACE_L_NOSE_WING, FACE_R_NOSE_WING, FACE_NOSE_BOTTOM_CENTER
    ]

    # --- Eye and Nose Indices (68-point) ---
    NOSE_POLYGON_INDICES = [30, 31, 32, 33, 34, 35]  # Nose ridge and wings
    LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

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

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("mask_eyewear", "mask_nose", "mask_eyes", "mask_composite")
    FUNCTION = "generate_masks"
    CATEGORY = "image/masking/landmarks"
    OUTPUT_NODE = False

    def reshape_keypoints(self, kps_list, source_name):
        if isinstance(kps_list, list) and len(kps_list) > 0 and len(kps_list) % 3 == 0:
            try:
                num_keypoints = len(kps_list) // 3
                if EINOPS_AVAILABLE: return rearrange(np.array(kps_list), "(n c) -> n c", n=num_keypoints, c=3)
                else: return np.array(kps_list).reshape(num_keypoints, 3)
            except Exception as e: print(f"{self.__class__.__name__}: Error reshaping {source_name}: {e}"); return None
        elif isinstance(kps_list, list) and len(kps_list) > 0 and isinstance(kps_list[0], list) and len(kps_list[0]) == 3: return np.array(kps_list)
        else: print(f"{self.__class__.__name__}: {source_name} unexpected format."); return None

    def get_polygon_mask(self, img_h, img_w, keypoints, indices, confidence_threshold):
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        polygon_coords = []
        for idx in indices:
            x, y, conf = keypoints[idx]
            if conf >= confidence_threshold and x > 0 and y > 0:
                polygon_coords.append((int(x), int(y)))
        if len(polygon_coords) >= 3:
            polygon_np = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [polygon_np], 255)
        return mask

    def get_ellipse_mask(self, img_h, img_w, keypoints, indices, confidence_threshold):
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        valid_points = []
        for idx in indices:
            x, y, conf = keypoints[idx]
            if conf >= confidence_threshold and x > 0 and y > 0:
                valid_points.append((int(x), int(y)))
        if len(valid_points) >= 3:
            pts = np.array(valid_points)
            center = tuple(np.mean(pts, axis=0).astype(int))
            axes = tuple(np.maximum(np.std(pts, axis=0).astype(int) * 2, 5))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        return mask

    def generate_masks(self, pose_keypoints, reference_image, confidence_threshold, dilation_kernel_size, dilation_iterations):
        batch_size, img_h, img_w, _ = reference_image.shape
        masks_eyewear, masks_nose, masks_eyes, masks_composite = [], [], [], []
        if not isinstance(pose_keypoints, list): pose_keypoints = [pose_keypoints]
        num_poses = len(pose_keypoints)

        for i in range(batch_size):
            mask_eyewear = np.zeros((img_h, img_w), dtype=np.uint8)
            mask_nose = np.zeros((img_h, img_w), dtype=np.uint8)
            mask_eyes = np.zeros((img_h, img_w), dtype=np.uint8)
            face_kps_array = None
            pose_frame = pose_keypoints[i] if i < num_poses else (pose_keypoints[-1] if num_poses > 0 else None)

            if isinstance(pose_frame, dict) and "people" in pose_frame and pose_frame["people"]:
                person_data = pose_frame["people"][0]
                if isinstance(person_data, dict) and 'face_keypoints_2d' in person_data:
                    face_kps_array = self.reshape_keypoints(person_data['face_keypoints_2d'], "Facial")

            can_draw = True
            if face_kps_array is None or face_kps_array.shape[0] <= self.FACE_MAX_INDEX:
                can_draw = False
            else:
                # Eyewear region
                polygon_coords = []
                for index in self.POLYGON_ORDER_INDICES:
                    x, y, conf = face_kps_array[index]
                    if conf >= confidence_threshold and x > 0 and y > 0:
                        polygon_coords.append((int(x), int(y)))
                if len(polygon_coords) >= 3:
                    polygon_np = np.array(polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask_eyewear, [polygon_np], 255)
                # Nose mask (polygon or ellipse)
                mask_nose = self.get_polygon_mask(img_h, img_w, face_kps_array, self.NOSE_POLYGON_INDICES, confidence_threshold)
                # Eyes mask (union of left and right eye polygons)
                mask_left_eye = self.get_polygon_mask(img_h, img_w, face_kps_array, self.LEFT_EYE_INDICES, confidence_threshold)
                mask_right_eye = self.get_polygon_mask(img_h, img_w, face_kps_array, self.RIGHT_EYE_INDICES, confidence_threshold)
                mask_eyes = cv2.bitwise_or(mask_left_eye, mask_right_eye)

            # Dilation (if needed)
            if dilation_kernel_size > 0 and dilation_iterations > 0:
                k_size = dilation_kernel_size if dilation_kernel_size % 2 != 0 else dilation_kernel_size + 1
                kernel = np.ones((k_size, k_size), np.uint8)
                mask_eyewear = cv2.dilate(mask_eyewear, kernel, iterations=dilation_iterations)
                mask_nose = cv2.dilate(mask_nose, kernel, iterations=dilation_iterations)
                mask_eyes = cv2.dilate(mask_eyes, kernel, iterations=dilation_iterations)

            # Composite: eyewear minus nose and eyes
            mask_composite = cv2.bitwise_and(mask_eyewear, cv2.bitwise_not(cv2.bitwise_or(mask_nose, mask_eyes)))

            # Convert to torch
            masks_eyewear.append(torch.from_numpy(mask_eyewear.astype(np.float32) / 255.0))
            masks_nose.append(torch.from_numpy(mask_nose.astype(np.float32) / 255.0))
            masks_eyes.append(torch.from_numpy(mask_eyes.astype(np.float32) / 255.0))
            masks_composite.append(torch.from_numpy(mask_composite.astype(np.float32) / 255.0))

        return (
            torch.stack(masks_eyewear, dim=0),
            torch.stack(masks_nose, dim=0),
            torch.stack(masks_eyes, dim=0),
            torch.stack(masks_composite, dim=0),
        )

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "MaskFromFacialKeypoints": MaskFromFacialKeypoints
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromFacialKeypoints": "Mask From Facial Keypoints"
}
