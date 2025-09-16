import torch
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp

MAX_RESOLUTION = 4096

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class MediaPipeFaceToMask:
    # MediaPipe face mesh landmark indices for different facial regions
    # These are based on the MediaPipe face mesh 468 landmark model

    # Face contour indices (outer face boundary)
    FACE_INDICES = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
    ]

    # Mouth region indices
    MOUTH_INDICES = [
        61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
    ]

    # Left eyebrow indices
    LEFT_EYEBROW_INDICES = [
        46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305, 292, 282, 295, 285, 336, 296, 334, 293, 300, 276, 283, 282, 295, 285
    ]

    # Left eye indices
    LEFT_EYE_INDICES = [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33
    ]

    # Left pupil indices (smaller region within left eye)
    LEFT_PUPIL_INDICES = [
        468, 469, 470, 471, 472
    ]

    # Right eyebrow indices
    RIGHT_EYEBROW_INDICES = [
        276, 283, 282, 295, 285, 336, 296, 334, 293, 300, 276, 283, 282, 295, 285
    ]

    # Right eye indices
    RIGHT_EYE_INDICES = [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362
    ]

    # Right pupil indices (smaller region within right eye)
    RIGHT_PUPIL_INDICES = [
        473, 474, 475, 476, 477
    ]

    @classmethod
    def INPUT_TYPES(cls):
        bool_true_widget = ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"})
        bool_false_widget = ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"})

        return {
            "required": {
                "image": ("IMAGE",),
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                "bbox_fill": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "crop_min_size": ("INT", {"min": 10, "max": MAX_RESOLUTION, "step": 1, "default": 50}),
                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 1}),
                "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "face": bool_true_widget,
                "mouth": bool_false_widget,
                "left_eyebrow": bool_false_widget,
                "left_eye": bool_false_widget,
                "left_pupil": bool_false_widget,
                "right_eyebrow": bool_false_widget,
                "right_eye": bool_false_widget,
                "right_pupil": bool_false_widget,
                "min_detection_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "refine_landmarks": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_masks"
    CATEGORY = "image/masking"
    OUTPUT_NODE = False

    def create_masks(self, image, crop_factor, bbox_fill, crop_min_size, drop_size, dilation,
                    face, mouth, left_eyebrow, left_eye, left_pupil, right_eyebrow, right_eye, right_pupil,
                    min_detection_confidence, refine_landmarks):

        batch_size, img_h, img_w, channels = image.shape
        output_masks = []

        # Process each image in the batch
        for i in range(batch_size):
            img_tensor = image[i]

            # Convert ComfyUI image tensor to NumPy array for MediaPipe
            img_np = img_tensor.cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            # Ensure image is HWC format
            if img_np.shape[0] == channels and img_np.shape[2] != channels:
                img_np = np.transpose(img_np, (1, 2, 0))

            # Ensure 3 channels (RGB) for MediaPipe
            if img_np.shape[-1] == 1:
                img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[-1] == 4:
                img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            else:
                img_np_rgb = img_np

            # Create a blank mask
            current_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            try:
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=refine_landmarks,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=0.5) as face_mesh:

                    results = face_mesh.process(img_np_rgb)

                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark

                        # Helper function to create mask for a region
                        def create_region_mask(indices):
                            if not indices:
                                return np.zeros((img_h, img_w), dtype=np.uint8)

                            polygon_points = []
                            for idx in indices:
                                if idx < len(landmarks):
                                    lm = landmarks[idx]
                                    cx, cy = int(lm.x * img_w), int(lm.y * img_h)
                                    polygon_points.append([cx, cy])

                            if len(polygon_points) >= 3:
                                region_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                                polygon_np = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.fillPoly(region_mask, [polygon_np], 255)
                                return region_mask
                            return np.zeros((img_h, img_w), dtype=np.uint8)

                        # Create masks for each enabled region
                        region_masks = []

                        if face:
                            region_masks.append(create_region_mask(self.FACE_INDICES))
                        if mouth:
                            region_masks.append(create_region_mask(self.MOUTH_INDICES))
                        if left_eyebrow:
                            region_masks.append(create_region_mask(self.LEFT_EYEBROW_INDICES))
                        if left_eye:
                            region_masks.append(create_region_mask(self.LEFT_EYE_INDICES))
                        if left_pupil:
                            region_masks.append(create_region_mask(self.LEFT_PUPIL_INDICES))
                        if right_eyebrow:
                            region_masks.append(create_region_mask(self.RIGHT_EYEBROW_INDICES))
                        if right_eye:
                            region_masks.append(create_region_mask(self.RIGHT_EYE_INDICES))
                        if right_pupil:
                            region_masks.append(create_region_mask(self.RIGHT_PUPIL_INDICES))

                        # Combine all enabled region masks
                        if region_masks:
                            for region_mask in region_masks:
                                current_mask = cv2.bitwise_or(current_mask, region_mask)

                        # Apply dilation if specified
                        if dilation != 0:
                            if dilation > 0:
                                # Positive dilation - expand mask
                                kernel = np.ones((abs(dilation), abs(dilation)), np.uint8)
                                current_mask = cv2.dilate(current_mask, kernel, iterations=1)
                            else:
                                # Negative dilation - shrink mask (erosion)
                                kernel = np.ones((abs(dilation), abs(dilation)), np.uint8)
                                current_mask = cv2.erode(current_mask, kernel, iterations=1)

            except Exception as e:
                print(f"Error during MediaPipe processing for image {i+1}/{batch_size}: {e}")

            # Convert mask to torch tensor
            mask_tensor = torch.from_numpy(current_mask.astype(np.float32) / 255.0)
            output_masks.append(mask_tensor)

        # Stack all masks in the batch
        final_mask_batch = torch.stack(output_masks, dim=0)
        return (final_mask_batch,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MediaPipeFaceToMask": MediaPipeFaceToMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeFaceToMask": "MediaPipe Face to Mask"
}