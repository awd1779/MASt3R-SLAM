# mast3r_slam/seem_utils.py
"""
Utilities for SAM model loading and inference.
"""
import torch
import numpy as np
import cv2 # OpenCV will be used for image format conversion if needed

# Attempt to import SAM specific modules
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("[Warning] segment_anything library not found. SAM functionalities will not be available.")
    print("[Warning] Please install it via: pip install git+https://github.com/facebookresearch/segment-anything.git")

_sam_model = None
_sam_mask_generator = None

# --- Configuration for SAM ---
# TODO: User should update these paths and model types as needed
SAM_CHECKPOINT_PATH = "checkpoints/sam_vit_b_01ec64.pth"  # UPDATE THIS PATH
SAM_MODEL_TYPE = "vit_b"
# Use "cuda" if GPU is available, otherwise "cpu"
SAM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------

def load_sam_model(checkpoint_path: str = SAM_CHECKPOINT_PATH,
                   model_type: str = SAM_MODEL_TYPE,
                   device: str = SAM_DEVICE):
    """
    Loads the SAM model and prepares the automatic mask generator.
    """
    global _sam_model, _sam_mask_generator, SAM_AVAILABLE

    if not SAM_AVAILABLE:
        raise ImportError("segment_anything library is required but not installed.")

    if _sam_mask_generator is not None:
        print("[INFO] SAM model and mask generator already loaded.")
        return _sam_model, _sam_mask_generator

    try:
        print(f"[INFO] Loading SAM model: type='{model_type}' from checkpoint='{checkpoint_path}' to device='{device}'")
        _sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        _sam_model.to(device=device)
        _sam_model.eval()

        # Initialize the automatic mask generator
        # You can tune the parameters of SamAutomaticMaskGenerator for different results
        # See: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
        _sam_mask_generator = SamAutomaticMaskGenerator(
            model=_sam_model,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=200, # Added: Filter out small regions (area in pixels)
            # box_nms_thresh=0.7,
            # crop_n_layers=0,
            # crop_nms_thresh=0.7,
            # output_mode="binary_mask",
        )
        print("[INFO] SAM model and SamAutomaticMaskGenerator loaded successfully (with min_mask_region_area=200).")
    except FileNotFoundError:
        print(f"[ERROR] SAM Checkpoint file not found at: {checkpoint_path}")
        print(f"[ERROR] Please download the SAM checkpoint and update SAM_CHECKPOINT_PATH in mast3r_slam/seem_utils.py")
        _sam_model = None
        _sam_mask_generator = None
        raise
    except Exception as e:
        print(f"[Error] Failed to load SAM model or initialize generator: {e}")
        _sam_model = None
        _sam_mask_generator = None
        raise e

    return _sam_model, _sam_mask_generator

def run_sam_inference(image_tensor_chw: torch.Tensor, vocabulary: list = None):
    """
    Runs SAM automatic mask generation on the input image.

    Args:
        image_tensor_chw (torch.Tensor): Input image tensor (C x H x W), RGB, values 0-1.
                                         This is assumed to be the `frame.rgb` tensor.
        vocabulary (list, optional): Not directly used by SamAutomaticMaskGenerator but kept for API consistency.

    Returns:
        tuple: (segmentation_mask, label_map)
            - segmentation_mask (torch.Tensor): HxW tensor with integer instance IDs for each segment.
                                                0 is typically reserved for unassigned/background.
            - label_map (dict): Dictionary mapping segment IDs to generic label strings
                                (e.g., {1: "object_1", 2: "object_2", ...}).
    """
    global _sam_mask_generator, SAM_DEVICE, SAM_AVAILABLE

    if not SAM_AVAILABLE:
        raise ImportError("segment_anything library is required but not installed.")

    if _sam_mask_generator is None:
        print("[INFO] SAM model not loaded. Attempting to load now...")
        # This will use the global SAM_CHECKPOINT_PATH, SAM_MODEL_TYPE, SAM_DEVICE
        load_sam_model()
        if _sam_mask_generator is None: # Check again if loading failed
            raise RuntimeError("SAM model could not be loaded. Please check SAM_CHECKPOINT_PATH and ensure the file exists.")

    _c, h, w = image_tensor_chw.shape

    # SAM's SamAutomaticMaskGenerator expects image in HWC uint8 format (0-255), BGR or RGB.
    # The documentation suggests it handles RGB internally.
    # Input image_tensor_chw is C x H x W, float (0-1) from frame.rgb (which is normalized)
    # Let's assume frame.rgb was derived from an RGB image.

    # Convert CHW (0-1 float) to HWC (0-255 uint8)
    image_hwc_uint8 = (image_tensor_chw.permute(1, 2, 0) * 255.0).byte().cpu().numpy()

    # SAM's automatic mask generator can be a bit slow.
    # print(f"[INFO] Running SAM Automatic Mask Generation on image of shape {image_hwc_uint8.shape}...")
    masks = _sam_mask_generator.generate(image_hwc_uint8)

    if not masks:
        # print("[Warning] SAM produced no masks for the current image.")
        final_segmentation_mask = torch.zeros((h, w), dtype=torch.int64, device=SAM_DEVICE)
        label_map = {0: "background_or_no_sam_masks"}
        return final_segmentation_mask, label_map

    # Sort masks by area (largest first). This helps ensure that if smaller masks are contained
    # within larger ones, the larger one (processed first) defines the base segment ID.
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    final_segmentation_mask = torch.zeros((h, w), dtype=torch.int64, device=SAM_DEVICE)
    label_map = {0: "background"} # Label 0 for background

    current_label_id = 1
    for i, mask_data in enumerate(masks):
        mask_h, mask_w = mask_data['segmentation'].shape
        # Basic check, though SAM's masks should match input image dimensions
        if mask_h != h or mask_w != w:
            print(f"[Warning] SAM mask {i} shape {mask_h}x{mask_w} differs from image {h}x{w}. Skipping this mask.")
            continue

        segment = torch.from_numpy(mask_data['segmentation']).to(device=SAM_DEVICE)

        # Assign current_label_id to pixels where segment is true AND final_mask is still 0 (background)
        # This gives priority to larger masks (due to sorting) in overlapping regions.
        valid_pixels_for_current_label = segment & (final_segmentation_mask == 0)
        final_segmentation_mask[valid_pixels_for_current_label] = current_label_id

        if valid_pixels_for_current_label.any(): # Only add label if it was actually used
            label_map[current_label_id] = f"object_{current_label_id}"
            current_label_id += 1
            if current_label_id > 255: # Max for uchar label_id in PLY
                # print("[Warning] Reached max label_id (255) for uchar. Subsequent segments will be ignored for distinct labeling in PLY.")
                break

    # print(f"[INFO] SAM produced {len(masks)} raw masks, resulting in {current_label_id-1} labeled segments.")
    return final_segmentation_mask, label_map

if __name__ == '__main__':
    # Example Usage (for testing this file directly)
    if not SAM_AVAILABLE:
        print("Cannot run SAM example: segment_anything library not installed.")
    else:
        print("Testing SAM utils...")
        print(f"Attempting to use device: {SAM_DEVICE}")
        print(f"Looking for SAM checkpoint at: {SAM_CHECKPOINT_PATH} with model type: {SAM_MODEL_TYPE}")

        # 1. Load the model (ensure SAM_CHECKPOINT_PATH is correct)
        try:
            load_sam_model() # Uses global path and type

            dummy_h, dummy_w = 240, 320 # Smaller image for faster testing
            print(f"Creating a dummy test image of size {dummy_h}x{dummy_w}")

            dummy_image_np_hwc_uint8 = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
            # Simple pattern: red square, green circle
            cv2.rectangle(dummy_image_np_hwc_uint8, (dummy_w//4, dummy_h//4), (dummy_w//2, dummy_h//2), (200,0,0), -1) # BGR Red
            cv2.circle(dummy_image_np_hwc_uint8, (3*dummy_w//4, 3*dummy_h//4), dummy_h//5, (0,200,0), -1) # BGR Green

            # Convert HWC uint8 (0-255) BGR (from OpenCV) to CHW float (0-1) RGB
            dummy_image_rgb_hwc_float = cv2.cvtColor(dummy_image_np_hwc_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            dummy_image_tensor_chw = torch.from_numpy(dummy_image_rgb_hwc_float).permute(2,0,1).to(SAM_DEVICE)

            print(f"Running SAM inference on dummy image...")
            seg_mask, labels = run_sam_inference(dummy_image_tensor_chw)

            print("\nTest Output from __main__:")
            print("Segmentation Mask shape:", seg_mask.shape)
            print("Segmentation Mask device:", seg_mask.device)
            unique_labels_found = torch.unique(seg_mask)
            print("Segmentation Mask (unique values):", unique_labels_found)
            print("Labels found in map:", labels)

            # Basic check: ensure all unique labels in mask are in the label_map
            for ul in unique_labels_found.tolist():
                if ul not in labels:
                    print(f"[ERROR] Unique label {ul} from mask is NOT in label_map!")


            # Visualize the mask (optional, requires opencv and matplotlib)
            try:
                import matplotlib.pyplot as plt
                print("Attempting to visualize SAM test output using Matplotlib...")
                plt.figure(figsize=(12,6))

                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(dummy_image_np_hwc_uint8, cv2.COLOR_BGR2RGB)) # Show original RGB
                plt.title("Original Test Image")
                plt.axis('off')

                # Create a colored version of the mask for visualization
                colored_seg_mask_vis = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
                # Use a more varied color generation for many segments
                np.random.seed(0) # for consistent colors
                viz_colors = np.random.randint(0, 255, size=(len(labels)+1, 3), dtype=np.uint8)

                for label_id_val in unique_labels_found.tolist():
                    if label_id_val == 0 and "background" in labels.get(0,"").lower() : # Special handling for background
                         colored_seg_mask_vis[seg_mask.cpu().numpy() == label_id_val] = [128, 128, 128] # Grey for background
                    elif label_id_val != 0 :
                         colored_seg_mask_vis[seg_mask.cpu().numpy() == label_id_val] = viz_colors[label_id_val % len(viz_colors)]


                plt.subplot(1, 2, 2)
                plt.imshow(colored_seg_mask_vis)
                plt.title(f"SAM Segmentation Overlay ({len(unique_labels_found)-1} segments found)")
                plt.axis('off')

                output_filename = "sam_test_output.png"
                plt.savefig(output_filename)
                print(f"Saved SAM test output visualization to {output_filename}")
            except ImportError:
                print("Matplotlib not installed, skipping visualization of SAM test output.")
            except Exception as e_vis:
                print(f"Error during SAM test visualization: {e_vis}")

        except FileNotFoundError:
            # This is already handled in load_sam_model, but good to catch here for the test script
            print(f"[CRITICAL ERROR in __main__] SAM checkpoint file not found. Please check SAM_CHECKPOINT_PATH in seem_utils.py.")
        except ImportError:
             print(f"[CRITICAL ERROR in __main__] 'segment_anything' library not found or other import error.")
        except Exception as e:
            print(f"Error in SAM utils test (__main__): {e}")
            import traceback
            traceback.print_exc()

        print("\nSAM utils __main__ test complete.")
