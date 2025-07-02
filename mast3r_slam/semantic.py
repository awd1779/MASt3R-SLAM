import numpy as np
import torch

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except Exception:
    SamAutomaticMaskGenerator = None
    sam_model_registry = None


class SemanticPerception:
    """Simple semantic perception module using SAM."""

    def __init__(self, sam_checkpoint: str = "", model_type: str = "vit_h", device: str = "cuda"):
        self.device = device
        self.model = None
        if SamAutomaticMaskGenerator is not None and sam_model_registry is not None and sam_checkpoint:
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device)
            self.model = SamAutomaticMaskGenerator(sam)

    def process_frame(self, image: np.ndarray):
        """Process single RGB image and return instance mask and labels.

        Parameters
        ----------
        image: np.ndarray
            HxWx3 RGB image in range [0,1] or [0,255].

        Returns
        -------
        instance_mask: np.ndarray
            2D array with instance ids.
        instance_labels: dict
            Mapping from id to textual label. Currently uses a placeholder label
            since open-vocabulary classification is not implemented here.
        """
        img_uint8 = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
        h, w = img_uint8.shape[:2]
        instance_mask = np.zeros((h, w), dtype=np.int32)
        instance_labels = {}

        if self.model is None:
            # fallback: no model available
            return instance_mask, instance_labels

        masks = self.model.generate(img_uint8)
        for idx, m in enumerate(masks):
            instance_mask[m["segmentation"]] = idx + 1
            instance_labels[str(idx + 1)] = m.get("label", "object")

        return instance_mask, instance_labels
