"""Uniform detector-only fallback; landmarks and generation keep original pixels."""
THRESHOLDS = (0.5, 0.3, 0.15, 0.1)


def autocontrast_bgr(image):
    import numpy as np
    from PIL import Image, ImageOps
    return np.asarray(ImageOps.autocontrast(Image.fromarray(image[:, :, ::-1]), cutoff=1))[:, :, ::-1].copy()


class DetectorPolicy:
    def __init__(self, detector, policy="shared_fallback", transform=None):
        if policy not in ("author_default", "shared_fallback"):
            raise ValueError("Unknown detection policy")
        self.detector = detector
        self.policy = policy
        self.transform = transform or autocontrast_bgr
        self.events = []
        self.case_id = None

    def __getattr__(self, key):
        return getattr(self.detector, key)

    def begin_case(self, case_id):
        self.case_id, self.events = case_id, []

    def detect(self, image, *args, **kwargs):
        original = float(self.detector.det_thresh)
        event = {"case_id": self.case_id, "call": len(self.events) + 1,
                 "policy": self.policy, "attempts": [], "image_shape": list(getattr(image, "shape", []))}
        try:
            variants = ("original",) if self.policy == "author_default" else ("original", "autocontrast_cutoff_1")
            thresholds = (original,) if self.policy == "author_default" else THRESHOLDS
            for variant in variants:
                detection_input = image if variant == "original" else self.transform(image)
                for threshold in thresholds:
                    self.detector.det_thresh = threshold
                    boxes, points = self.detector.detect(detection_input, *args, **kwargs)
                    event["attempts"].append({"preprocess": variant, "threshold": threshold, "candidate_count": len(boxes)})
                    if len(boxes):
                        event.update(detected=True, selected_preprocess=variant, selected_threshold=threshold)
                        rows = boxes.tolist() if hasattr(boxes, "tolist") else boxes
                        event["detections"] = [{"bbox": [float(x) for x in row[:4]], "score": float(row[4])} for row in rows]
                        return boxes, points
            event["detected"] = False
            return boxes, points
        finally:
            self.detector.det_thresh = original
            self.events.append(event)
