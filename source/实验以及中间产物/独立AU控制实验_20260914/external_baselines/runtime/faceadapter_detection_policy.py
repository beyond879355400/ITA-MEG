"""Explicit input-detection policies; never changes RGB generation input."""

THRESHOLDS = (0.5, 0.3, 0.15, 0.1)
STAGES = ("source_original", "source_author_crop", "target_original", "target_author_crop")


def autocontrast_bgr(image):
    import numpy as np
    from PIL import Image, ImageOps
    rgb = Image.fromarray(np.asarray(image)[:, :, ::-1])
    contrast = ImageOps.autocontrast(rgb, cutoff=1)
    return np.asarray(contrast)[:, :, ::-1].copy()


class DetectionPolicy:
    def __init__(self, app, policy="author_default", transform=None):
        if policy not in ("author_default", "shared_fallback"):
            raise ValueError("Unknown detection policy: " + policy)
        self._app = app
        self.policy = policy
        self.transform = transform or autocontrast_bgr
        self.events = []
        self.case_id = None

    def __getattr__(self, name):
        return getattr(self._app, name)

    def begin_case(self, case_id):
        self.case_id = case_id
        self.events = []

    def get(self, image, *args, **kwargs):
        original_threshold = float(self._app.det_model.det_thresh)
        call = len(self.events)
        event = {"case_id": self.case_id, "call": call + 1, "stage": STAGES[call] if call < len(STAGES) else "additional", "policy": self.policy, "attempts": [], "detected": False, "image_shape": list(getattr(image, "shape", []))}
        try:
            variants = ("original",) if self.policy == "author_default" else ("original", "autocontrast_cutoff_1")
            thresholds = (original_threshold,) if self.policy == "author_default" else THRESHOLDS
            for preprocessing in variants:
                detection_input = image if preprocessing == "original" else self.transform(image)
                for threshold in thresholds:
                    self._app.det_model.det_thresh = threshold
                    attempt = {"preprocess": preprocessing, "threshold": threshold, "candidate_count": None}
                    event["attempts"].append(attempt)
                    faces = self._app.get(detection_input, *args, **kwargs)
                    attempt["candidate_count"] = len(faces)
                    if faces:
                        event.update(detected=True, selected_preprocess=preprocessing, selected_threshold=threshold, candidate_count=len(faces))
                        event["detections"] = [{"bbox": [float(v) for v in face["bbox"]], "score": float(face["det_score"]) if face.get("det_score") is not None else None} for face in faces]
                        return faces
            event.update(candidate_count=0, selected_preprocess=None, selected_threshold=None)
            return []
        except Exception as error:
            event.update(error_type=type(error).__name__, error=str(error))
            raise
        finally:
            self._app.det_model.det_thresh = original_threshold
            self.events.append(event)
