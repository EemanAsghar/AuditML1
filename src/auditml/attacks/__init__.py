from .mia_threshold import ThresholdMIA
from .mia_shadow import ShadowMIA
from .model_inversion import ModelInversion
from .attribute_inference import AttributeInference


def get_attack(name: str, target_model, config=None, device=None):
    key = name.lower()
    if key in {"mia-threshold", "mia_threshold"}:
        return ThresholdMIA(target_model, config, device)
    if key in {"mia-shadow", "mia_shadow"}:
        return ShadowMIA(target_model, config, device)
    if key in {"inversion", "model-inversion", "model_inversion"}:
        return ModelInversion(target_model, config, device)
    if key in {"attribute", "attribute-inference", "attribute_inference"}:
        return AttributeInference(target_model, config, device)
    raise ValueError(f"Unknown attack: {name}")
