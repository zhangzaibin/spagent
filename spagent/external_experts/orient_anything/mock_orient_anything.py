from typing import Dict, Any


class MockOrientAnything:
    def predict(
        self,
        image_path: str,
        model_size: str = "large",
        use_tta: bool = False,
        remove_background: bool = True,
        device: str = "cpu",
        save_vis: bool = False,
        output_dir: str | None = None,
    ) -> Dict[str, Any]:
        return {
            "success": True,
            "azimuth": 35.0,
            "polar": 20.0,
            "rotation": 5.0,
            "confidence": 0.91,
            "model_size": model_size,
            "use_tta": use_tta,
            "remove_background": remove_background,
            "visualization_path": None,
        }