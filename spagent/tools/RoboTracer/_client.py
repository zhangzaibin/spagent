from typing import Dict, Any, List
import requests


class RoboTracerClient:
    """
    HTTP client for a real RoboTracer backend service.
    Expected endpoint:
        POST {server_url}/infer
    """

    def __init__(self, server_url: str = "http://localhost:20040", timeout: int = 120):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def infer(
        self,
        image_paths: List[str],
        coordinate_mode: str = "relative_2d",
        return_summary_only: bool = False
    ) -> Dict[str, Any]:
        payload = {
            "image_paths": image_paths,
            "coordinate_mode": coordinate_mode,
            "return_summary_only": return_summary_only
        }

        try:
            response = requests.post(
                f"{self.server_url}/infer",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            if "success" not in data:
                data["success"] = True
            return data
        except requests.RequestException as e:
            return {
                "success": False,
                "error": f"HTTP request failed: {str(e)}"
            }
        except ValueError as e:
            return {
                "success": False,
                "error": f"Invalid JSON response: {str(e)}"
            }

    def trace(
        self,
        image_paths: List[str],
        coordinate_mode: str = "relative_2d",
        return_summary_only: bool = False
    ) -> Dict[str, Any]:
        return self.infer(
            image_paths=image_paths,
            coordinate_mode=coordinate_mode,
            return_summary_only=return_summary_only
        )
