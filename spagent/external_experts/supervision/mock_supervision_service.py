import os
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MockSupervisionService:
    """Mock supervision service for object detection, segmentation, tracking, etc."""

    def __init__(self):
        self.is_healthy = True

    def health_check(self) -> Dict[str, Any]:
        logger.info("MockSupervisionService 健康检查")
        return {
            "status": "healthy",
            "service": "mock_supervision",
            "message": "Mock service is running"
        }

    def mock_infer(self, image_path: str, task_type: str = "det") -> Optional[Dict[str, Any]]:
        """
        模拟推理任务（检测/分割/跟踪）
        
        Args:
            image_path: 输入图像路径
            task_type: 任务类型，支持 "det", "seg", "track"
        
        Returns:
            推理结果的字典
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"图像文件不存在: {image_path}")
                return None

            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return None

            h, w, _ = image.shape
            logger.info(f"读取图像尺寸: {(w, h)}，任务类型: {task_type}")

            # 随机生成结果
            results = []
            for i in range(np.random.randint(1, 4)):
                x1, y1 = np.random.randint(0, w // 2), np.random.randint(0, h // 2)
                x2, y2 = np.random.randint(w // 2, w), np.random.randint(h // 2, h)
                box = [x1, y1, x2, y2]

                result = {"bbox": box, "score": np.round(np.random.uniform(0.6, 1.0), 2), "id": i}

                if task_type == "seg":
                    # 生成一个随机mask
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    result["segmentation"] = mask
                elif task_type == "track":
                    pass
                    # result["track"] = [(x1, y1), (x2, y2)]  # mock 轨迹
                    # result["keypoints"] = np.random.randint(0, min(h, w), (17, 3)).tolist()

                results.append(result)

            return {
                "success": True,
                "task_type": task_type,
                "num_results": len(results),
                "results": results,
                "image_size": (w, h)
            }

        except Exception as e:
            logger.error(f"mock_infer 执行失败: {e}")
            return None
