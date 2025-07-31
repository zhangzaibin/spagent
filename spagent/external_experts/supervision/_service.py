import os
import logging
import traceback
from typing import Optional, Dict, Any

from PIL import Image
import numpy as np
import cv2

from ultralytics import YOLO
from annotator import Annotator  # 假设你的Annotator类在annotator.py文件中

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnnotationService:
    """多功能目标标注服务（支持 mock 和真实模式）"""

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.model = None
        self.annotator = Annotator()

    def health_check(self) -> Dict[str, Any]:
        """服务健康检查"""
        try:
            if self.use_mock:
                logger.info("Mock 标注服务运行中")
                return {"status": "healthy", "mode": "mock"}
            else:
                logger.info("真实标注服务运行中")
                return {"status": "healthy", "mode": "real"}
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def test_infer(self) -> Dict[str, Any]:
        """执行测试推理"""
        try:
            if self.use_mock:
                test_img = np.zeros((256, 256, 3), dtype=np.uint8)
                cv2.rectangle(test_img, (60, 60), (180, 180), (0, 255, 0), 2)
                output_path = "mock_test_image.png"
                Image.fromarray(test_img).save(output_path)
                return {"success": True, "mode": "mock", "output_path": output_path}
            else:
                model = YOLO("yolov8n.pt")
                test_img = np.zeros((256, 256, 3), dtype=np.uint8)
                output_path = "real_test_output.jpg"
                self.annotator(test_img, task="image_det", model=model, target_path=output_path)
                return {"success": True, "mode": "real", "output_path": output_path}
        except Exception as e:
            logger.error(f"测试推理失败: {e}")
            logger.debug(traceback.format_exc())
            return {"success": False, "error": str(e)}

    def infer(self, input_path: str, task: str, model_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """执行真实推理或模拟推理"""
        try:
            if self.use_mock:
                mock_img = cv2.imread(input_path)
                if mock_img is None:
                    raise ValueError(f"读取图像失败：{input_path}")
                output_path = output_path or f"mock_output_{os.path.basename(input_path)}"
                cv2.putText(mock_img, f"Mock Task: {task}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.imwrite(output_path, mock_img)
                return {"success": True, "mode": "mock", "output_path": output_path}

            # 真实推理
            model = YOLO(model_path)
            result_path = self.annotator(input_path, task=task, model=model, target_path=output_path)
            return {
                "success": True,
                "mode": "real",
                "output_path": result_path
            }
        except Exception as e:
            logger.error(f"推理失败: {e}")
            logger.debug(traceback.format_exc())
            return {"success": False, "error": str(e)}


class AnnotationClient:
    """Annotation 客户端封装，用于统一访问标注服务"""

    def __init__(self, use_mock: bool = False):
        self.service = AnnotationService(use_mock=use_mock)

    def health_check(self) -> Dict[str, Any]:
        return self.service.health_check()

    def test_infer(self) -> Dict[str, Any]:
        return self.service.test_infer()

    def infer(self, input_path: str, task: str, model_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        return self.service.infer(input_path, task, model_path, output_path)


if __name__ == "__main__":
    client = AnnotationClient(use_mock=False)

    # 健康检查
    print("=== 健康检查 ===")
    print(client.health_check())

    # 测试推理
    print("=== 测试推理 ===")
    print(client.test_infer())

    # 实际推理
    print("=== 实际推理 ===")
    image_path = "assets/street.jpg"
    task = "image_seg"  # 支持 image_det, image_seg, video_id_only 等
    model_path = "yolov8n-seg.pt"
    output_path = "output/test_annotated.jpg"

    result = client.infer(image_path, task, model_path, output_path)
    print(result)
