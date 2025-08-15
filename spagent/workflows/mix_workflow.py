import json
import logging
import os
from typing import Dict, Any, List, Optional

# 导入VLLM推理函数和prompt模板
try:
    from spagent.vllm_models.gpt import gpt_single_image_inference, gpt_multiple_images_inference
    from spagent.workflows.mix_workflow_prompts import get_complete_prompt, get_follow_up_prompt
    from spagent.workflows.depth_qa_workflow import DepthQAWorkflow
    from spagent.workflows.sam2_qa_workflow import SAM2QAWorkflow
    from spagent.workflows.gdino_qa_workflow import GdinoQAWorkflow
except ImportError:
    # 如果绝对导入失败，尝试相对导入
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    # 添加项目根目录到Python路径
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    from vllm_models.gpt import gpt_single_image_inference, gpt_multiple_images_inference
    from utils.mix_workflow_prompts import get_complete_prompt, get_follow_up_prompt
    from workflows.depth_qa_workflow import DepthQAWorkflow
    from workflows.sam2_qa_workflow import SAM2QAWorkflow
    from workflows.gdino_qa_workflow import GdinoQAWorkflow

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MixedExpertWorkflow:
    """混合专家工作流 - 结合深度估计、SAM2分割和GroundingDINO检测"""
    
    def __init__(self, ip: str = "10.8.131.51", use_mock: bool = True):
        """
        初始化混合专家工作流
        
        Args:
            ip: 专家服务部署的IP地址
            use_mock: 是否使用mock服务
        """
        # 初始化各个专家工作流
        self.depth_workflow = DepthQAWorkflow(use_mock_depth=use_mock)
        self.sam2_workflow = SAM2QAWorkflow(use_mock_sam=use_mock)
        self.gdino_workflow = GdinoQAWorkflow(use_mock=use_mock)
        
    def needs_depth_tool(self, response: str) -> bool:
        """检查是否需要深度估计工具"""
        # 检查是否包含tool_call标签
        if "<tool_call>" in response.lower():
            if "depth_estimation_tool" in response.lower():
                return True
        return False
    
    def needs_segmentation_tool(self, response: str) -> bool:
        """检查是否需要分割工具"""
        # 检查是否包含tool_call标签
        if "<tool_call>" in response.lower():
            if "segment_image_tool" in response.lower():
                return True
        return False
    
    def needs_detection_tool(self, response: str) -> bool:
        """检查是否需要检测工具"""
        # 检查是否包含tool_call标签
        if "<tool_call>" in response.lower():
            if "detect_objects_tool" in response.lower():
                return True
        return False
    
    def extract_detection_targets(self, response: str) -> List[str]:
        """从响应中提取检测目标"""
        try:
            # 尝试从tool_call中提取text_prompt参数
            start_idx = response.find("<tool_call>")
            end_idx = response.find("</tool_call>")
            if start_idx != -1 and end_idx != -1:
                tool_call = response[start_idx:end_idx + len("</tool_call>")]
                tool_data = json.loads(tool_call.replace("<tool_call>", "").replace("</tool_call>", ""))
                if tool_data["name"] == "detect_objects_tool":
                    text_prompt = tool_data["arguments"].get("text_prompt", "")
                    if text_prompt:
                        return [text_prompt]
        except Exception as e:
            logger.warning(f"从tool_call提取检测目标失败: {e}")
        
        # 如果无法从tool_call提取，返回默认值
        return ["object"]
    
    def run_workflow(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        运行混合专家工作流
        
        Args:
            image_path: 输入图像路径
            question: 用户问题
            
        Returns:
            完整的工作流结果
        """
        logger.info("开始执行混合专家工作流")
        
        # 1. 获取VLLM初始响应
        complete_prompt = get_complete_prompt(question)
        initial_response = gpt_single_image_inference(
            image_path=image_path,
            prompt=complete_prompt,
            model="gpt-4o",
            temperature=0.8
        )
        
        logger.info(f"初始回答: {initial_response}")
        
        # 2. 分析需要调用哪些专家工具
        expert_results = {}
        additional_images = []
        used_tools = []
        
        # 深度估计
        if self.needs_depth_tool(initial_response):
            logger.info("调用深度估计专家...")
            depth_result = self.depth_workflow.call_depth_estimation(image_path)
            if depth_result and depth_result.get('success'):
                expert_results['depth'] = depth_result
                additional_images.append(depth_result['output_path'])
                used_tools.append('depth_estimation_tool')
                logger.info("深度估计完成")
            else:
                logger.warning("深度估计失败")
        
        # SAM2分割
        if self.needs_segmentation_tool(initial_response):
            logger.info("调用SAM2分割专家...")
            sam_result = self.sam2_workflow.call_sam_segmentation(image_path)
            if sam_result and sam_result.get('success'):
                expert_results['sam2'] = sam_result
                additional_images.append(sam_result['vis_path'])
                used_tools.append('segment_image_tool')
                logger.info("SAM2分割完成")
            else:
                logger.warning("SAM2分割失败")
        
        # GroundingDINO检测
        if self.needs_detection_tool(initial_response):
            logger.info("调用GroundingDINO检测专家...")
            detection_targets = self.extract_detection_targets(initial_response)
            gdino_results = []
            for target in detection_targets:
                gdino_result = self.gdino_workflow._call_gdino_detection(image_path, text_prompt=target)
                if gdino_result and gdino_result != image_path:  # gdino_workflow returns original path on failure
                    gdino_results.append(gdino_result)
                    additional_images.append(gdino_result)
                    if 'detect_objects_tool' not in used_tools:
                        used_tools.append('detect_objects_tool')
                    logger.info(f"GroundingDINO检测完成: {target}")
                else:
                    logger.warning(f"GroundingDINO检测失败: {target}")
            if gdino_results:
                expert_results['gdino'] = gdino_results
        
        # 3. 如果有专家工具结果，重新询问VLLM
        if expert_results:
            follow_up_prompt = get_follow_up_prompt(question, initial_response, used_tools)
            
            # 构建包含所有图像的列表
            all_images = [image_path] + additional_images
            
            logger.info("基于专家结果重新询问VLLM...")
            final_response = gpt_multiple_images_inference(
                image_paths=all_images,
                prompt=follow_up_prompt,
                model="gpt-4o-mini",
                temperature=0.7
            )
            
            logger.info(f"最终回答: {final_response}")
        else:
            logger.info("未调用任何专家工具，使用初始回答")
            final_response = initial_response
        
        # 4. 返回完整结果
        return {
            "answer": final_response,
            "initial_response": initial_response,
            "expert_results": expert_results,
            "used_tools": used_tools,
            "additional_images": additional_images
        }


def infer(image_path: str, question: str, ip: str = "10.8.131.51", use_mock: bool = True) -> Dict[str, Any]:
    """
    简化接口，用于混合专家推理
    
    Args:
        image_path: 输入图像路径
        question: 用户问题
        ip: 专家服务IP地址
        use_mock: 是否使用mock服务
        
    Returns:
        推理结果
    """
    workflow = MixedExpertWorkflow(ip, use_mock)
    return workflow.run_workflow(image_path, question)


# 示例用法
if __name__ == "__main__":
    # 测试混合专家工作流
    image_path = "path/to/your/image.jpg"
    question = "分析这张图片中物体的深度关系和分割区域"
    
    result = infer(image_path, question)
    print(f"答案: {result['answer']}")
    print(f"使用的工具: {result['used_tools']}")
    print(f"生成的额外图像: {result['additional_images']}") 