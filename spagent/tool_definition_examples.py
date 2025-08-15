#!/usr/bin/env python3
"""
SPAgent 自定义工具定义示例
Custom Tool Definition Examples for SPAgent

这个文件展示如何为SPAgent系统定义自定义工具
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.tool import Tool


class CustomImageAnalysisTool(Tool):
    """
    自定义图像分析工具示例
    
    这个例子展示了如何创建一个自定义工具来集成你自己的图像分析算法
    """
    
    def __init__(self, analysis_type: str = "general"):
        """
        初始化自定义图像分析工具
        
        Args:
            analysis_type: 分析类型 ("general", "detailed", "fast")
        """
        super().__init__(
            name="custom_image_analysis_tool",
            description=f"Perform {analysis_type} image analysis using custom algorithms to extract features, patterns, and insights from images."
        )
        self.analysis_type = analysis_type
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """定义工具参数schema"""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image for analysis"
                },
                "analysis_mode": {
                    "type": "string",
                    "enum": ["colors", "textures", "shapes", "all"],
                    "description": "Type of analysis to perform",
                    "default": "all"
                },
                "confidence_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum confidence threshold for results",
                    "default": 0.5
                }
            },
            "required": ["image_path"]
        }
    
    def call(
        self, 
        image_path: str, 
        analysis_mode: str = "all",
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        执行自定义图像分析
        
        Args:
            image_path: 图像路径
            analysis_mode: 分析模式
            confidence_threshold: 置信度阈值
            
        Returns:
            分析结果字典
        """
        try:
            # 这里是你的自定义分析逻辑
            # 实际应用中，你会调用你的算法或模型
            
            print(f"正在执行 {self.analysis_type} 图像分析...")
            print(f"图像: {image_path}")
            print(f"模式: {analysis_mode}")
            print(f"置信度阈值: {confidence_threshold}")
            
            # 模拟分析结果
            analysis_results = {
                "colors": {
                    "dominant_colors": ["#FF5733", "#33FF57", "#3357FF"],
                    "color_distribution": {"red": 0.3, "green": 0.4, "blue": 0.3}
                },
                "textures": {
                    "texture_types": ["smooth", "rough", "pattern"],
                    "texture_scores": [0.8, 0.6, 0.7]
                },
                "shapes": {
                    "detected_shapes": ["circle", "rectangle", "triangle"],
                    "shape_count": {"circle": 3, "rectangle": 2, "triangle": 1}
                }
            }
            
            # 根据分析模式筛选结果
            if analysis_mode != "all":
                filtered_results = {analysis_mode: analysis_results.get(analysis_mode, {})}
            else:
                filtered_results = analysis_results
            
            return {
                "success": True,
                "analysis_type": self.analysis_type,
                "analysis_mode": analysis_mode,
                "confidence_threshold": confidence_threshold,
                "results": filtered_results,
                "summary": f"完成了 {analysis_mode} 模式的 {self.analysis_type} 分析",
                "output_path": f"outputs/custom_analysis_{Path(image_path).stem}.json"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"自定义图像分析失败: {str(e)}"
            }


class TextExtractionTool(Tool):
    """
    文本提取工具示例
    
    展示如何创建一个OCR文本提取工具
    """
    
    def __init__(self, language: str = "auto"):
        super().__init__(
            name="text_extraction_tool",
            description="Extract text from images using OCR (Optical Character Recognition) technology to identify and digitize text content."
        )
        self.language = language
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image containing text"
                },
                "language": {
                    "type": "string",
                    "description": "Language for OCR recognition (e.g., 'en', 'zh', 'auto')",
                    "default": self.language
                },
                "text_regions": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "Optional bounding boxes for specific text regions [[x1,y1,x2,y2], ...]"
                }
            },
            "required": ["image_path"]
        }
    
    def call(
        self, 
        image_path: str, 
        language: Optional[str] = None,
        text_regions: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """执行文本提取"""
        try:
            lang = language or self.language
            
            # 模拟OCR处理
            extracted_texts = [
                {
                    "text": "Sample extracted text 1",
                    "confidence": 0.95,
                    "bbox": [10, 20, 200, 40],
                    "language": lang
                },
                {
                    "text": "Another text block",
                    "confidence": 0.88,
                    "bbox": [10, 50, 180, 70],
                    "language": lang
                }
            ]
            
            # 如果指定了文本区域，只返回这些区域的结果
            if text_regions:
                # 这里应该实现区域匹配逻辑
                pass
            
            return {
                "success": True,
                "extracted_texts": extracted_texts,
                "total_text_blocks": len(extracted_texts),
                "language": lang,
                "full_text": " ".join([t["text"] for t in extracted_texts]),
                "average_confidence": sum(t["confidence"] for t in extracted_texts) / len(extracted_texts)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"文本提取失败: {str(e)}"
            }


class ImageComparisonTool(Tool):
    """
    图像比较工具示例
    
    展示如何创建一个比较多张图像的工具
    """
    
    def __init__(self):
        super().__init__(
            name="image_comparison_tool",
            description="Compare multiple images to identify similarities, differences, and relationships between them."
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of image paths to compare (minimum 2 images)"
                },
                "comparison_type": {
                    "type": "string",
                    "enum": ["similarity", "difference", "content", "structure"],
                    "description": "Type of comparison to perform",
                    "default": "similarity"
                },
                "generate_visualization": {
                    "type": "boolean",
                    "description": "Whether to generate comparison visualization",
                    "default": True
                }
            },
            "required": ["image_paths"]
        }
    
    def call(
        self, 
        image_paths: List[str],
        comparison_type: str = "similarity",
        generate_visualization: bool = True
    ) -> Dict[str, Any]:
        """执行图像比较"""
        try:
            if len(image_paths) < 2:
                return {
                    "success": False,
                    "error": "至少需要2张图像进行比较"
                }
            
            # 模拟比较结果
            comparison_results = {
                "pairwise_similarities": [
                    {"image1": image_paths[0], "image2": image_paths[1], "similarity": 0.85}
                ],
                "content_analysis": {
                    "common_objects": ["car", "building", "tree"],
                    "unique_objects": {
                        image_paths[0]: ["person", "bicycle"],
                        image_paths[1]: ["dog", "bench"]
                    }
                },
                "structural_comparison": {
                    "layout_similarity": 0.78,
                    "color_similarity": 0.82,
                    "texture_similarity": 0.71
                }
            }
            
            result = {
                "success": True,
                "comparison_type": comparison_type,
                "num_images": len(image_paths),
                "results": comparison_results,
                "summary": f"完成了 {len(image_paths)} 张图像的 {comparison_type} 比较"
            }
            
            if generate_visualization:
                result["visualization_path"] = "outputs/comparison_visualization.png"
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"图像比较失败: {str(e)}"
            }


def demo_custom_tools():
    """演示自定义工具的使用"""
    print("🛠️ 自定义工具定义和使用演示")
    print("="*80)
    
    try:
        from core import SPAgent
        from models import GPTModel
        
        # 创建自定义工具实例
        print("1. 创建自定义工具...")
        custom_analysis_tool = CustomImageAnalysisTool(analysis_type="detailed")
        text_extraction_tool = TextExtractionTool(language="auto")
        image_comparison_tool = ImageComparisonTool()
        
        print(f"   - {custom_analysis_tool.name}")
        print(f"   - {text_extraction_tool.name}")
        print(f"   - {image_comparison_tool.name}")
        
        # 创建智能体
        print("2. 创建包含自定义工具的智能体...")
        model = GPTModel(model_name="gpt-4o-mini")
        custom_tools = [
            custom_analysis_tool,
            text_extraction_tool,
            image_comparison_tool
        ]
        
        agent = SPAgent(model=model, tools=custom_tools)
        
        print(f"3. 智能体配置完成，包含 {len(custom_tools)} 个自定义工具:")
        for tool_name in agent.list_tools():
            print(f"   - {tool_name}")
        
        # 测试工具调用
        print("4. 测试自定义工具...")
        
        # 创建测试图像文件
        test_image = "test_image.jpg"
        with open(test_image, "w") as f:
            f.write("dummy image for testing")
        
        try:
            # 测试图像分析工具
            analysis_result = custom_analysis_tool.call(
                image_path=test_image,
                analysis_mode="colors",
                confidence_threshold=0.7
            )
            print(f"   分析工具结果: {analysis_result['success']}")
            
            # 测试文本提取工具
            text_result = text_extraction_tool.call(
                image_path=test_image,
                language="zh"
            )
            print(f"   文本提取结果: {text_result['success']}")
            
            # 测试图像比较工具
            comparison_result = image_comparison_tool.call(
                image_paths=[test_image, test_image],  # 用同一图像演示
                comparison_type="similarity"
            )
            print(f"   图像比较结果: {comparison_result['success']}")
            
        finally:
            # 清理测试文件
            import os
            if os.path.exists(test_image):
                os.remove(test_image)
        
        print("✅ 自定义工具演示完成！")
        
        return agent
        
    except Exception as e:
        print(f"❌ 自定义工具演示失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_tool_development_guide():
    """打印工具开发指南"""
    print("\n" + "="*80)
    print("📚 自定义工具开发指南")
    print("="*80)
    
    print("""
🎯 工具开发步骤:

1. 继承Tool基类:
   class MyTool(Tool):
       def __init__(self):
           super().__init__(name="my_tool", description="...")

2. 定义参数schema:
   @property
   def parameters(self) -> Dict[str, Any]:
       return {
           "type": "object",
           "properties": {...},
           "required": [...]
       }

3. 实现call方法:
   def call(self, **kwargs) -> Dict[str, Any]:
       # 你的工具逻辑
       return {
           "success": True/False,
           "result": ...,  # 工具特定结果
           "error": "..."  # 如果失败
       }

🔧 参数定义规范:
- 使用JSON Schema格式
- 支持type: "string", "number", "boolean", "array", "object"
- 可以设置enum, minimum, maximum等约束
- required数组指定必需参数

💡 最佳实践:
- 工具名使用下划线命名: "my_custom_tool"
- 描述要清晰说明工具功能和用途
- 返回结果包含success字段
- 失败时提供有意义的错误信息
- 支持可选参数和默认值
- 考虑性能和错误处理

🚀 集成到SPAgent:
1. 创建工具实例: tool = MyTool()
2. 添加到智能体: agent.add_tool(tool)
3. 或初始化时传入: SPAgent(model, tools=[tool])

📊 工具类型示例:
- 图像处理: 滤镜、增强、分析
- 文本处理: OCR、翻译、总结
- 数据分析: 统计、可视化、预测
- 外部API: 第三方服务集成
- 文件操作: 格式转换、合并、拆分
""")


if __name__ == "__main__":
    print("🎨 SPAgent自定义工具定义示例")
    print("本示例展示如何为SPAgent系统创建自定义工具")
    
    # 运行演示
    demo_custom_tools()
    
    # 打印开发指南
    print_tool_development_guide()
    
    print("\n🎯 总结:")
    print("- 自定义工具让SPAgent可以集成任何算法或服务")
    print("- 通过Tool基类可以轻松创建新工具")
    print("- 工具可以处理图像、文本、数据等各种任务")
    print("- SPAgent会自动管理工具的调用和结果集成")
    print("\n�� 查看更多示例和文档以了解详细用法") 