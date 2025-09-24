#!/usr/bin/env python3
"""
彩色PLY文件可视化脚本
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from spagent.external_experts.Pi3.pi3.utils.basic import load_ply

def visualize_ply_colorful(ply_path, max_points=10000, output_file=None):
    """
    创建彩色PLY文件的交互式可视化
    """
    try:
        print(f"正在加载PLY文件: {ply_path}")
        xyz, rgb = load_ply(ply_path)
        
        if xyz is None:
            print("❌ 无法加载PLY文件")
            return
        
        print(f"✓ 成功加载 {len(xyz):,} 个点")
        
        # 应用官方的场景旋转 (Y轴100°, X轴155°)
        print("应用官方场景旋转: Y轴100°, X轴155°")
        r_y = R.from_euler('y', 100, degrees=True)
        r_x = R.from_euler('x', 155, degrees=True)
        official_rotation = r_x * r_y
        xyz = official_rotation.apply(xyz)
        print(f"旋转后点云范围: X[{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}], Y[{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}], Z[{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")
        
        # 子采样
        if len(xyz) > max_points:
            indices = np.random.choice(len(xyz), max_points, replace=False)
            xyz = xyz[indices]
            if rgb is not None:
                rgb = rgb[indices]
            print(f"  子采样到 {len(xyz):,} 个点")
        
        # 检查颜色数据
        if rgb is not None:
            print(f"✓ 颜色数据可用，形状: {rgb.shape}")
            print(f"  颜色范围: [{rgb.min():.3f}, {rgb.max():.3f}]")
            
            # 转换颜色到0-255范围
            rgb_255 = np.clip(rgb * 255, 0, 255).astype(int)
            unique_colors = len(np.unique(rgb_255.view(np.void), axis=0))
            print(f"  唯一颜色: {unique_colors}")
            
            # 生成RGB字符串
            colors = [f'rgb({r},{g},{b})' for r, g, b in rgb_255]
        else:
            print("❌ 无颜色数据，使用默认蓝色")
            colors = ['blue'] * len(xyz)
        
        # 设置输出文件名
        if output_file is None:
            output_file = ply_path.replace('.ply', '_colorful.html')
        
        # 计算点云中心和范围，用于设置相机位置
        center = xyz.mean(axis=0)
        ranges = xyz.max(axis=0) - xyz.min(axis=0)
        max_range = ranges.max()
        
        # 根据点云实际位置设置相机
        camera_distance = max_range * 1.5
        camera_x = center[0] + camera_distance * 0.3
        camera_y = center[1] - camera_distance * 0.8  
        camera_z = center[2] + camera_distance * 0.5
        
        print(f"点云中心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        print(f"相机位置: [{camera_x:.3f}, {camera_y:.3f}, {camera_z:.3f}]")
        
        # 生成HTML可视化
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>彩色点云可视化 - {ply_path}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .info {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .controls {{ margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>彩色点云可视化</h1>
    <div class="info">
        <strong>文件:</strong> {ply_path}<br>
        <strong>点数:</strong> {len(xyz):,}<br>
        <strong>颜色:</strong> {'彩色' if rgb is not None else '单色'}<br>
        <strong>唯一颜色:</strong> {unique_colors if rgb is not None else 'N/A'}
    </div>
    
    <div class="controls">
        <button onclick="resetView()">重置视角</button>
        <button onclick="toggleBackground()">切换背景</button>
    </div>
    
    <div id="plot" style="width:100%;height:600px;"></div>
    
    <script>
        var trace = {{
            x: {xyz[:, 0].tolist()},
            y: {xyz[:, 1].tolist()},
            z: {xyz[:, 2].tolist()},
            mode: 'markers',
            marker: {{
                size: 1,
                color: {colors},
                opacity: 0.8
            }},
            type: 'scatter3d',
            name: '点云'
        }};
        
        var layout = {{
            title: '彩色点云 - {len(xyz):,} 点 (已应用官方旋转)',
            scene: {{
                aspectmode: 'cube',
                camera: {{
                    eye: {{x: {camera_x:.3f}, y: {camera_y:.3f}, z: {camera_z:.3f}}},
                    center: {{x: {center[0]:.3f}, y: {center[1]:.3f}, z: {center[2]:.3f}}},
                    up: {{x: 0, y: 0, z: 1}}
                }},
                xaxis: {{title: 'X'}},
                yaxis: {{title: 'Y'}},
                zaxis: {{title: 'Z'}}
            }},
            margin: {{l: 0, r: 0, b: 0, t: 50}}
        }};
        
        Plotly.newPlot('plot', [trace], layout, {{responsive: true}});
        
        // 工具函数
        function resetView() {{
            Plotly.relayout('plot', {{
                'scene.camera': {{
                    eye: {{x: {camera_x:.3f}, y: {camera_y:.3f}, z: {camera_z:.3f}}},
                    center: {{x: {center[0]:.3f}, y: {center[1]:.3f}, z: {center[2]:.3f}}},
                    up: {{x: 0, y: 0, z: 1}}
                }}
            }});
        }}
        
        var bgWhite = true;
        function toggleBackground() {{
            var color = bgWhite ? '#000000' : '#ffffff';
            var gridcolor = bgWhite ? '#444444' : '#cccccc';
            bgWhite = !bgWhite;
            
            Plotly.relayout('plot', {{
                'scene.bgcolor': color,
                'scene.xaxis.gridcolor': gridcolor,
                'scene.yaxis.gridcolor': gridcolor,
                'scene.zaxis.gridcolor': gridcolor
            }});
        }}
        
        console.log('✓ 点云加载完成');
        console.log('📊 统计信息:', {{
            总点数: {len(xyz)},
            颜色数: {unique_colors if rgb is not None else 0},
            X范围: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}],
            Y范围: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}],
            Z范围: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]
        }});
    </script>
</body>
</html>
"""
        
        # 保存HTML文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ 彩色可视化已生成: {output_file}")
        print(f"✓ 请在浏览器中打开该文件查看彩色点云")
        
        # 显示前几个颜色样本
        if rgb is not None:
            print(f"\n前5个颜色样本:")
            for i in range(min(5, len(colors))):
                print(f"  点{i}: {colors[i]}")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="彩色PLY文件可视化")
    parser.add_argument("ply_file", type=str, help="PLY文件路径")
    parser.add_argument("--max_points", type=int, default=100000, help="最大显示点数")
    parser.add_argument("--output", type=str, help="输出HTML文件路径")
    
    args = parser.parse_args()
    visualize_ply_colorful(args.ply_file, args.max_points, args.output)
