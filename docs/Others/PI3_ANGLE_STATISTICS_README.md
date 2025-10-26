# Pi3 Tool Angle Distribution Statistics

## Overview

This feature adds comprehensive statistics tracking for Pi3 tool angle combinations used during evaluation. It helps analyze how the model explores different viewpoints when performing 3D reconstruction tasks.

## What's New

### 1. Angle Extraction
- Automatically extracts `azimuth_angle` and `elevation_angle` parameters from all Pi3 tool calls
- Records angle combinations as tuples (azimuth, elevation)

### 2. Distribution Statistics
The evaluation now tracks:
- **Total Pi3 calls**: Total number of times Pi3 tool was invoked
- **Unique angle combinations**: Number of distinct angle pairs used
- **Full distribution**: Complete breakdown of all angle combinations and their frequencies
- **Top 5 combinations**: Most frequently used angle combinations with percentages

## Usage

### Running Evaluation with Angle Statistics

```bash
python examples/evaluation/evaluate_img.py \
    --data_path dataset/VSI_route_plan.jsonl \
    --image_base_path dataset \
    --model gpt-4o \
    --max_iterations 3 \
    --max_samples 10
```

### Example Output

```
======================================================================
Pi3 Tool Angle Distribution Statistics
======================================================================
Total Pi3 calls:              45
Unique angle combinations:    12

Top 5 Most Used Angle Combinations:
Angle (azimuth, elevation)          Count      Percentage     
----------------------------------------------------------------------
(45, 0)                             12         26.7%          
(90, 0)                             10         22.2%          
(-45, 0)                            8          17.8%          
(0, 45)                             6          13.3%          
(90, 45)                            5          11.1%          

Full Distribution:
Angle (azimuth, elevation)          Count      
----------------------------------------------------------------------
(45, 0)                             12         
(90, 0)                             10         
(-45, 0)                            8          
(0, 45)                             6          
(90, 45)                            5          
(-90, 0)                            2          
(0, -30)                            1          
(180, 0)                            1          
======================================================================
```

## Angle Parameter Reference

### Azimuth Angle (方位角)
- **Range**: -180° to 180°
- **Controls**: Left-right rotation
- **Examples**:
  - `-90°`: Left 90 degrees
  - `0°`: Front view (matches input camera)
  - `90°`: Right 90 degrees
  - `180°`: Back view

### Elevation Angle (仰角)
- **Range**: -90° to 90°
- **Controls**: Up-down rotation
- **Examples**:
  - `-45°`: Look down 45 degrees
  - `0°`: Horizontal (matches input camera)
  - `45°`: Look up 45 degrees

### Common Angle Combinations

| Angle Combination | Description | Use Case |
|-------------------|-------------|----------|
| `(0, 0)` | Front view | Matches first input image (cam1) |
| `(45, 0)` | Right side view | Horizontal right rotation |
| `(-45, 0)` | Left side view | Horizontal left rotation |
| `(90, 0)` | Right profile | Full right side |
| `(-90, 0)` | Left profile | Full left side |
| `(180, 0)` | Back view | Behind the scene |
| `(0, 45)` | Overhead front | Looking down from front |
| `(0, -45)` | Upward front | Looking up from front |
| `(45, 30)` | Right-up diagonal | Combined right and up |

## Implementation Details

### Modified Files

1. **`spagent_evaluation.py`**
   - Added `extract_pi3_angles()` function to extract angle parameters from tool calls
   - Modified `evaluate_single_sample()` to include angle tracking
   - Modified `evaluate_single_video()` to include angle tracking
   - Enhanced `evaluate_tool_config()` to collect and aggregate angle statistics
   - Added angle distribution calculation using Counter

2. **`evaluate_img.py`**
   - Added `print_pi3_angle_statistics()` function for formatted output
   - Integrated statistics printing in main evaluation loop
   - Added Counter import for distribution analysis

### Data Flow

```
Agent.solve_problem()
    ↓
Returns: { "tool_calls": [...], ... }
    ↓
extract_pi3_angles(agent_result)
    ↓
Returns: [(azimuth, elevation), ...]
    ↓
Collect all angles across samples
    ↓
Counter() for distribution analysis
    ↓
Format and print statistics
```

## Benefits

1. **Model Behavior Analysis**: Understand which viewpoints the model prefers to explore
2. **Prompt Optimization**: Identify if model is exploring angles effectively
3. **Performance Tuning**: Detect if certain angles are underutilized
4. **Debugging**: Verify that angle parameters are being passed correctly
5. **Research Insights**: Analyze correlation between angle exploration and task accuracy

## JSON Output

Statistics are automatically saved in the evaluation results JSON file:

```json
{
  "config_name": "depth_detection_segmentation",
  "overall_accuracy": 0.85,
  "pi3_angle_distribution": {
    "total_pi3_calls": 45,
    "unique_angle_combinations": 12,
    "distribution": {
      "(45, 0)": 12,
      "(90, 0)": 10,
      "(-45, 0)": 8,
      ...
    },
    "top_5_combinations": [
      {
        "angle": "(45, 0)",
        "count": 12,
        "percentage": "26.7%"
      },
      ...
    ]
  }
}
```

## Notes

- Angles are rounded to nearest integer for cleaner statistics
- Only successful evaluations contribute to angle statistics
- Failed tool calls or evaluations are excluded from distribution
- Statistics are computed per configuration when using multiple tool configs

## Future Enhancements

Potential improvements:
- [ ] Heatmap visualization of angle distribution
- [ ] Correlation analysis between angle patterns and accuracy
- [ ] Per-task angle distribution breakdown
- [ ] Angle exploration coverage metrics
- [ ] Sequential pattern analysis (which angles follow others)

