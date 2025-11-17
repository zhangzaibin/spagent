"""
Re-export existing training data in simple format

This script reads existing collected data and exports it in the new simple format.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from spagent.core import DataCollector

def re_export_simple_format(data_dir: str):
    """
    Re-export existing training data in simple format
    
    Args:
        data_dir: Path to existing training data directory
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        return
    
    if not (data_path / "sessions").exists():
        print(f"❌ Error: No sessions directory found in: {data_dir}")
        return
    
    print(f"{'='*60}")
    print(f"Re-exporting data from: {data_dir}")
    print(f"{'='*60}\n")
    
    # Create a DataCollector pointing to the existing directory
    collector = DataCollector(
        output_dir=str(data_path),
        save_images=False,  # Don't re-copy images
        auto_save=False     # We're just re-exporting
    )
    
    try:
        # Export in simple format
        print("Exporting in simple format...")
        
        collector.export_for_training(
            output_file=f"{data_dir}/train_simple.jsonl",
            format="simple"
        )
        print(f"✓ Exported: {data_dir}/train_simple.jsonl")
        
        collector.export_for_training(
            output_file=f"{data_dir}/train_sharegpt_simple.json",
            format="sharegpt",
            simple_format=True
        )
        print(f"✓ Exported: {data_dir}/train_sharegpt_simple.json")
        
        print(f"\n{'='*60}")
        print("✅ Re-export completed!")
        print(f"{'='*60}")
        
        # Show sample
        print("\nSample from train_simple.jsonl:")
        import json
        with open(f"{data_dir}/train_simple.jsonl") as f:
            first_line = f.readline()
            sample = json.loads(first_line)
            print(f"  ID: {sample['id']}")
            print(f"  Images: {len(sample['images'])} images")
            print(f"  Question (first 100 chars): {sample['question'][:100]}...")
            print(f"  Answer (first 100 chars): {sample['answer'][:100]}...")
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Re-export training data in simple format')
    parser.add_argument('data_dir', type=str, 
                       help='Path to training data directory (e.g., training_data/xxx)')
    
    args = parser.parse_args()
    
    re_export_simple_format(args.data_dir)


