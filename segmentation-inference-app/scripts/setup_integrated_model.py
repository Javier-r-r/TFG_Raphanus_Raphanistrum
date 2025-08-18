"""
Setup script to prepare your model files for the integrated Tkinter app.

This script helps you copy your trained model files to the correct location
and creates the necessary configuration for the integrated app.

Usage:
    python setup_integrated_model.py
"""

import os
import json
import shutil
from pathlib import Path

def setup_integrated_model():
    """Interactive setup for integrated model"""
    print("🚀 Setting up integrated model for Tkinter app")
    print("=" * 50)
    
    script_dir = Path(__file__).parent
    print(f"Script directory: {script_dir}")
    
    # Get model weights path
    print("\n1. Model Weights (.pth file)")
    weights_path = input("Enter path to your best_model.pth file: ").strip().strip('"')
    
    if not weights_path or not os.path.exists(weights_path):
        print("❌ Weights file not found!")
        return
    
    # Copy weights to script directory
    target_weights = script_dir / "best_model.pth"
    shutil.copy2(weights_path, target_weights)
    print(f"✅ Copied weights to: {target_weights}")
    
    # Check for config.json in the same directory as weights
    weights_dir = Path(weights_path).parent
    config_source = weights_dir / "config.json"
    
    if config_source.exists():
        target_config = script_dir / "config.json"
        shutil.copy2(config_source, target_config)
        print(f"✅ Copied config.json to: {target_config}")
    else:
        print("⚠️  config.json not found next to weights")
        
        # Create basic config
        arch = input("Enter model architecture (default: Unet): ").strip() or "Unet"
        encoder = input("Enter encoder name (default: resnet34): ").strip() or "resnet34"
        
        config = {
            "arch": arch,
            "encoder_name": encoder,
            "created_by": "setup_integrated_model.py"
        }
        
        target_config = script_dir / "config.json"
        with open(target_config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Created config.json: {target_config}")
    
    # Check for dataset stats
    mean_source = weights_dir / "dataset_mean.npy"
    std_source = weights_dir / "dataset_std.npy"
    
    if mean_source.exists():
        target_mean = script_dir / "dataset_mean.npy"
        shutil.copy2(mean_source, target_mean)
        print(f"✅ Copied dataset_mean.npy to: {target_mean}")
    else:
        print("⚠️  dataset_mean.npy not found (will use ImageNet stats)")
    
    if std_source.exists():
        target_std = script_dir / "dataset_std.npy"
        shutil.copy2(std_source, target_std)
        print(f"✅ Copied dataset_std.npy to: {target_std}")
    else:
        print("⚠️  dataset_std.npy not found (will use ImageNet stats)")
    
    print("\n🎉 Setup complete!")
    print("\nFiles in scripts directory:")
    for file in ["best_model.pth", "config.json", "dataset_mean.npy", "dataset_std.npy"]:
        path = script_dir / file
        status = "✅" if path.exists() else "❌"
        print(f"  {status} {file}")
    
    print(f"\n🚀 Ready to run: python {script_dir}/tk_inference_app.py")

if __name__ == "__main__":
    setup_integrated_model()
