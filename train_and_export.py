#!/usr/bin/env python3
"""
YOLO Segmentation Training and ONNX Export Script
This script trains a YOLOv11 segmentation model and exports it to ONNX format.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)

def train_yolo_model(
    data_yaml,
    model_size="n",
    epochs=100,
    imgsz=640,
    batch=16,
    device="auto",
    project="runs/segment",
    name="train",
    patience=10,
    save_period=10,
    resume=False,
    pretrained=True
):
    """
    Train a YOLO segmentation model
    
    Args:
        data_yaml: Path to dataset YAML file
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device to use (auto, cpu, 0, 1, etc.)
        project: Project directory
        name: Experiment name
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
        resume: Resume from last checkpoint
        pretrained: Use pretrained weights
    
    Returns:
        Path to best model weights
    """
    
    # Initialize model
    if pretrained:
        model_name = f"yolo11{model_size}-seg.pt"
        print(f"Loading pretrained model: {model_name}")
        model = YOLO(model_name)
    else:
        model_name = f"yolo11{model_size}-seg.yaml"
        print(f"Loading model architecture: {model_name}")
        model = YOLO(model_name)
    
    # Train the model
    print(f"\nStarting training with the following parameters:")
    print(f"  Data: {data_yaml}")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    print(f"  Project: {project}")
    print(f"  Name: {name}")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save_period=save_period,
        resume=resume,
        verbose=True,
        plots=True,
        save_json=True,
        val=True,
        save=True,
        exist_ok=True
    )
    
    # Get the path to the best weights
    best_weights = results.save_dir / "weights" / "best.pt"
    print(f"\nTraining completed! Best weights saved to: {best_weights}")
    
    return best_weights

def export_to_onnx(model_path, imgsz=640, simplify=True, dynamic=False, half=False):
    """
    Export trained model to ONNX format
    
    Args:
        model_path: Path to trained model weights
        imgsz: Image size for export
        simplify: Simplify ONNX model
        dynamic: Use dynamic axes
        half: Use FP16 precision
    
    Returns:
        Path to exported ONNX model
    """
    
    print(f"\nExporting model to ONNX...")
    print(f"  Model: {model_path}")
    print(f"  Image size: {imgsz}")
    print(f"  Simplify: {simplify}")
    print(f"  Dynamic: {dynamic}")
    print(f"  Half precision: {half}")
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Export to ONNX
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=simplify,
        dynamic=dynamic,
        half=half,
        opset=11
    )
    
    print(f"ONNX export completed! Model saved to: {onnx_path}")
    return onnx_path

def validate_dataset(data_yaml):
    """Validate that the dataset exists and is properly configured"""
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    
    # Try to load the dataset to check if it's valid
    try:
        model = YOLO("yolo11n-seg.pt")  # Load a temporary model
        model.val(data=data_yaml, verbose=False)  # Quick validation
        print(f"Dataset validation passed: {data_yaml}")
    except Exception as e:
        print(f"WARNING: Dataset validation failed: {e}")
        print("Continuing anyway - check your dataset paths and format")

def main():
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model and export to ONNX")
    
    # Dataset arguments
    parser.add_argument("--data", 
                       default="datasets/segmentation/dataset.yaml",
                       help="Path to dataset YAML file")
    
    # Model arguments
    parser.add_argument("--model-size", 
                       choices=["n", "s", "m", "l", "x"], 
                       default="n",
                       help="Model size (n=nano, s=small, m=medium, l=large, x=extra-large)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size for training")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--device", default="0",
                       help="Device to use (auto, cpu, 0, 1, etc.)")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--save-period", type=int, default=10,
                       help="Save checkpoint every N epochs")
    
    # Project arguments
    parser.add_argument("--project", default="runs/segment",
                       help="Project directory")
    parser.add_argument("--name", default="train",
                       help="Experiment name")
    
    # Export arguments
    parser.add_argument("--export-imgsz", type=int, default=640,
                       help="Image size for ONNX export")
    parser.add_argument("--no-simplify", action="store_true",
                       help="Don't simplify ONNX model")
    parser.add_argument("--dynamic", action="store_true",
                       help="Use dynamic axes in ONNX export")
    parser.add_argument("--half", action="store_true",
                       help="Use FP16 precision in ONNX export")
    
    # Control arguments
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from last checkpoint")
    parser.add_argument("--no-pretrained", action="store_true",
                       help="Don't use pretrained weights")
    parser.add_argument("--train-only", action="store_true",
                       help="Only train, don't export to ONNX")
    parser.add_argument("--export-only", 
                       help="Only export existing model to ONNX (provide path to .pt file)")
    parser.add_argument("--validate-dataset", action="store_true",
                       help="Validate dataset before training")
    
    args = parser.parse_args()
    
    try:
        # Validate dataset if requested
        if args.validate_dataset:
            validate_dataset(args.data)
        
        # Export only mode
        if args.export_only:
            if not os.path.exists(args.export_only):
                raise FileNotFoundError(f"Model file not found: {args.export_only}")
            
            onnx_path = export_to_onnx(
                model_path=args.export_only,
                imgsz=args.export_imgsz,
                simplify=not args.no_simplify,
                dynamic=args.dynamic,
                half=args.half
            )
            print(f"\nExport completed successfully!")
            print(f"ONNX model: {onnx_path}")
            return
        
        # Train the model
        best_weights = train_yolo_model(
            data_yaml=args.data,
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            save_period=args.save_period,
            resume=args.resume,
            pretrained=not args.no_pretrained
        )
        
        # Export to ONNX unless train-only mode
        if not args.train_only:
            onnx_path = export_to_onnx(
                model_path=best_weights,
                imgsz=args.export_imgsz,
                simplify=not args.no_simplify,
                dynamic=args.dynamic,
                half=args.half
            )
            
            print(f"\n{'='*60}")
            print(f"TRAINING AND EXPORT COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Best model weights: {best_weights}")
            print(f"ONNX model: {onnx_path}")
            print(f"\nYou can now use the ONNX model with inference2.py:")
            print(f"python inference2.py --model {onnx_path} --image your_image.jpg --top-k 5")
        else:
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Best model weights: {best_weights}")
            print(f"\nTo export to ONNX later, run:")
            print(f"python {__file__} --export-only {best_weights}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
