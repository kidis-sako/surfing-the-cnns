"""
Preprocess surfing videos and cache them as .pt files for faster training.

This script:
1. Loads all videos from train/val/test directories
2. Extracts frames uniformly
3. Applies S3D preprocessing (resize, normalize)
4. Saves preprocessed tensors as .pt files

Run this once before training to speed up data loading by 10-20x.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms.functional as F

# S3D preprocessing configuration
NUM_FRAMES = 64
MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]

# Class names
class_names = [
    'cutback-frontside',
    'take-off',
    '360',
    'roller'
]


def uniform_sample_indices(num_frames, target_frames):
    """Uniformly sample frame indices from a video."""
    if num_frames < target_frames:
        indices = np.linspace(0, num_frames - 1, target_frames).round().astype(int)
    else:
        indices = np.linspace(0, num_frames - 1, target_frames).round().astype(int)
    return indices


def preprocess_video(video_path, num_frames=NUM_FRAMES):
    """
    Load and preprocess a video file.
    
    Returns:
        Preprocessed video tensor [C, T, H, W] ready for S3D model
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"Could not read video: {video_path}")
    
    # Uniform temporal sampling
    frame_indices = uniform_sample_indices(total_frames, num_frames)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # If frame reading fails, use the last successful frame
            if frames:
                frames.append(frames[-1])
            else:
                raise ValueError(f"Could not read frame {idx} from {video_path}")
    
    cap.release()
    
    # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
    video = np.stack(frames)
    video = torch.from_numpy(video).to(torch.uint8)
    video = video.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
    
    # Convert to float and scale to [0, 1]
    video = video.float() / 255.0
    
    # Resize and crop each frame
    T = video.shape[1]
    resized_frames = []
    for t in range(T):
        frame = video[:, t, :, :]  # [C, H, W]
        # Resize to 256x256
        frame = F.resize(frame, [256, 256], antialias=True)
        # Center crop to 224x224
        frame = F.center_crop(frame, [224, 224])
        resized_frames.append(frame)
    
    # Stack back to [C, T, H, W]
    video = torch.stack(resized_frames, dim=1)
    
    # Normalize with S3D mean and std
    mean = torch.tensor(MEAN).view(3, 1, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1, 1)
    video = (video - mean) / std
    
    return video


def preprocess_dataset(data_dir, cache_dir, split_name):
    """
    Preprocess all videos in a dataset directory and save to cache.
    
    Args:
        data_dir: Path to dataset directory (e.g., ./surfing_dataset/train)
        cache_dir: Path to cache directory (e.g., ./surfing_dataset_cache/train)
        split_name: Name of the split (train/val/test) for progress display
    """
    data_path = Path(data_dir)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Supported video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    
    # Count total videos
    total_videos = 0
    for class_name in class_names:
        class_dir = data_path / class_name
        if class_dir.exists():
            total_videos += sum(1 for f in class_dir.iterdir() if f.suffix in video_extensions)
    
    print(f"\n{'='*60}")
    print(f"Processing {split_name} set: {total_videos} videos")
    print(f"{'='*60}")
    
    processed = 0
    failed = 0
    
    # Process each class
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_path / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        # Create cache directory for this class
        cache_class_dir = cache_path / class_name
        cache_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all videos in this class
        video_files = [f for f in class_dir.iterdir() if f.suffix in video_extensions]
        
        print(f"\nProcessing class '{class_name}': {len(video_files)} videos")
        
        # Process each video with progress bar
        for video_file in tqdm(video_files, desc=f"  {class_name}", ncols=80):
            try:
                # Check if already cached
                cache_file = cache_class_dir / f"{video_file.stem}.pt"
                if cache_file.exists():
                    continue  # Skip already processed videos
                
                # Preprocess video
                video_tensor = preprocess_video(video_file, num_frames=NUM_FRAMES)
                
                # Save preprocessed tensor with label
                torch.save({
                    'video': video_tensor,
                    'label': class_idx,
                    'original_path': str(video_file),
                    'class_name': class_name
                }, cache_file)
                
                processed += 1
                
            except Exception as e:
                print(f"\n  ERROR processing {video_file.name}: {e}")
                failed += 1
    
    print(f"\n{'='*60}")
    print(f"{split_name} set complete:")
    print(f"  ✓ Successfully processed: {processed}")
    if failed > 0:
        print(f"  ✗ Failed: {failed}")
    print(f"  Cache directory: {cache_path}")
    print(f"{'='*60}")


def main():
    """Preprocess all datasets (train, val, test)."""
    print("\n" + "="*60)
    print("Video Preprocessing and Caching for S3D Training")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Frames per video: {NUM_FRAMES}")
    print(f"  - Output size: 224x224")
    print(f"  - Normalization: S3D/Kinetics-400")
    
    # Define paths
    base_dir = Path('./surfing_dataset')
    cache_base_dir = Path('./surfing_dataset_cache')
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        data_dir = base_dir / split
        cache_dir = cache_base_dir / split
        
        if data_dir.exists():
            preprocess_dataset(data_dir, cache_dir, split.upper())
        else:
            print(f"\nWarning: {split} directory not found: {data_dir}")
    
    print("\n" + "="*60)
    print("✓ ALL PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nCached data saved to: {cache_base_dir.absolute()}")
    print("\nYou can now use CachedSurfingManeuverDataset in your training!")
    print("Expected speedup: 10-20x faster data loading")


if __name__ == "__main__":
    main()

