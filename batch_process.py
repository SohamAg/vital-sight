"""
Batch Processing Pipeline for VitalSight
Processes all videos in demo_clips folder and generates:
1. Annotated videos with YOLO detections -> data/processed/
2. Gemini VLM reports for detected situations -> data/demo_reports/
"""
import os
import argparse
from pathlib import Path
from edge.detector_v2 import VitalSightV2

# Your API key
DEFAULT_API_KEY = "AIzaSyDPE3QNZqVino7KJvFDeZ_nfYcQ627FcMo"

def find_videos(directory, exclude_dirs=None):
    """
    Find all video files in directory, excluding specified subdirectories.
    
    Args:
        directory: Root directory to search
        exclude_dirs: List of subdirectory names to exclude
        
    Returns:
        List of video file paths
    """
    if exclude_dirs is None:
        exclude_dirs = []
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    videos = []
    
    directory = Path(directory)
    
    for file_path in directory.iterdir():
        # Skip directories in exclude list
        if file_path.is_dir() and file_path.name in exclude_dirs:
            print(f"[SKIP] Excluding directory: {file_path.name}")
            continue
        
        # Only process files with video extensions
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            videos.append(file_path)
    
    return sorted(videos)

def clean_output_directories():
    """Clean processed videos and reports before batch processing"""
    import shutil
    
    processed_dir = Path("data/processed")
    reports_dir = Path("data/demo_reports")
    
    print("\n🗑️  CLEANING OUTPUT DIRECTORIES...")
    
    # Clean processed videos
    if processed_dir.exists():
        for file in processed_dir.glob("*"):
            if file.is_file():
                file.unlink()
                print(f"   Deleted: {file.name}")
        print(f"✓ Cleaned: data/processed/")
    
    # Clean reports
    if reports_dir.exists():
        for file in reports_dir.glob("*"):
            if file.is_file():
                file.unlink()
                print(f"   Deleted: {file.name}")
        print(f"✓ Cleaned: data/demo_reports/")
    
    print()

def process_batch(input_dir="data/demo_clips", 
                 exclude_dirs=None,
                 gemini_api_key=None,
                 config_path="config.yaml",
                 clean_first=True):
    """
    Process all videos in batch mode.
    
    Args:
        input_dir: Directory containing input videos
        exclude_dirs: Subdirectories to exclude (e.g., ['clips'])
        gemini_api_key: Google Gemini API key for report generation
        config_path: Path to config file
        clean_first: If True, delete all existing processed videos and reports
    """
    if exclude_dirs is None:
        exclude_dirs = ['clips']
    
    print("=" * 80)
    print("VITALSIGHT BATCH PROCESSING PIPELINE")
    print("=" * 80)
    print(f"\nInput Directory: {input_dir}")
    print(f"Excluding: {exclude_dirs}")
    print(f"Gemini Reporting: {'ENABLED' if gemini_api_key else 'DISABLED'}")
    print(f"Output Videos: data/processed/")
    print(f"Output Reports: data/demo_reports/")
    print(f"Clean First: {'YES' if clean_first else 'NO'}")
    print("\n" + "=" * 80)
    
    # Clean output directories if requested
    if clean_first:
        clean_output_directories()
    
    # Find all videos
    videos = find_videos(input_dir, exclude_dirs)
    
    if not videos:
        print(f"\n[ERROR] No videos found in {input_dir}")
        return
    
    print(f"Found {len(videos)} video(s) to process:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video.name}")
    print("\n" + "=" * 80)
    
    # Create output directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/demo_reports").mkdir(parents=True, exist_ok=True)
    
    # Process each video
    results = {
        "success": [],
        "failed": []
    }
    
    for i, video_path in enumerate(videos, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing [{i}/{len(videos)}]: {video_path.name}")
        print(f"{'=' * 80}\n")
        
        try:
            # Create detector instance for this video
            detector = VitalSightV2(cfg_path=config_path, gemini_api_key=gemini_api_key)
            
            # Process video with output saving, no display
            detector.process(
                source=str(video_path),
                display=False,
                save_output=True
            )
            
            results["success"].append(video_path.name)
            print(f"\n[✓] Successfully processed: {video_path.name}")
            
        except Exception as e:
            results["failed"].append((video_path.name, str(e)))
            print(f"\n[✗] Failed to process {video_path.name}: {e}")
        
        print(f"\n{'-' * 80}\n")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nSuccessfully processed: {len(results['success'])}/{len(videos)}")
    
    if results["success"]:
        print("\n✓ Success:")
        for video in results["success"]:
            print(f"  - {video}")
    
    if results["failed"]:
        print("\n✗ Failed:")
        for video, error in results["failed"]:
            print(f"  - {video}: {error}")
    
    print("\n" + "=" * 80)
    print("OUTPUT LOCATIONS:")
    print("=" * 80)
    print(f"\nProcessed Videos: data/processed/")
    print(f"Detection Reports: data/demo_reports/")
    print("\nCheck these directories for results!")
    print("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Batch process videos with VitalSight detection and Gemini reporting"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/demo_clips",
        help="Directory containing input videos (default: data/demo_clips)"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=["clips"],
        help="Subdirectories to exclude (default: clips)"
    )
    parser.add_argument(
        "--gemini-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="Google Gemini API key"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Disable Gemini report generation"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't clean output directories before processing"
    )
    
    args = parser.parse_args()
    
    # Disable Gemini if flag is set
    gemini_key = None if args.no_gemini else args.gemini_key
    
    # Run batch processing
    process_batch(
        input_dir=args.input_dir,
        exclude_dirs=args.exclude,
        gemini_api_key=gemini_key,
        config_path=args.config,
        clean_first=not args.no_clean
    )

if __name__ == "__main__":
    main()

