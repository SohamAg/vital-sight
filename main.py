#!/usr/bin/env python3
"""
VitalSight Emergency Detection System
Main entry point for the application

Author: VitalSight Team
Description: Real-time emergency action detection using YOLO-Pose and custom algorithms
"""

import sys
import os
import argparse
from video_processor import VideoProcessor


def print_banner():
    """Print application banner"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║              VITALSIGHT EMERGENCY DETECTION SYSTEM            ║
║                                                               ║
║  Real-time Emergency Action Detection using YOLO-Pose        ║
║  + Custom Rule-Based Algorithms                              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def print_system_info():
    """Print system information"""
    print("\n[System Information]")
    print("─" * 60)
    print("Detection Engine: YOLO-Pose + Custom Algorithms")
    print("VLM Status: DISABLED")
    print("MCP Integration: Enabled (Mock Server)")
    print("─" * 60)
    print("\n[Detectable Emergency Actions]")
    print("  1. Violence/Assault        → Email Alert")
    print("  2. Sudden Falling           → Email Alert")
    print("  3. Fainting/Collapse        → Email Alert")
    print("  4. Cardiac Distress         → Phone Call")
    print("  5. Fire (Limited)           → Phone Call")
    print("─" * 60)


def list_sample_videos():
    """List available sample videos"""
    sample_dir = "sample_videos"
    if not os.path.exists(sample_dir):
        print(f"\n[Warning] Sample videos directory not found: {sample_dir}")
        return []
    
    videos = [f for f in os.listdir(sample_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if videos:
        print(f"\n[Available Sample Videos]")
        print("─" * 60)
        for i, video in enumerate(videos, 1):
            filepath = os.path.join(sample_dir, video)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {i}. {video:<40} ({size_mb:.1f} MB)")
        print("─" * 60)
    
    return videos


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='VitalSight Emergency Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use webcam
  python main.py
  
  # Process video file
  python main.py --source sample_videos/fall.mp4
  
  # Process webcam with specific ID
  python main.py --source 0
  
  # List available sample videos
  python main.py --list-samples
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default=None,
        help='Video source (webcam ID or video file path). Default: webcam (0)'
    )
    
    parser.add_argument(
        '--list-samples', '-l',
        action='store_true',
        help='List available sample videos and exit'
    )
    
    parser.add_argument(
        '--sample', '-p',
        type=int,
        help='Process sample video by index (use --list-samples to see indices)'
    )
    
    return parser.parse_args()


def main():
    """Main application entry point"""
    # Print banner
    print_banner()
    
    # Parse arguments
    args = parse_arguments()
    
    # Print system info
    print_system_info()
    
    # List sample videos if requested
    sample_videos = list_sample_videos()
    
    if args.list_samples:
        print("\nUse --sample <index> to process a specific sample video")
        return 0
    
    # Determine video source
    source = args.source
    
    if args.sample is not None:
        if not sample_videos:
            print("\n[Error] No sample videos found")
            return 1
        
        if args.sample < 1 or args.sample > len(sample_videos):
            print(f"\n[Error] Invalid sample index. Must be between 1 and {len(sample_videos)}")
            return 1
        
        source = os.path.join("sample_videos", sample_videos[args.sample - 1])
        print(f"\n[Selected] Processing sample video: {source}")
    
    if source is None:
        print("\n[Mode] Using webcam (default)")
        print("[Info] Press 'q' in the video window to quit")
    else:
        if not source.isdigit() and not os.path.exists(source):
            print(f"\n[Error] Video file not found: {source}")
            return 1
        print(f"\n[Mode] Processing video source: {source}")
        print("[Info] Press 'q' in the video window to quit")
    
    print("\n[Starting] Initializing system...")
    print("─" * 60)
    
    try:
        # Initialize and start video processor
        processor = VideoProcessor(source)
        processor.start()
        
        print("\n[Completed] Processing finished successfully")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Processing interrupted by user")
        return 0
        
    except Exception as e:
        print(f"\n[Error] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
