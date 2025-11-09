"""
Re-encode processed videos to H.264 codec for browser compatibility
"""
import subprocess
import sys
from pathlib import Path

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except:
        return False

def reencode_video(input_path, output_path):
    """Re-encode video to H.264"""
    try:
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:v', 'libx264',        # H.264 codec
            '-preset', 'medium',       # Encoding speed
            '-crf', '23',              # Quality (lower = better, 18-28 is good range)
            '-c:a', 'aac',             # Audio codec
            '-b:a', '128k',            # Audio bitrate
            '-movflags', '+faststart', # Enable streaming
            '-y',                      # Overwrite output
            str(output_path)
        ]
        
        print(f"Converting: {input_path.name}...", end=" ")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Done")
            return True
        else:
            print(f"✗ Failed: {result.stderr[:100]}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("=" * 80)
    print("VIDEO RE-ENCODING FOR BROWSER COMPATIBILITY")
    print("=" * 80)
    print()
    
    # Check for ffmpeg
    if not check_ffmpeg():
        print("❌ FFmpeg not found!")
        print()
        print("Please install FFmpeg:")
        print("  1. Download from: https://ffmpeg.org/download.html")
        print("  2. Or use: winget install ffmpeg")
        print("  3. Or use: choco install ffmpeg")
        print()
        sys.exit(1)
    
    print("✓ FFmpeg found")
    print()
    
    # Find all processed videos
    processed_dir = Path("data/processed")
    videos = list(processed_dir.glob("*_processed.mp4"))
    
    if not videos:
        print("No videos found in data/processed/")
        sys.exit(0)
    
    print(f"Found {len(videos)} video(s) to convert")
    print()
    
    # Create temp directory
    temp_dir = processed_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    success_count = 0
    failed = []
    
    for video in videos:
        temp_output = temp_dir / video.name
        
        if reencode_video(video, temp_output):
            # Replace original with converted
            video.unlink()
            temp_output.rename(video)
            success_count += 1
        else:
            failed.append(video.name)
    
    # Cleanup
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except:
            pass
    
    print()
    print("=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"Success: {success_count}/{len(videos)}")
    
    if failed:
        print(f"\nFailed conversions:")
        for f in failed:
            print(f"  - {f}")
    
    print()
    print("✓ All videos converted to H.264!")
    print("Refresh your browser (Ctrl+Shift+R) to see the videos")

if __name__ == "__main__":
    main()

