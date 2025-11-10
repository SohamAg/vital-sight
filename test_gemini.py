"""
Quick test script for Gemini VLM integration
"""
import argparse
from edge.detector_v2 import VitalSightV2

# Your API key (you can also use --gemini-key argument or GEMINI_API_KEY env var)
DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY")

def test_sample_videos():
    """Test Gemini integration with sample videos"""
    
    test_videos = [
        "data/demo_clips/fall_sample1.mp4",
        "data/demo_clips/fire_sample1.mp4",
        "data/demo_clips/distress_sample1.mp4",
        "data/demo_clips/injury_sample1.mp4",
        "data/demo_clips/crowd_sample1.mp4",
    ]
    
    print("=" * 80)
    print("VITALSIGHT GEMINI VLM INTEGRATION TEST")
    print("=" * 80)
    print("\nThis will process sample videos and generate Gemini reports.")
    print("Reports will be saved to: data/demo_reports/")
    print("\nVideos to test:")
    for i, video in enumerate(test_videos, 1):
        print(f"  {i}. {video}")
    print("\n" + "=" * 80)
    
    choice = input("\nPress ENTER to test all, or enter video number (1-5), or 'q' to quit: ").strip()
    
    if choice.lower() == 'q':
        print("Cancelled.")
        return
    
    if choice.isdigit() and 1 <= int(choice) <= len(test_videos):
        videos_to_test = [test_videos[int(choice) - 1]]
    else:
        videos_to_test = test_videos
    
    print(f"\n[INFO] Testing {len(videos_to_test)} video(s)...\n")
    
    for video in videos_to_test:
        print("\n" + "=" * 80)
        print(f"Testing: {video}")
        print("=" * 80)
        
        try:
            vs = VitalSightV2(cfg_path="config.yaml", gemini_api_key=DEFAULT_API_KEY)
            vs.process(source=video, display=True)
            print(f"\n[SUCCESS] Completed: {video}")
        except Exception as e:
            print(f"\n[ERROR] Failed to process {video}: {e}")
        
        print("\n" + "-" * 80)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nCheck the 'data/demo_reports/' folder for generated reports!")
    print("Each report includes:")
    print("  - A detailed text report from Gemini VLM")
    print("  - The captured frame where detection occurred")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Gemini VLM integration")
    parser.add_argument("--video", type=str, help="Specific video to test")
    parser.add_argument("--gemini-key", type=str, default=DEFAULT_API_KEY, 
                       help="Google Gemini API key")
    args = parser.parse_args()
    
    if args.video:
        # Test specific video
        print(f"Testing single video: {args.video}")
        vs = VitalSightV2(cfg_path="config.yaml", gemini_api_key=args.gemini_key)
        vs.process(source=args.video, display=True)
    else:
        # Interactive test mode
        test_sample_videos()

