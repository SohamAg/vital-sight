"""
Vital Sight - CCTV Distress Detection System
Main application file
"""

import cv2
import argparse
import time
import sys
import os
from pose_estimator import PoseEstimator
from distress_detector import DistressDetector
from notification_system import NotificationSystem
import config


class VitalSight:
    """Main application class for distress detection"""
    
    def __init__(self, video_source):
        self.video_source = video_source
        self.pose_estimator = PoseEstimator()
        self.distress_detector = DistressDetector()
        self.notification_system = NotificationSystem()
        
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        print("\n" + "="*60)
        print("VITAL SIGHT - DISTRESS DETECTION SYSTEM")
        print("="*60)
        print(f"Video Source: {video_source}")
        print(f"Fall Detection Threshold: {config.FALL_DETECTION_THRESHOLD}°")
        print(f"Distress Time Threshold: {config.DISTRESS_TIME_THRESHOLD}s")
        print("="*60 + "\n")
    
    def process_video(self):
        """Main video processing loop"""
        
        # Open video source
        if isinstance(self.video_source, str) and self.video_source.isdigit():
            # Webcam
            cap = cv2.VideoCapture(int(self.video_source))
        else:
            # Video file or stream
            cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {self.video_source}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video Resolution: {frame_width}x{frame_height}")
        print(f"Video FPS: {video_fps:.2f}\n")
        print("Press 'q' to quit, 'p' to pause\n")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("\nEnd of video or error reading frame")
                        break
                    
                    self.frame_count += 1
                    
                    # Skip frames if configured
                    if self.frame_count % config.FRAME_SKIP != 0:
                        continue
                    
                    # Process frame
                    processed_frame = self.process_frame(frame, frame_width, frame_height)
                    
                    # Calculate FPS
                    elapsed_time = time.time() - self.start_time
                    if elapsed_time > 0:
                        self.fps = self.frame_count / elapsed_time
                    
                    # Display FPS if enabled
                    if config.DISPLAY_FPS:
                        cv2.putText(processed_frame, f"FPS: {self.fps:.1f}", 
                                  (10, frame_height - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Show frame
                    cv2.imshow('Vital Sight - Distress Detection', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('p'):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"\n{status}")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.pose_estimator.close()
            print("\n✓ Application closed cleanly\n")
    
    def process_frame(self, frame, frame_width, frame_height):
        """Process a single frame for distress detection"""
        
        # Make a copy for display
        display_frame = frame.copy()
        
        # Draw detection zones if configured
        if config.DRAW_DETECTION_ZONES and config.DETECTION_ZONES:
            for zone in config.DETECTION_ZONES:
                x1, y1, x2, y2 = zone
                cv2.rectangle(display_frame,
                            (int(x1 * frame_width), int(y1 * frame_height)),
                            (int(x2 * frame_width), int(y2 * frame_height)),
                            config.ZONE_COLOR, 2)
        
        # Run pose estimation
        results = self.pose_estimator.process_frame(frame)
        landmarks = self.pose_estimator.get_landmarks(results)
        
        if landmarks:
            # Check if person is in detection zone
            in_zone = self.pose_estimator.is_person_in_zone(
                landmarks, frame_width, frame_height, config.DETECTION_ZONES
            )
            
            if in_zone:
                # Detect distress
                distress_info = self.distress_detector.get_distress_info(landmarks, person_id=0)
                
                # Determine visualization color
                if distress_info['is_distress']:
                    color = config.ALERT_COLOR
                    label = f"⚠️ {distress_info['distress_type']} ({distress_info['confidence']:.0%})"
                    
                    # Send alert
                    self.notification_system.send_alert(frame, distress_info, person_id=0)
                else:
                    color = config.NORMAL_COLOR
                    label = "Normal"
                
                # Draw bounding box if enabled
                if config.DRAW_BOUNDING_BOXES:
                    bbox = self.pose_estimator.get_bounding_box(landmarks, frame_width, frame_height)
                    display_frame = self.pose_estimator.draw_bounding_box(
                        display_frame, bbox, color, label
                    )
                
                # Draw pose landmarks if enabled
                if config.DRAW_POSE_LANDMARKS:
                    display_frame = self.pose_estimator.draw_landmarks(
                        display_frame, results, color=color
                    )
                
                # Display debug info
                if config.DEBUG_MODE and distress_info['body_angle'] is not None:
                    debug_text = f"Angle: {distress_info['body_angle']:.1f}° | Pos: {distress_info['vertical_position']:.2f}"
                    cv2.putText(display_frame, debug_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def process_batch(self, video_paths):
        """Process multiple video files"""
        for video_path in video_paths:
            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")
                continue
            
            print(f"\n\nProcessing: {video_path}")
            print("-" * 60)
            
            self.video_source = video_path
            self.frame_count = 0
            self.start_time = time.time()
            
            self.process_video()


def download_sample_video():
    """Download a sample video for testing"""
    import requests
    
    # Create sample_videos directory
    os.makedirs("sample_videos", exist_ok=True)
    
    # Sample fall detection videos (public domain or creative commons)
    sample_urls = [
        "https://github.com/MVIG-SJTU/AlphaPose/raw/master/examples/demo/2.mp4",
    ]
    
    print("\nDownloading sample video...")
    
    for i, url in enumerate(sample_urls):
        try:
            filename = f"sample_videos/sample_{i+1}.mp4"
            
            if os.path.exists(filename):
                print(f"✓ {filename} already exists")
                continue
            
            response = requests.get(url, stream=True, timeout=30)
            
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✓ Downloaded: {filename}")
                return filename
            else:
                print(f"✗ Failed to download from {url}")
        except Exception as e:
            print(f"✗ Error downloading sample video: {e}")
    
    return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Vital Sight - CCTV Distress Detection System'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='0',
        help='Input video source (file path, URL, or webcam index). Default: 0 (webcam)'
    )
    parser.add_argument(
        '--batch', '-b',
        nargs='+',
        help='Process multiple video files'
    )
    parser.add_argument(
        '--download-sample',
        action='store_true',
        help='Download sample video for testing'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f'Detection confidence threshold (default: {config.CONFIDENCE_THRESHOLD})'
    )
    
    args = parser.parse_args()
    
    # Update config with command line args
    config.CONFIDENCE_THRESHOLD = args.confidence
    
    # Download sample video if requested
    if args.download_sample:
        sample_file = download_sample_video()
        if sample_file:
            args.input = sample_file
        else:
            print("\nFailed to download sample video.")
            print("Please provide your own video file or use webcam (--input 0)")
            return
    
    # Create application instance
    app = VitalSight(args.input)
    
    # Process batch or single video
    if args.batch:
        app.process_batch(args.batch)
    else:
        app.process_video()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
