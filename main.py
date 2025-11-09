from edge.detector import VitalSightDetector
import argparse

def main():
    parser = argparse.ArgumentParser(description="VitalSight Detection System")
    parser.add_argument("--source", type=str, default="0",
                        help="Path to video file or webcam index (default 0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Path to YOLO model")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable visualization window")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated video")
    args = parser.parse_args()

    source = 0 if args.source == "0" else args.source

    detector = VitalSightDetector(model_path=args.model)
    detector.process_video(source=source,
                           display=not args.no_display,
                           save_output=args.save)

if __name__ == "__main__":
    main()
