import argparse, yaml
import os
from edge.detector_v2 import VitalSightV2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="0 for webcam or path to video")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--gemini-key", type=str, default=None, help="Google Gemini API key for VLM reporting")
    args = ap.parse_args()

    # Get Gemini API key from args or environment variable
    gemini_api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
    
    vs = VitalSightV2(cfg_path=args.config, gemini_api_key=gemini_api_key)
    source = 0 if args.source == "0" else args.source
    vs.process(source=source, display=not args.no_display)

if __name__ == "__main__":
    main()
