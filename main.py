import argparse, yaml
import os
from edge.detector_v2 import VitalSightV2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="0 for webcam or path to video")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--gemini-key", type=str, default=None, help="Google Gemini API key for VLM reporting")
    ap.add_argument("--enable-twilio", action="store_true", help="Enable Twilio SMS/call alerts")
    args = ap.parse_args()

    # Get Gemini API key from args or environment variable
    gemini_api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
    
    # Get Twilio config from environment variables if enabled
    twilio_config = None
    if args.enable_twilio:
        twilio_config = {
            'account_sid': os.environ.get('TWILIO_ACCOUNT_SID'),
            'auth_token': os.environ.get('TWILIO_AUTH_TOKEN'),
            'from_number': os.environ.get('TWILIO_PHONE_NUMBER'),
            'to_number': os.environ.get('ALERT_PHONE_NUMBER')
        }
    
    vs = VitalSightV2(cfg_path=args.config, gemini_api_key=gemini_api_key, twilio_config=twilio_config)
    source = 0 if args.source == "0" else args.source
    vs.process(source=source, display=not args.no_display)

if __name__ == "__main__":
    main()
