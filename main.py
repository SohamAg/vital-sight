import argparse, yaml
from edge.detector_v2 import VitalSightV2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="0 for webcam or path to video")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    vs = VitalSightV2(cfg_path=args.config)
    source = 0 if args.source == "0" else args.source
    vs.process(source=source, display=not args.no_display)

if __name__ == "__main__":
    main()
