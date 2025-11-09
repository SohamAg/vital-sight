# Batch Processing Pipeline - User Guide

## Overview

The batch processing pipeline automatically processes all videos in `data/demo_clips/` (excluding the `clips/` subfolder) and generates:

1. **Processed Videos** → `data/processed/` (with all YOLO annotations, bounding boxes, poses, etc.)
2. **Gemini Reports** → `data/demo_reports/` (detailed emergency reports for detected situations)

## Quick Start

```bash
# Activate your environment
myenv\Scripts\activate

# Run batch processing with Gemini enabled
python batch_process.py
```

That's it! The script will:
- Find all videos in `data/demo_clips/` (skipping `clips/` folder)
- Process each video without displaying (headless mode)
- Save annotated videos to `data/processed/`
- Generate Gemini reports to `data/demo_reports/`
- Show a summary when complete

## Command Line Options

### Basic Usage

```bash
# Process with default settings (Gemini enabled)
python batch_process.py

# Specify different input directory
python batch_process.py --input-dir path/to/videos

# Exclude multiple subdirectories
python batch_process.py --exclude clips subfolder1 subfolder2

# Use different Gemini API key
python batch_process.py --gemini-key YOUR_API_KEY

# Disable Gemini (only process videos, no reports)
python batch_process.py --no-gemini

# Use custom config file
python batch_process.py --config my_config.yaml
```

### Full Command Reference

```bash
python batch_process.py \
  --input-dir data/demo_clips \    # Input directory
  --exclude clips \                # Folders to skip
  --gemini-key YOUR_KEY \          # Gemini API key
  --config config.yaml             # Config file
```

## Output Structure

```
vital-sight/
├── data/
│   ├── demo_clips/              # Original videos
│   │   ├── fall_sample1.mp4
│   │   ├── fire_sample1.mp4
│   │   └── ...
│   ├── processed/               # 📹 Generated annotated videos
│   │   ├── fall_sample1_processed.mp4
│   │   ├── fire_sample1_processed.mp4
│   │   └── ...
│   └── demo_reports/            # 📄 Generated Gemini reports
│       ├── fall_sample1_fall_report.txt
│       ├── fall_sample1_fall_frame.jpg
│       ├── fire_sample1_fire_report.txt
│       ├── fire_sample1_fire_frame.jpg
│       └── ...
```

## What Gets Generated

### 1. Processed Videos (`data/processed/`)

Each video includes:
- ✅ Person bounding boxes
- ✅ Pose skeletons (if enabled)
- ✅ Detection labels (fall, fire, distress, etc.)
- ✅ Confidence scores
- ✅ FPS counter
- ✅ Motion indicators
- ✅ Impact visualizations (for injury cases)

**Naming:** `{original_name}_processed.mp4`

### 2. Gemini Reports (`data/demo_reports/`)

For each detected situation:
- ✅ Detailed text report with emergency analysis
- ✅ Captured frame (JPG) showing the exact moment
- ✅ Notification protocol based on severity
- ✅ Response time expectations

**Naming:** `{original_name}_{category}_report.txt` and `{category}_frame.jpg`

## Example Run

```bash
python batch_process.py
```

**Console Output:**
```
================================================================================
VITALSIGHT BATCH PROCESSING PIPELINE
================================================================================

Input Directory: data/demo_clips
Excluding: ['clips']
Gemini Reporting: ENABLED
Output Videos: data/processed/
Output Reports: data/demo_reports/

================================================================================

Found 10 video(s) to process:
  1. fall_sample1.mp4
  2. fall_sample2.mp4
  3. fire_sample1.mp4
  4. distress_sample1.mp4
  ...

================================================================================

================================================================================
Processing [1/10]: fall_sample1.mp4
================================================================================

[INFO] Gemini VLM reporting enabled
[INFO] Saving processed video to: data\processed\fall_sample1_processed.mp4
[HINT] Video suggests focus on: fall
...
[FIRST DETECTION] fall detected at confidence 78.5%
[GEMINI] Report generation started in background for fall
...
[GEMINI] ✓ Report saved to: data/demo_reports/fall_sample1_fall_report.txt
[INFO] Processed video saved successfully
[GEMINI] All reports completed.

[✓] Successfully processed: fall_sample1.mp4

--------------------------------------------------------------------------------

... (continues for all videos) ...

================================================================================
BATCH PROCESSING COMPLETE
================================================================================

Successfully processed: 10/10

✓ Success:
  - fall_sample1.mp4
  - fall_sample2.mp4
  - fire_sample1.mp4
  ...

================================================================================
OUTPUT LOCATIONS:
================================================================================

Processed Videos: data/processed/
Detection Reports: data/demo_reports/

Check these directories for results!
================================================================================
```

## Report Format (Updated)

### Example Report: `fall_sample1_fall_report.txt`

```
================================================================================
🚨 VITALSIGHT EMERGENCY DETECTION REPORT
================================================================================

ALERT TYPE: FALL
Severity Level: 🟡 MEDIUM
Detection Confidence: 78.5%
Source: data/demo_clips/fall_sample1.mp4
Timestamp: 2025-11-09 05:30:15
Evidence Frame: fall_sample1_fall_frame.jpg

================================================================================
SITUATION REPORT:
================================================================================

A person is lying flat on the ground in an outdoor paved area, with their body
fully horizontal and limbs extended outward. The individual appears motionless
in a prone position suggesting a sudden fall or collapse. The surrounding 
environment shows a residential setting with no visible obstacles that could 
have caused the fall. Immediate medical assessment is needed to check for 
injuries, particularly head trauma or fractures.

================================================================================
NOTIFICATION PROTOCOL:
================================================================================

Notification Method: EMAIL + SMS
Expected Response Time: PROMPT (< 10 minutes)
Authorities to be contacted: Emergency Services, Security Team, Medical Response

================================================================================
```

### Example Report: `fire_sample1_fire_report.txt`

```
================================================================================
🚨 VITALSIGHT EMERGENCY DETECTION REPORT
================================================================================

ALERT TYPE: FIRE
Severity Level: 🔴 CRITICAL
Detection Confidence: 92.3%
Source: data/demo_clips/fire_sample1.mp4
Timestamp: 2025-11-09 05:32:48
Evidence Frame: fire_sample1_fire_frame.jpg

================================================================================
SITUATION REPORT:
================================================================================

Active flames are visible in an interior space with orange and yellow fire 
consuming what appears to be furniture or structural materials. Heavy smoke 
is present, significantly reducing visibility. The fire appears to be in the 
early to mid stages of development with rapid spread potential. No persons 
are visible in the immediate frame but the structure is clearly at risk of 
catastrophic failure. Immediate evacuation and fire suppression are critical.

================================================================================
NOTIFICATION PROTOCOL:
================================================================================

Notification Method: EMAIL + SMS + IMMEDIATE PHONE CALL
Expected Response Time: IMMEDIATE (< 2 minutes)
Authorities to be contacted: Emergency Services, Security Team, Medical Response

================================================================================
```

## Severity Levels & Notification Protocols

### 🔴 CRITICAL (Immediate Phone Call)
- **Fire**: Highly imminent, rapid escalation risk
- **Distress**: Can lead to cardiac arrest within minutes
- **Severe Injury**: Immediate medical attention prevents death/disability

**Protocol:** EMAIL + SMS + IMMEDIATE PHONE CALL  
**Response Time:** < 2 minutes

### 🟡 MEDIUM (Email/SMS Only)
- **Fall**: Injuries possible but lower immediate life threat
- **Violence/Panic**: Requires security response, coordinated intervention

**Protocol:** EMAIL + SMS  
**Response Time:** < 10 minutes

## Performance Notes

- **Headless Processing**: Videos process faster without display window
- **Parallel Gemini**: Reports generate in background threads, no slowdown
- **Auto-Save**: Both videos and reports save automatically with proper naming
- **Override**: Running twice on same video will overwrite previous outputs

## Troubleshooting

### No videos found
**Check:** Is `data/demo_clips/` the correct path? Are there video files?

### Videos not processing
**Check:** Video formats supported (.mp4, .avi, .mov, .mkv, .flv, .wmv)

### Gemini reports not generating
**Check:** 
- API key is valid
- Internet connection is stable
- Detection is actually happening (some videos may not trigger detections)

### Output video quality
**Default codec:** mp4v (widely compatible)  
**Quality:** Matches input resolution and framerate

## Tips

1. **Test Single Video First**: Before batch processing, test one video with display:
   ```bash
   python main.py --source data/demo_clips/fall_sample1.mp4 --gemini-key YOUR_KEY
   ```

2. **Monitor Console**: Watch for detection messages and errors

3. **Check Outputs Periodically**: Look in `data/processed/` and `data/demo_reports/`

4. **Disk Space**: Processed videos are similar size to originals. Ensure adequate space.

5. **Processing Time**: Expect ~1-2x real-time processing (10 min video = 10-20 min to process)

## What's Different from Single Video Mode

| Feature | Single Video | Batch Processing |
|---------|-------------|------------------|
| Display | Optional | Disabled (headless) |
| Video Output | Optional | Automatic |
| Gemini Reports | Optional | Automatic |
| Override Reports | Yes | Yes |
| Speed | Normal | Optimized |
| Progress Tracking | Video window | Console output |

## Next Steps

After batch processing completes:

1. **Review Videos**: Check `data/processed/` for annotated videos
2. **Review Reports**: Check `data/demo_reports/` for Gemini analyses
3. **Validate Detections**: Ensure accuracy matches expectations
4. **Archive/Share**: Videos and reports are ready for documentation

## Advanced Usage

### Process Specific Videos Only

Create a custom folder:
```bash
mkdir data/to_process
# Copy specific videos there
python batch_process.py --input-dir data/to_process --exclude ""
```

### Disable Gemini for Speed

```bash
python batch_process.py --no-gemini
# Only generates processed videos, no reports
```

### Custom Config

```bash
# Use different detection thresholds
python batch_process.py --config custom_config.yaml
```

---

**Ready to process? Run:**
```bash
python batch_process.py
```

And let VitalSight do its magic! 🚀

