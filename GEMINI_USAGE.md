# Gemini VLM Integration Guide

## Overview

VitalSight now integrates with Google Gemini Vision Language Model (VLM) to generate detailed reports when situations are first detected. When the system detects a fall, fire, distress, violence/panic, or severe injury for the first time in a video, it automatically:

1. Captures the first frame where the situation was detected
2. Sends it to Google Gemini VLM for analysis
3. Generates a detailed text report
4. Saves both the frame and report to `data/demo_reports/`

## Setup

### 1. API Key

You have three ways to provide your Gemini API key:

**Option A: Command Line Argument (Recommended for testing)**
```bash
python main.py --source data/demo_clips/fall_sample1.mp4 --gemini-key AIzaSyDPE3QNZqVino7KJvFDeZ_nfYcQ627FcMo
```

**Option B: Environment Variable (Recommended for production)**
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="AIzaSyDPE3QNZqVino7KJvFDeZ_nfYcQ627FcMo"
python main.py --source data/demo_clips/fall_sample1.mp4

# Windows Command Prompt
set GEMINI_API_KEY=AIzaSyDPE3QNZqVino7KJvFDeZ_nfYcQ627FcMo
python main.py --source data/demo_clips/fall_sample1.mp4

# Linux/Mac
export GEMINI_API_KEY="AIzaSyDPE3QNZqVino7KJvFDeZ_nfYcQ627FcMo"
python main.py --source data/demo_clips/fall_sample1.mp4
```

**Option C: No API Key (Disables Gemini)**
```bash
python main.py --source data/demo_clips/fall_sample1.mp4
# System will work normally but won't generate Gemini reports
```

## Usage Examples

### Test with Demo Videos

```bash
# Activate virtual environment
myenv\Scripts\activate

# Test fall detection with Gemini
python main.py --source data/demo_clips/fall_sample1.mp4 --gemini-key YOUR_API_KEY

# Test fire detection with Gemini
python main.py --source data/demo_clips/fire_sample1.mp4 --gemini-key YOUR_API_KEY

# Test distress detection with Gemini
python main.py --source data/demo_clips/distress_sample1.mp4 --gemini-key YOUR_API_KEY

# Test injury detection with Gemini
python main.py --source data/demo_clips/injury_sample1.mp4 --gemini-key YOUR_API_KEY

# Test crowd/violence detection with Gemini
python main.py --source data/demo_clips/crowd_sample1.mp4 --gemini-key YOUR_API_KEY
```

## Output Files

Reports are saved to `data/demo_reports/` with the following naming convention:

- **Report**: `{video_name}_{category}_report.txt`
- **Frame**: `{video_name}_{category}_frame.jpg`

### Example Output Structure
```
data/demo_reports/
├── fall_sample1_fall_report.txt
├── fall_sample1_fall_frame.jpg
├── fire_sample1_fire_report.txt
├── fire_sample1_fire_frame.jpg
├── distress_sample1_distress_report.txt
├── distress_sample1_distress_frame.jpg
└── ...
```

## Report Format

Each report contains:

1. **Header Section**
   - Detection category
   - Confidence score
   - Source video path
   - Timestamp
   - Reference to captured frame

2. **Gemini VLM Analysis**
   - Visual evidence supporting the detection
   - Number of people involved
   - Specific actions or states visible
   - Environmental hazards or conditions
   - Urgency level and immediate concerns

### Example Report

```
================================================================================
VITALSIGHT AI DETECTION REPORT
================================================================================

Category: Fall
Confidence: 78.50%
Source: data/demo_clips/fall_sample1.mp4
Timestamp: 2025-11-09 04:58:32
Frame Image: fall_sample1_fall_frame.jpg

--------------------------------------------------------------------------------
GEMINI VLM ANALYSIS:
--------------------------------------------------------------------------------

The image shows a person lying horizontally on the ground in an outdoor setting,
confirming the fall detection. The individual's body position is fully prone
with limbs extended, indicating a sudden loss of balance or collapse. The 
surrounding environment appears to be a paved area near a residential building.
This requires immediate medical attention as the person is motionless and may
have sustained injuries from the fall. Emergency services should be contacted
immediately.

================================================================================
```

## How It Works

1. **First Detection**: System tracks when each category (fall, fire, distress, etc.) is first detected
2. **Frame Capture**: The exact frame where detection occurs is captured in full resolution
3. **Gemini Analysis**: Frame is sent to Gemini with a specialized prompt that:
   - Assumes our detection is correct
   - Asks for visual confirmation and detailed analysis
   - Requests professional, actionable insights
4. **Report Generation**: Both the frame and Gemini's analysis are saved to disk
5. **One Report Per Category**: Each category only generates ONE report per video (first detection only)

## Features

- ✅ **Automatic**: No manual intervention needed
- ✅ **Non-blocking**: Report generation happens during detection, doesn't slow down processing
- ✅ **Comprehensive**: Captures both visual evidence (frame) and textual analysis (report)
- ✅ **Professional Format**: Reports are structured and ready for documentation
- ✅ **Matches Video Names**: Report filenames mirror video filenames for easy tracking
- ✅ **Error Handling**: Gracefully handles API failures and network issues

## Troubleshooting

### "Failed to initialize Gemini reporter"
- Check your API key is valid
- Ensure you have internet connectivity
- Verify the google-generativeai package is installed: `pip install google-generativeai`

### "Failed to generate report"
- Check your Gemini API quota/limits
- Verify internet connection is stable
- Check the error message in console for specific details

### Reports not generating
- Ensure API key is provided via command line or environment variable
- Check that detections are actually happening (see console output)
- Verify `data/demo_reports/` directory exists

## Technical Details

### Prompt Engineering

The system uses a carefully crafted prompt that:
- Establishes Gemini as a security/safety expert
- Provides our detection category and confidence
- Instructs Gemini to assume we're correct
- Requests specific types of analysis (visual evidence, people count, actions, hazards, urgency)
- Enforces professional, concise format (3-5 sentences)

### Performance

- Uses Gemini 1.5 Flash for fast analysis
- Frame upload and analysis typically takes 2-5 seconds
- Does not block real-time video processing
- Only generates ONE report per category per video

### Privacy & Security

- Frames are temporarily uploaded to Google's Gemini API
- Uploaded files are deleted after analysis
- Reports and frames are stored locally only
- API key should be kept secure (use environment variables in production)

## Future Enhancements

Possible improvements:
- Asynchronous/threaded report generation for zero impact on processing
- Batch processing multiple videos
- Email/SMS integration for automatic alert distribution
- Cloud storage integration for reports
- Customizable prompt templates per category
- Multi-language report generation

