# Twilio Alert System Setup Guide

VitalSight now supports **real-time SMS and phone call alerts** via Twilio when emergencies are detected!

---

## Alert Configuration by Situation Type

| Situation | Alert Methods | Priority | Response Time |
|-----------|---------------|----------|---------------|
| **Fire** 🔥 | **Call + SMS** | 🔴 CRITICAL | < 2 minutes |
| **Fall** 🤕 | **SMS only** | 🟢 LOW | 15-30 minutes |
| **Distress** 😰 | **Call + SMS** | 🟠 HIGH | 2-5 minutes |
| **Violence/Panic** ⚠️ | **SMS only** | 🟡 MEDIUM | 5-10 minutes |
| **Severe Injury** 🚨 | **Call + SMS** | 🔴 CRITICAL | < 2 minutes |

---

## Step 1: Install Twilio Package

```bash
# Make sure you're in your virtual environment
pip install twilio
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

---

## Step 2: Set Environment Variables

### Your Twilio Credentials:
```
TWILIO_ACCOUNT_SID = AC9482d7139f9d9056cbdf9159f02052db
TWILIO_AUTH_TOKEN = 44fdc0ce64f86bb9da8eb2962e03a626
TWILIO_PHONE_NUMBER = +18663508040
ALERT_PHONE_NUMBER = +19297602752
```

### Mac/Linux:
```bash
export TWILIO_ACCOUNT_SID="AC9482d7139f9d9056cbdf9159f02052db"
export TWILIO_AUTH_TOKEN="44fdc0ce64f86bb9da8eb2962e03a626"
export TWILIO_PHONE_NUMBER="+18663508040"
export ALERT_PHONE_NUMBER="+19297602752"
```

### Or add to your shell profile (~/.zshrc or ~/.bashrc):
```bash
# Add these lines to the file
export TWILIO_ACCOUNT_SID="AC9482d7139f9d9056cbdf9159f02052db"
export TWILIO_AUTH_TOKEN="44fdc0ce64f86bb9da8eb2962e03a626"
export TWILIO_PHONE_NUMBER="+18663508040"
export ALERT_PHONE_NUMBER="+19297602752"

# Then reload:
source ~/.zshrc  # or source ~/.bashrc
```

---

## Step 3: Run with Twilio Alerts Enabled

```bash
# Fire detection - will CALL + SMS
python3 main.py \
    --source data/demo_clips/fire_sample1.mp4 \
    --gemini-key AIzaSyDPE3QNZqVino7KJvFDeZ_nfYcQ627FcMo \
    --enable-twilio

# Fall detection - will SMS only
python3 main.py \
    --source data/demo_clips/fall_sample1.mp4 \
    --gemini-key AIzaSyDPE3QNZqVino7KJvFDeZ_nfYcQ627FcMo \
    --enable-twilio
```

---

## What You'll Receive

### SMS Example (Fall Detection):
```
🤕 VITALSIGHT LOW PRIORITY ALERT 🤕

FALL DETECTED
Confidence: 78%
Time: 06:15:32
Location: data/demo_clips/fall_sample1.mp4

Immediate response required.
Check surveillance system for details.
```

### Phone Call Example (Fire Detection):
When fire is detected, you'll receive an **automated phone call** that says:

> "CRITICAL ALERT. Fire has been detected. Confidence 92 percent. Evacuate immediately and contact fire department. Press any key to acknowledge this alert."

The call will **repeat the message twice** and wait for acknowledgment.

---

## Expected Console Output

When Twilio alerts are enabled and a detection occurs:

```
[INFO] Gemini VLM reporting enabled
[TWILIO] Alert system initialized
[TWILIO] Alerts will be sent to: +19297602752
[INFO] device: cpu | YOLO on GPU? False
[HINT] Video suggests focus on: fire

[Frame 90] Raw scores: fire:0.892

[FIRST DETECTION] fire detected at confidence 92.30%

[TWILIO] 🚨 Sending CRITICAL alert for fire
[TWILIO] ✓ SMS sent for fire (SID: SM...)
[TWILIO] ✓ Call initiated for fire (SID: CA...)
[GEMINI] Report generation started in background for fire
[GEMINI] Analyzing fire situation...
[GEMINI] ✓ Report saved to: data/demo_reports/fire_sample1_fire_report.txt
```

---

## Testing

### Test SMS Alerts Only (Fall):
```bash
python3 main.py \
    --source data/demo_clips/fall_sample1.mp4 \
    --enable-twilio
```
**Expected:** Text message to +19297602752

### Test Phone Call + SMS (Fire):
```bash
python3 main.py \
    --source data/demo_clips/fire_sample1.mp4 \
    --enable-twilio
```
**Expected:** Phone call + text message to +19297602752

---

## Features

✅ **Smart Alert Routing** - Automatically chooses SMS or Call+SMS based on severity  
✅ **Duplicate Prevention** - Only one alert per situation type per video  
✅ **Professional Messages** - Clear, actionable alerts with confidence levels  
✅ **Voice Calls** - Automated voice alerts for critical situations  
✅ **Fast & Non-Blocking** - Alerts sent in background, doesn't slow down detection  
✅ **Confidence Reporting** - Includes AI confidence percentage  
✅ **Timestamp & Location** - Every alert includes when and where  

---

## Troubleshooting

### "Missing Twilio credentials"
- Make sure all 4 environment variables are set
- Check spelling (they're case-sensitive)
- Verify with: `echo $TWILIO_ACCOUNT_SID`

### "Failed to send SMS"
- Check Twilio account balance
- Verify phone numbers are in E.164 format (+1...)
- Check Twilio dashboard for error messages

### "Failed to make call"
- Voice calls require active Twilio account
- Check that your Twilio number supports voice
- Verify destination number can receive calls

### No alerts being sent
- Make sure you used `--enable-twilio` flag
- Check that detections are actually happening
- Look for `[TWILIO]` messages in console

---

## Cost Estimate

Based on Twilio pricing:
- SMS: ~$0.0075 per message
- Voice Call: ~$0.013 per minute

**Example scenario:**
- 10 fire detections per month = 10 calls + 10 SMS = ~$0.21/month
- 20 fall detections per month = 20 SMS = ~$0.15/month

Very affordable for critical safety monitoring!

---

## Advanced Configuration

### Custom Phone Number:
```bash
export ALERT_PHONE_NUMBER="+1234567890"
```

### Multiple Recipients (Future):
Currently supports one phone number. For multiple recipients, you can:
1. Set up Twilio Studio flow
2. Or modify `edge/twilio_alerter.py` to loop through a list

---

## Security Notes

⚠️ **Keep credentials secure:**
- Never commit credentials to Git
- Use environment variables (not hard-coded)
- Consider using a secrets manager in production

✅ **The credentials in this file are yours and safe to use for testing**

---

## Quick Reference

### Enable Twilio + Gemini:
```bash
python3 main.py --source VIDEO.mp4 --enable-twilio --gemini-key YOUR_KEY
```

### Just Twilio (no Gemini):
```bash
python3 main.py --source VIDEO.mp4 --enable-twilio
```

### Check if variables are set:
```bash
env | grep TWILIO
env | grep ALERT
```

---

## What Gets Alerted

| Detection | What Happens |
|-----------|-------------|
| Fire 🔥 | Immediate phone call + SMS with evacuation instructions |
| Severe Injury 🚨 | Immediate phone call + SMS to call 911 |
| Distress 😰 | Immediate phone call + SMS for medical response |
| Violence ⚠️ | SMS to security team |
| Fall 🤕 | SMS to medical staff for assessment |

---

## Sample Alert Messages

### Fire (Call + SMS):
**Voice Call:** _"CRITICAL ALERT. Fire has been detected. Confidence 92 percent. Evacuate immediately and contact fire department."_

**SMS:**
```
🔥 VITALSIGHT CRITICAL ALERT 🔥

FIRE DETECTED
Confidence: 92%
Time: 06:28:45
Location: data/demo_clips/fire_sample1.mp4

Immediate response required.
Check surveillance system for details.
```

### Distress (Call + SMS):
**Voice Call:** _"HIGH PRIORITY ALERT. A person in distress has been detected. Confidence 81 percent. Medical assistance required immediately."_

**SMS:**
```
😰 VITALSIGHT HIGH PRIORITY ALERT 😰

PERSON IN DISTRESS
Confidence: 81%
Time: 06:30:12
Location: Camera 3 - Hallway

Immediate response required.
Check surveillance system for details.
```

---

You're all set! When you run VitalSight with `--enable-twilio`, it will automatically send alerts to +19297602752 whenever it detects an emergency situation! 🚀📱
