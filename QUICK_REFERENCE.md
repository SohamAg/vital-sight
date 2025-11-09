# VitalSight - Quick Reference Card

## 🚀 Run Commands

### Single Video (Watch with Display)
```bash
python main.py --source data/demo_clips/fall_sample1.mp4 --gemini-key AIzaSyDPE3QNZqVino7KJvFDeZ_nfYcQ627FcMo
```

### Batch Process All Videos (No Display)
```bash
python batch_process.py
```

### Test Script (Interactive)
```bash
python test_gemini.py
```

---

## 📁 Output Locations

| Type | Location | Content |
|------|----------|---------|
| **Processed Videos** | `data/processed/` | Annotated MP4s with YOLO detections |
| **Reports** | `data/demo_reports/` | Gemini text reports |
| **Frames** | `data/demo_reports/` | JPG captures at detection moment |

---

## 🎯 Severity Levels

### 🔴 CRITICAL (Phone + SMS + Email)
- **Fire** - Rapid escalation, immediate response needed
- **Distress** - Cardiac arrest risk within minutes
- **Severe Injury** - Death/disability prevention

### 🟡 MEDIUM (SMS + Email)
- **Fall** - Medical assessment needed, lower immediate threat
- **Violence/Panic** - Security response, coordinated intervention

---

## 🔧 Key Features

✅ **Gemini 2.0 Flash** - Fast, accurate vision analysis  
✅ **Async Processing** - Video never pauses  
✅ **Auto-Override** - No duplicate reports  
✅ **Batch Pipeline** - Process all videos at once  
✅ **Present Tense** - Reports describe live situation  
✅ **Phone/Email Ready** - Can be read aloud or sent directly  

---

## 📝 API Key

Your key: `AIzaSyDPE3QNZqVino7KJvFDeZ_nfYcQ627FcMo`

Set as environment variable:
```bash
# PowerShell
$env:GEMINI_API_KEY="AIzaSyDPE3QNZqVino7KJvFDeZ_nfYcQ627FcMo"
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No reports generated | Check API key, internet connection |
| Video pauses | Make sure async update is applied |
| Batch not finding videos | Check `data/demo_clips/` path |
| Reports keep multiplying | They now auto-override ✓ |

---

## 📄 Documentation

- `UPDATES_SUMMARY.md` - What changed
- `BATCH_PROCESSING_GUIDE.md` - Full batch pipeline docs
- `GEMINI_USAGE.md` - Original Gemini integration guide
- `ASYNC_UPDATE.md` - Async implementation details

---

**Quick Start: Just run `python batch_process.py` and check outputs!** 🎉

