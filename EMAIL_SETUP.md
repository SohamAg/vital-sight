# 📧 Email Notifications Setup - SendGrid

VitalSight sends automated email alerts for **ALL** detected incidents using **SendGrid**, with beautifully formatted HTML emails containing the full Gemini AI report and evidence image.

---

## ✅ **Setup Complete!**

Your SendGrid integration is **already configured** and ready to go!

### **What's Configured:**

```python
SendGrid API Key: SG.-ReSb2izRhG7glXd7NgeXA... (Set in webapp.py)
From Email: sohamkagrawal@gmail.com (Verified in SendGrid)
To Email: sohamkagrawal@gmail.com (Alert recipient)
```

---

## 🚀 **How to Test**

Just upload a video through the web interface!

1. **Start the webapp:**
   ```bash
   python webapp.py
   ```

2. **Look for this in the console:**
   ```
   [INFO] SendGrid email notifications enabled - will send to: sohamkagrawal@gmail.com
   [INFO] Email notifications enabled for ALL detections
   ```

3. **Upload a video** at `http://localhost:5000/upload`

4. **Check your email** at `sohamkagrawal@gmail.com` 📬

---

## 📨 **What You'll Receive**

Every detection (Fall, Crowd, Distress, Injury, Fire) triggers an email with:

### **Subject Line:**
`🚨 VitalSight Alert: [Category] - [Priority] Priority`

### **Email Contents:**
- **Priority Badge**: 🟢 LOW | 🟡 MEDIUM | 🟠 HIGH | 🔴 CRITICAL
- **Camera Location**: e.g., "Camera 01 - Loading Dock"
- **Timestamp**: When the incident was detected
- **Confidence Score**: Detection accuracy percentage
- **Evidence Frame**: Embedded image of the exact moment
- **Full AI Report**: Complete Gemini analysis with:
  - Immediate situation description
  - Observable details
  - Assessment
  - Recommended actions
- **Professional HTML formatting** with gradient headers and styled content

---

## 📋 **Email Sending Rules**

| Priority | Category | Email | Voice Call |
|----------|----------|-------|------------|
| 🟢 LOW | Fall | ✅ Yes | ❌ No |
| 🟡 MEDIUM | Crowd/Violence | ✅ Yes | ❌ No |
| 🟠 HIGH | Distress | ✅ Yes | ✅ Yes |
| 🔴 CRITICAL | Severe Injury | ✅ Yes | ✅ Yes |
| 🔴 CRITICAL | Fire | ✅ Yes | ✅ Yes |

**All detections send emails. Only HIGH/CRITICAL also trigger voice calls.**

---

## 🔧 **Troubleshooting**

### **"Email notifications not enabled" in console**
Make sure the SendGrid API key is set in `webapp.py` (it already is!)

### **"SendGrid initialization failed"**
- Check that the API key is correct
- Verify your SendGrid account is active
- Make sure you completed sender verification

### **"Emails not arriving"**
1. **Check Spam folder** - first emails often go to spam
2. **Verify sender in SendGrid**: Go to https://app.sendgrid.com → Settings → Sender Authentication
3. **Check SendGrid Activity**: Go to https://app.sendgrid.com → Activity to see if emails were sent
4. **SendGrid free tier limit**: 100 emails/day - check if you've hit the limit

### **Images not showing in email**
- Images are embedded as inline attachments with Content-ID
- Some email clients may block images by default
- Click "Show images" or "Display images" in your email client

---

## 🎯 **SendGrid Dashboard**

Monitor your emails at: https://app.sendgrid.com

- **Activity**: See all sent emails
- **Statistics**: Delivery rates, opens, clicks
- **Sender Authentication**: Verify additional sender emails
- **API Keys**: Manage your API keys

---

## 📹 **Camera Name Mapping**

Emails show friendly camera names instead of file paths:

| Video File | Email Location |
|-----------|----------------|
| `fire_sample1.mp4` | Camera 01 - Loading Dock |
| `fire_sample2.mp4` | Camera 02 - Main Entrance |
| `fire_sample3.mp4` | Camera 03 - Warehouse |
| `fall_sample1.mp4` | Camera 04 - Hallway A |
| `fall_sample2.mp4` | Camera 05 - Cafeteria |
| `distress_sample1.mp4` | Camera 07 - Office Floor 2 |
| `crowd_sample1.mp4` | Camera 09 - Main Plaza |
| `injury_sample1.mp4` | Camera 12 - Factory Floor |
| *(+ 7 more cameras)* | |

*To add new cameras, edit `CAMERA_MAPPING` in `webapp.py`*

---

## 🔄 **Using Environment Variables (Optional)**

For production, you can set the API key as an environment variable:

**PowerShell:**
```powershell
$env:SENDGRID_API_KEY="SG.your-actual-api-key"
```

**Linux/Mac:**
```bash
export SENDGRID_API_KEY="SG.your-actual-api-key"
```

Then remove the hardcoded key from `webapp.py`.

---

## ✨ **Summary**

✅ **SendGrid is configured and ready**  
✅ **Emails sent for ALL detections**  
✅ **Professional HTML formatting**  
✅ **Evidence images embedded**  
✅ **Camera names instead of file paths**  
✅ **100 emails/day free tier**

**Just start the webapp and upload a video to test!** 🚀

---

## 📞 **Voice Calls Too!**

For HIGH/CRITICAL priorities, VitalSight also makes voice calls using:
- **Twilio**: Phone call delivery
- **ElevenLabs**: AI-generated voice (Rachel voice, professional tone)

Voice calls include a 2-3 sentence summary of the incident for immediate response.

---

**Everything is ready to go! Start VitalSight and try uploading a video!** 🎉
