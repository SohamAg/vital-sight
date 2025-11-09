"""
Notification Service - Voice call alerts using Twilio + ElevenLabs
"""
import os
import tempfile
from pathlib import Path
from twilio.rest import Client
from elevenlabs.client import ElevenLabs


class NotificationService:
    """Handles emergency voice call notifications via Twilio + ElevenLabs"""
    
    def __init__(self, 
                 twilio_account_sid=None,
                 twilio_auth_token=None,
                 twilio_phone_number=None,
                 alert_phone_number=None,
                 elevenlabs_api_key=None):
        """
        Initialize notification service.
        
        Args:
            twilio_account_sid: Twilio Account SID
            twilio_auth_token: Twilio Auth Token
            twilio_phone_number: Your Twilio phone number (e.g., '+18663508040')
            alert_phone_number: Phone number to call for alerts (e.g., '+19297602752')
            elevenlabs_api_key: ElevenLabs API key for text-to-speech
        """
        self.enabled = False
        self.alert_phone = alert_phone_number
        
        # Initialize Twilio
        if twilio_account_sid and twilio_auth_token and twilio_phone_number:
            try:
                self.twilio_client = Client(twilio_account_sid, twilio_auth_token)
                self.twilio_phone = twilio_phone_number
                print("[INFO] Twilio voice calls enabled")
            except Exception as e:
                print(f"[WARNING] Twilio initialization failed: {e}")
                return
        else:
            print("[WARNING] Twilio credentials not provided, voice calls disabled")
            return
        
        # Initialize ElevenLabs
        if elevenlabs_api_key:
            try:
                self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
                print("[INFO] ElevenLabs text-to-speech enabled")
            except Exception as e:
                print(f"[WARNING] ElevenLabs initialization failed: {e}")
                return
        else:
            print("[WARNING] ElevenLabs API key not provided, TTS disabled")
            return
        
        # Priority configuration
        self.priority_config = {
            "fall": {"priority": "LOW", "needs_call": False},
            "violence_panic": {"priority": "MEDIUM", "needs_call": False},
            "distress": {"priority": "HIGH", "needs_call": True},
            "severe_injury": {"priority": "CRITICAL", "needs_call": True},
            "fire": {"priority": "CRITICAL", "needs_call": True}
        }
        
        if self.alert_phone:
            self.enabled = True
            print(f"[INFO] Voice alerts will be sent to: {self.alert_phone}")
        else:
            print("[WARNING] No alert phone number configured")
    
    def send_alert(self, category, report_text, source_path=None):
        """
        Send voice call alert for CRITICAL situations.
        
        Args:
            category: Detection category (fall, fire, etc.)
            report_text: Full Gemini report text
            source_path: Source video/camera identifier
        
        Returns:
            dict with status of the call
        """
        if not self.enabled:
            return {"call": False, "reason": "Service not enabled"}
        
        config = self.priority_config.get(category, {"priority": "MEDIUM", "needs_call": False})
        
        # Only make calls for HIGH/CRITICAL priority
        if not config['needs_call']:
            print(f"[INFO] {category} is {config['priority']} priority - no voice call needed")
            return {"call": False, "reason": "Priority level does not require call"}
        
        print(f"\n[CRITICAL ALERT] Initiating voice call for {category}...")
        
        # Extract brief summary (first 2-3 lines for voice call)
        brief_summary = self._extract_brief_summary(report_text)
        
        # Make voice call
        result = self._make_voice_call(
            category=category,
            priority=config['priority'],
            brief_summary=brief_summary,
            source_path=source_path
        )
        
        return result
    
    def _extract_brief_summary(self, report_text):
        """Extract first 2-3 sentences for voice call"""
        lines = [line.strip() for line in report_text.split('\n') if line.strip()]
        
        # Find the "IMMEDIATE SITUATION" section
        brief_lines = []
        in_immediate = False
        for line in lines:
            if "IMMEDIATE SITUATION" in line.upper():
                in_immediate = True
                continue
            if in_immediate:
                # Skip formatting lines
                if line and not line.startswith("=") and not line.startswith("**") and not line.startswith("#"):
                    brief_lines.append(line)
                    if len(brief_lines) >= 3:  # Get first 3 sentences
                        break
                # Stop at next section
                if any(section in line.upper() for section in ["OBSERVABLE DETAILS", "ASSESSMENT", "RECOMMENDED"]):
                    break
        
        if brief_lines:
            summary = ' '.join(brief_lines[:3])
        else:
            # Fallback: use first 3 non-empty lines
            summary = ' '.join(lines[:3]) if lines else "Emergency situation detected."
        
        # Limit length to avoid very long messages
        if len(summary) > 500:
            summary = summary[:497] + "..."
        
        return summary
    
    def _make_voice_call(self, category, priority, brief_summary, source_path):
        """Make voice call using ElevenLabs TTS + Twilio"""
        try:
            # Construct the alert message
            location = source_path if source_path else "live camera feed"
            alert_message = f"""
            This is an urgent alert from VitalSight emergency detection system.
            
            Priority level: {priority}.
            
            Category: {category.replace('_', ' ')}.
            
            Location: {location}.
            
            {brief_summary}
            
            Please respond immediately. This is an automated emergency alert from VitalSight.
            """
            
            print(f"[TTS] Generating audio with ElevenLabs...")
            print(f"[TTS] Message: {alert_message[:100]}...")
            
            # Generate speech with ElevenLabs
            # Using "Rachel" voice - professional, clear American English  
            # Using turbo v2.5 model (available on free tier)
            audio = self.elevenlabs_client.text_to_speech.convert(
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice ID
                text=alert_message,
                model_id="eleven_turbo_v2_5"
            )
            
            # Save to temporary file
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_audio_path = temp_audio.name
            
            # Write audio content
            with open(temp_audio_path, 'wb') as f:
                for chunk in audio:
                    f.write(chunk)
            
            print(f"[TTS] Audio saved to: {temp_audio_path}")
            
            # Upload audio to a publicly accessible URL (Twilio needs a URL)
            # We'll use Twilio's built-in hosting by uploading as Media
            print(f"[TWILIO] Initiating call to {self.alert_phone}...")
            
            # For Twilio, we need to host the audio file somewhere accessible
            # Since we don't have a public server, we'll use TwiML Say instead
            # This is a fallback that uses Twilio's built-in TTS
            twiml_message = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="en-US">
        This is an urgent alert from VitalSight emergency detection system.
        Priority level: {priority}.
        Category: {category.replace('_', ' ')}.
        Location: {location}.
        {brief_summary}
        Please respond immediately.
    </Say>
    <Pause length="2"/>
    <Say voice="alice">
        Repeating: {priority} priority {category.replace('_', ' ')} detected. Immediate response required.
    </Say>
</Response>"""
            
            # Make the call
            call = self.twilio_client.calls.create(
                twiml=twiml_message,
                to=self.alert_phone,
                from_=self.twilio_phone
            )
            
            print(f"[✓] Voice call initiated - SID: {call.sid}")
            print(f"[✓] Status: {call.status}")
            
            # Clean up temp file
            try:
                os.unlink(temp_audio_path)
            except:
                pass
            
            return {
                "call": True,
                "call_sid": call.sid,
                "status": call.status,
                "to": self.alert_phone,
                "category": category,
                "priority": priority
            }
            
        except Exception as e:
            print(f"[ERROR] Voice call failed: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                "call": False,
                "error": str(e)
            }

