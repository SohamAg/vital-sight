"""
Twilio Alert System for VitalSight
Sends SMS and voice calls based on detected situations
"""
import os
from twilio.rest import Client
from datetime import datetime
from pathlib import Path

# ElevenLabs imports (handle both old and new API versions)
try:
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_NEW_API = True
except ImportError:
    try:
        from elevenlabs import generate, save, set_api_key
        ELEVENLABS_NEW_API = False
    except ImportError:
        ELEVENLABS_NEW_API = None


class TwilioAlerter:
    """
    Handles real-time SMS and voice call alerts via Twilio
    """
    
    def __init__(self, account_sid=None, auth_token=None, from_number=None, to_number=None):
        """
        Initialize Twilio client
        
        Args:
            account_sid: Twilio Account SID (or use TWILIO_ACCOUNT_SID env var)
            auth_token: Twilio Auth Token (or use TWILIO_AUTH_TOKEN env var)
            from_number: Twilio phone number (or use TWILIO_PHONE_NUMBER env var)
            to_number: Destination phone number (or use ALERT_PHONE_NUMBER env var)
        """
        # Get credentials from arguments or environment variables
        self.account_sid = account_sid or os.environ.get('TWILIO_ACCOUNT_SID')
        self.auth_token = auth_token or os.environ.get('TWILIO_AUTH_TOKEN')
        self.from_number = from_number or os.environ.get('TWILIO_PHONE_NUMBER')
        self.to_number = to_number or os.environ.get('ALERT_PHONE_NUMBER')
        
        # Validate credentials
        if not all([self.account_sid, self.auth_token, self.from_number, self.to_number]):
            raise ValueError("Missing Twilio credentials. Provide them as arguments or set environment variables.")
        
        # Initialize Twilio client
        self.client = Client(self.account_sid, self.auth_token)
        
        # Track sent alerts to avoid duplicates
        self.sent_alerts = set()
        
        # Alert configuration by category
        self.alert_config = {
            "fire": {
                "methods": ["call", "sms"],
                "priority": "CRITICAL",
                "emoji": "🔥"
            },
            "fall": {
                "methods": ["sms"],
                "priority": "LOW",
                "emoji": "🤕"
            },
            "distress": {
                "methods": ["call", "sms"],
                "priority": "HIGH",
                "emoji": "😰"
            },
            "violence_panic": {
                "methods": ["sms"],
                "priority": "MEDIUM",
                "emoji": "⚠️"
            },
            "severe_injury": {
                "methods": ["call", "sms"],
                "priority": "CRITICAL",
                "emoji": "🚨"
            }
        }
        
        # ElevenLabs API key (optional)
        self.elevenlabs_key = os.environ.get('ELEVENLABS_API_KEY')
        self.elevenlabs_client = None
        if self.elevenlabs_key:
            if ELEVENLABS_NEW_API:
                self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_key)
                print(f"[TWILIO] ElevenLabs voice synthesis enabled (new API)")
            elif ELEVENLABS_NEW_API is False:
                set_api_key(self.elevenlabs_key)
                print(f"[TWILIO] ElevenLabs voice synthesis enabled (legacy API)")
            else:
                print(f"[TWILIO] ElevenLabs not installed, voice synthesis disabled")
        
        # Pending voice calls (waiting for Gemini report)
        self.pending_calls = {}  # {category: {"situation_text": None, "confidence": 0, "source": None}}
        
        print(f"[TWILIO] Alert system initialized")
        print(f"[TWILIO] Alerts will be sent to: {self.to_number}")
    
    def _generate_sms_message(self, category, confidence, source_path=None):
        """
        Generate SMS message text
        
        Args:
            category: Detection category
            confidence: Detection confidence
            source_path: Video source path
            
        Returns:
            SMS message text
        """
        config = self.alert_config.get(category, {})
        emoji = config.get("emoji", "⚠️")
        priority = config.get("priority", "ALERT")
        
        category_names = {
            "fall": "FALL DETECTED",
            "fire": "FIRE DETECTED",
            "distress": "PERSON IN DISTRESS",
            "violence_panic": "VIOLENCE/PANIC DETECTED",
            "severe_injury": "SEVERE INJURY DETECTED"
        }
        
        alert_title = category_names.get(category, "ALERT")
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        location = ""
        if source_path:
            location = f"\nLocation: {source_path}"
        
        message = f"""
{emoji} VITALSIGHT {priority} ALERT {emoji}

{alert_title}
Confidence: {confidence:.0%}
Time: {timestamp}{location}

Immediate response required.
Check surveillance system for details.
        """.strip()
        
        return message
    
    def _generate_voice_message(self, category, confidence):
        """
        Generate voice call message (TwiML)
        
        Args:
            category: Detection category
            confidence: Detection confidence
            
        Returns:
            Voice message text
        """
        category_messages = {
            "fire": "CRITICAL ALERT. Fire has been detected. Confidence {confidence} percent. Evacuate immediately and contact fire department.",
            "distress": "HIGH PRIORITY ALERT. A person in distress has been detected. Confidence {confidence} percent. Medical assistance required immediately.",
            "severe_injury": "CRITICAL ALERT. Severe injury has been detected. Confidence {confidence} percent. Emergency medical services needed immediately."
        }
        
        message = category_messages.get(
            category,
            "ALERT. {category} has been detected. Confidence {confidence} percent. Please check surveillance system."
        )
        
        # Format confidence as integer percentage
        message = message.format(
            category=category.replace("_", " "),
            confidence=int(confidence * 100)
        )
        
        return message
    
    def send_sms(self, category, confidence, source_path=None):
        """
        Send SMS alert
        
        Args:
            category: Detection category
            confidence: Detection confidence
            source_path: Video source path
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            message = self._generate_sms_message(category, confidence, source_path)
            
            result = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.to_number
            )
            
            print(f"[TWILIO] ✓ SMS sent for {category} (SID: {result.sid})")
            return True
            
        except Exception as e:
            print(f"[TWILIO] ✗ Failed to send SMS: {e}")
            return False
    
    def make_call(self, category, confidence):
        """
        Make voice call alert
        
        Args:
            category: Detection category
            confidence: Detection confidence
            
        Returns:
            True if call initiated successfully, False otherwise
        """
        try:
            message = self._generate_voice_message(category, confidence)
            
            # Create TwiML for voice message
            twiml = f"""
            <Response>
                <Say voice="alice" language="en-US" loop="2">
                    {message}
                </Say>
                <Pause length="2"/>
                <Say voice="alice">
                    Press any key to acknowledge this alert.
                </Say>
            </Response>
            """.strip()
            
            result = self.client.calls.create(
                twiml=twiml,
                from_=self.from_number,
                to=self.to_number
            )
            
            print(f"[TWILIO] ✓ Call initiated for {category} (SID: {result.sid})")
            return True
            
        except Exception as e:
            print(f"[TWILIO] ✗ Failed to make call: {e}")
            return False
    
    def send_alert(self, category, confidence, source_path=None, gemini_enabled=False):
        """
        Send appropriate alert(s) based on category
        
        Args:
            category: Detection category
            confidence: Detection confidence
            source_path: Video source path
            gemini_enabled: If True, skip immediate call (wait for Gemini callback)
        """
        # Create unique alert ID to prevent duplicates
        alert_id = f"{category}_{source_path}_{int(datetime.now().timestamp() / 60)}"
        
        # Skip if already sent in the last minute
        if alert_id in self.sent_alerts:
            print(f"[TWILIO] Alert for {category} already sent recently, skipping...")
            return
        
        # Get alert configuration
        config = self.alert_config.get(category, {"methods": ["sms"], "priority": "ALERT"})
        methods = config["methods"]
        
        print(f"\n[TWILIO] 🚨 Sending {config['priority']} alert for {category}")
        
        # Send SMS immediately (always instant)
        if "sms" in methods:
            self.send_sms(category, confidence, source_path)
        
        # Make call immediately ONLY if Gemini is not enabled
        # If Gemini is enabled, the call will be made by on_situation_report_ready()
        if "call" in methods and not gemini_enabled:
            self.make_call(category, confidence)
        elif "call" in methods and gemini_enabled:
            print(f"[TWILIO] Call will be made after Gemini report is ready (with ElevenLabs voice)")
        
        # Mark as sent
        self.sent_alerts.add(alert_id)
        
        # Clean up old alerts (keep last 100)
        if len(self.sent_alerts) > 100:
            self.sent_alerts = set(list(self.sent_alerts)[-100:])
    
    def generate_elevenlabs_audio(self, text, category):
        """
        Generate audio using ElevenLabs from situation report text
        
        Args:
            text: The situation report text to convert to speech
            category: Detection category
            
        Returns:
            Path to generated audio file, or None if failed
        """
        if not self.elevenlabs_key:
            print("[TWILIO] ElevenLabs not configured, skipping voice generation")
            return None
        
        try:
            print(f"[ELEVENLABS] Generating voice for {category} report...")
            
            # Save to temporary file
            audio_dir = Path("data/temp_audio")
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / f"{category}_{int(datetime.now().timestamp())}.mp3"
            
            # Generate audio using appropriate API version
            if ELEVENLABS_NEW_API and self.elevenlabs_client:
                # New API (v1.0+) - Use better model and voice settings for natural speech
                audio_generator = self.elevenlabs_client.generate(
                    text=text,
                    voice="Rachel",  # Can also try: "Bella", "Elli", "Charlotte"
                    model="eleven_turbo_v2",  # Faster and more natural than v1
                    voice_settings={
                        "stability": 0.5,        # Lower = more expressive (0.3-0.6 for natural)
                        "similarity_boost": 0.75,  # Higher = more like original voice
                        "style": 0.5,            # Exaggeration of speaking style
                        "use_speaker_boost": True  # Enhances clarity
                    }
                )
                
                # Write audio to file
                with open(audio_path, 'wb') as f:
                    for chunk in audio_generator:
                        if chunk:
                            f.write(chunk)
            else:
                # Legacy API (v0.x)
                audio = generate(
                    text=text,
                    voice="Rachel",
                    model="eleven_monolingual_v1"
                )
                save(audio, str(audio_path))
            
            print(f"[ELEVENLABS] ✓ Voice generated: {audio_path}")
            return audio_path
            
        except Exception as e:
            print(f"[ELEVENLABS] ✗ Failed to generate voice: {e}")
            return None
    
    def make_call_with_audio(self, category, confidence, audio_url):
        """
        Make voice call with custom audio URL
        
        Args:
            category: Detection category
            confidence: Detection confidence
            audio_url: URL to audio file to play
            
        Returns:
            True if call initiated successfully, False otherwise
        """
        try:
            # Create TwiML to play the audio file
            twiml = f"""
            <Response>
                <Play>{audio_url}</Play>
                <Pause length="2"/>
                <Say voice="alice">
                    Press any key to acknowledge this alert.
                </Say>
            </Response>
            """.strip()
            
            result = self.client.calls.create(
                twiml=twiml,
                from_=self.from_number,
                to=self.to_number
            )
            
            print(f"[TWILIO] ✓ Call initiated for {category} with custom audio (SID: {result.sid})")
            return True
            
        except Exception as e:
            print(f"[TWILIO] ✗ Failed to make call with audio: {e}")
            return False
    
    def on_situation_report_ready(self, situation_text, category, confidence, source_path):
        """
        Callback function called when Gemini situation report is ready.
        Generates voice using ElevenLabs and makes the call.
        
        Args:
            situation_text: The generated situation report text
            category: Detection category
            confidence: Detection confidence
            source_path: Video source path
        """
        # Only process if this category requires voice calls
        config = self.alert_config.get(category, {})
        if "call" not in config.get("methods", []):
            return
        
        print(f"\n[TWILIO] Situation report ready for {category}, generating voice call...")
        
        # Check if ElevenLabs is available
        if not self.elevenlabs_key:
            print("[TWILIO] ElevenLabs not configured, using generic TwiML voice")
            self.make_call(category, confidence)
            return
        
        # Generate audio using ElevenLabs
        audio_path = self.generate_elevenlabs_audio(situation_text, category)
        
        if not audio_path:
            print("[TWILIO] Falling back to generic TwiML voice")
            self.make_call(category, confidence)
            return
        
        # For now, we need a publicly accessible URL for the audio
        # TODO: Upload to S3, Firebase, or use ngrok for local testing
        print("[TWILIO] Note: Audio file generated locally at:", audio_path)
        print("[TWILIO] To use ElevenLabs voice, you need to:")
        print("[TWILIO]   1. Upload audio to publicly accessible URL (S3, Firebase, etc.)")
        print("[TWILIO]   2. Pass that URL to make_call_with_audio()")
        print("[TWILIO] For now, using generic TwiML voice...")
        
        # Fallback to generic voice for now
        self.make_call(category, confidence)
