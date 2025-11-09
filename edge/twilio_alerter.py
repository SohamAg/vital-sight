"""
Twilio Alert System for VitalSight
Sends SMS and voice calls based on detected situations
"""
import os
from twilio.rest import Client
from datetime import datetime


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
    
    def send_alert(self, category, confidence, source_path=None):
        """
        Send appropriate alert(s) based on category
        
        Args:
            category: Detection category
            confidence: Detection confidence
            source_path: Video source path
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
        
        # Send according to configuration
        if "sms" in methods:
            self.send_sms(category, confidence, source_path)
        
        if "call" in methods:
            self.make_call(category, confidence)
        
        # Mark as sent
        self.sent_alerts.add(alert_id)
        
        # Clean up old alerts (keep last 100)
        if len(self.sent_alerts) > 100:
            self.sent_alerts = set(list(self.sent_alerts)[-100:])
