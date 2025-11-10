"""
Notification Service - Voice call alerts using Twilio + ElevenLabs + Email via SendGrid
"""
import os
import tempfile
from pathlib import Path
from twilio.rest import Client
from elevenlabs.client import ElevenLabs
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition, ContentId
import base64


class NotificationService:
    """Handles emergency voice call notifications via Twilio + ElevenLabs"""
    
    def __init__(self, 
                 twilio_account_sid=None,
                 twilio_auth_token=None,
                 twilio_phone_number=None,
                 alert_phone_number=None,
                 elevenlabs_api_key=None,
                 alert_email=None,
                 sendgrid_api_key=None,
                 from_email=None):
        """
        Initialize notification service.
        
        Args:
            twilio_account_sid: Twilio Account SID
            twilio_auth_token: Twilio Auth Token
            twilio_phone_number: Your Twilio phone number (e.g., '+18663508040')
            alert_phone_number: Phone number to call for alerts (e.g., '+19297602752')
            elevenlabs_api_key: ElevenLabs API key for text-to-speech
            alert_email: Email address to send alerts to
            sendgrid_api_key: SendGrid API key for email
            from_email: Sender email address (verified in SendGrid)
        """
        self.enabled = False
        self.alert_phone = alert_phone_number
        self.alert_email = alert_email
        self.from_email = from_email or alert_email  # Default to alert_email if not specified
        self.email_enabled = False
        
        # Initialize Twilio (optional - for voice calls)
        self.twilio_enabled = False
        if twilio_account_sid and twilio_auth_token and twilio_phone_number:
            try:
                self.twilio_client = Client(twilio_account_sid, twilio_auth_token)
                self.twilio_phone = twilio_phone_number
                self.twilio_enabled = True
                print("[INFO] Twilio voice calls enabled")
            except Exception as e:
                print(f"[WARNING] Twilio initialization failed: {e}")
        else:
            print("[WARNING] Twilio credentials not provided, voice calls disabled")
        
        # Initialize ElevenLabs
        if elevenlabs_api_key:
            try:
                self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
                print("[INFO] ElevenLabs text-to-speech enabled")
            except Exception as e:
                print(f"[WARNING] ElevenLabs initialization failed: {e}")
                # Don't return, email can still work
        else:
            print("[WARNING] ElevenLabs API key not provided, TTS disabled")
        
        # Initialize SendGrid Email
        if alert_email and sendgrid_api_key:
            try:
                self.sendgrid_client = SendGridAPIClient(sendgrid_api_key)
                self.email_enabled = True
                print(f"[INFO] SendGrid email notifications enabled - will send to: {alert_email}")
            except Exception as e:
                print(f"[WARNING] SendGrid initialization failed: {e}")
        else:
            print("[WARNING] SendGrid API key or alert email not configured")
        
        # Priority configuration
        self.priority_config = {
            "fall": {"priority": "LOW", "needs_call": False},
            "violence_panic": {"priority": "MEDIUM", "needs_call": False},
            "distress": {"priority": "HIGH", "needs_call": True},
            "severe_injury": {"priority": "CRITICAL", "needs_call": True},
            "fire": {"priority": "CRITICAL", "needs_call": True}
        }
        
        # Set enabled if we have either email or voice calls working
        if self.email_enabled or (self.twilio_enabled and self.alert_phone):
            self.enabled = True
            if self.alert_phone and self.twilio_enabled:
                print(f"[INFO] Voice alerts will be sent to: {self.alert_phone}")
        else:
            print("[WARNING] No notification services configured")
    
    def send_alert(self, category, report_text, source_path=None, frame_path=None, confidence=None, timestamp=None):
        """
        Send alert notifications (email always, voice call for HIGH/CRITICAL).
        
        Args:
            category: Detection category (fall, fire, etc.)
            report_text: Full Gemini report text
            source_path: Source video/camera identifier
            frame_path: Path to evidence frame image
            confidence: Detection confidence score
            timestamp: Detection timestamp
        
        Returns:
            dict with status of email and call
        """
        config = self.priority_config.get(category, {"priority": "MEDIUM", "needs_call": False})
        results = {"email": False, "call": False}
        
        # Extract brief summary (for voice call)
        brief_summary = self._extract_brief_summary(report_text)
        
        # ALWAYS send email for ALL detections
        if self.email_enabled and self.alert_email:
            print(f"[EMAIL] Sending alert email for {category}...")
            results["email"] = self._send_email(
                category=category,
                priority=config['priority'],
                report_text=report_text,
                source_path=source_path,
                frame_path=frame_path,
                confidence=confidence,
                timestamp=timestamp
            )
        else:
            print("[WARNING] Email not configured, skipping email notification")
        
        # Make voice call ONLY for HIGH/CRITICAL priority (if Twilio is configured)
        if config['needs_call'] and self.twilio_enabled and self.alert_phone:
            print(f"\n[CRITICAL ALERT] Initiating voice call for {category}...")
            call_result = self._make_voice_call(
                category=category,
                priority=config['priority'],
                brief_summary=brief_summary,
                source_path=source_path
            )
            results["call"] = call_result.get("call", False)
            if "call_sid" in call_result:
                results["call_sid"] = call_result["call_sid"]
        elif config['needs_call'] and not self.twilio_enabled:
            print(f"[INFO] Voice call needed for {category} but Twilio not configured")
        else:
            print(f"[INFO] {category} is {config['priority']} priority - no voice call needed")
        
        return results
    
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
    
    def _send_email(self, category, priority, report_text, source_path, frame_path=None, confidence=None, timestamp=None):
        """Send email notification with detailed report and evidence image using SendGrid"""
        try:
            # Priority emoji based on level
            priority_emoji = {
                "LOW": "🟢",
                "MEDIUM": "🟡",
                "HIGH": "🟠",
                "CRITICAL": "🔴"
            }.get(priority, "🔵")
            
            # Create HTML body
            html_content = f"""
            <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                        .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                        .header {{ background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); 
                                  color: white; padding: 20px; border-radius: 10px 10px 0 0; }}
                        .alert-badge {{ background: rgba(255,255,255,0.2); padding: 10px 20px; 
                                       border-radius: 20px; display: inline-block; margin-top: 10px; }}
                        .content {{ background: #f9fafb; padding: 20px; border: 2px solid #dc2626; 
                                   border-radius: 0 0 10px 10px; }}
                        .info-grid {{ display: grid; grid-template-columns: 150px 1fr; gap: 10px; 
                                     margin: 20px 0; }}
                        .info-label {{ font-weight: bold; color: #374151; }}
                        .info-value {{ color: #1f2937; }}
                        .report {{ background: white; padding: 20px; border-radius: 5px; 
                                  margin: 20px 0; white-space: pre-wrap; 
                                  border-left: 4px solid #dc2626; }}
                        .footer {{ text-align: center; color: #6b7280; font-size: 12px; 
                                  margin-top: 20px; padding-top: 20px; border-top: 1px solid #e5e7eb; }}
                        .image-container {{ text-align: center; margin: 20px 0; }}
                        .evidence-image {{ max-width: 100%; height: auto; border-radius: 5px; 
                                          border: 2px solid #dc2626; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1 style="margin: 0; font-size: 28px;">⚠️ EMERGENCY ALERT</h1>
                            <div class="alert-badge">
                                <strong style="font-size: 20px;">{priority_emoji} {priority} PRIORITY</strong>
                            </div>
                        </div>
                        
                        <div class="content">
                            <div class="info-grid">
                                <div class="info-label">Alert Type:</div>
                                <div class="info-value"><strong>{category.replace('_', ' ').title()}</strong></div>
                                
                                <div class="info-label">Location:</div>
                                <div class="info-value">{source_path or 'Unknown Camera'}</div>
                                
                                <div class="info-label">Timestamp:</div>
                                <div class="info-value">{timestamp or 'Just now'}</div>
                                
                                <div class="info-label">Confidence:</div>
                                <div class="info-value">{f"{confidence:.1%}" if confidence else "N/A"}</div>
                            </div>
                            
                            {"<div class='image-container'><img src='cid:evidence_frame' class='evidence-image' alt='Evidence Frame'/></div>" if frame_path else ""}
                            
                            <h2 style="color: #dc2626; margin-top: 20px;">📋 Detailed AI Analysis Report</h2>
                            <div class="report">{report_text}</div>
                            
                            <div class="footer">
                                <p><strong>VitalSight Context-Aware Environment System</strong></p>
                                <p>This is an automated alert. Please respond according to your emergency protocols.</p>
                                <p>Generated at {timestamp or 'now'}</p>
                            </div>
                        </div>
                    </div>
                </body>
            </html>
            """
            
            # Create message
            message = Mail(
                from_email=self.from_email,
                to_emails=self.alert_email,
                subject=f"🚨 VitalSight Alert: {category.replace('_', ' ').title()} - {priority} Priority",
                html_content=html_content
            )
            
            # Attach evidence image if available
            if frame_path and Path(frame_path).exists():
                try:
                    with open(frame_path, 'rb') as f:
                        img_data = f.read()
                        encoded = base64.b64encode(img_data).decode()
                        
                        attachment = Attachment()
                        attachment.file_content = FileContent(encoded)
                        attachment.file_name = FileName(Path(frame_path).name)
                        attachment.file_type = FileType('image/jpeg')
                        attachment.disposition = Disposition('inline')
                        attachment.content_id = ContentId('evidence_frame')
                        
                        message.add_attachment(attachment)
                except Exception as img_err:
                    print(f"[WARNING] Could not attach image: {img_err}")
            
            # Send email via SendGrid
            response = self.sendgrid_client.send(message)
            
            if response.status_code in [200, 202]:
                print(f"[✓] Email sent via SendGrid to {self.alert_email}")
                return True
            else:
                print(f"[ERROR] SendGrid returned status {response.status_code}")
                return False
            
        except Exception as e:
            print(f"[ERROR] SendGrid email send failed: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
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

