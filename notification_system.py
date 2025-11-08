"""
Notification System Module
Handles alerts and notifications when distress is detected
"""

import os
import time
import config
from datetime import datetime


class NotificationSystem:
    """Manages notifications for distress alerts"""
    
    def __init__(self):
        self.last_alert_time = {}  # Track last alert time per person to avoid spam
        self.alert_cooldown = 10   # Seconds between alerts for same person
        self.alert_count = 0
        
        # Initialize alert directory
        if config.SAVE_ALERTS:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        # Try to import optional notification libraries
        self.plyer_available = False
        self.email_available = False
        
        try:
            from plyer import notification
            self.notification = notification
            self.plyer_available = True
        except ImportError:
            if config.DEBUG_MODE:
                print("Warning: plyer not available. Desktop notifications disabled.")
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            self.smtplib = smtplib
            self.MIMEText = MIMEText
            self.MIMEMultipart = MIMEMultipart
            self.email_available = True
        except ImportError:
            if config.DEBUG_MODE:
                print("Warning: email libraries not available. Email alerts disabled.")
    
    def should_send_alert(self, person_id):
        """Check if enough time has passed since last alert for this person"""
        current_time = time.time()
        
        if person_id not in self.last_alert_time:
            self.last_alert_time[person_id] = current_time
            return True
        
        time_since_last = current_time - self.last_alert_time[person_id]
        
        if time_since_last > self.alert_cooldown:
            self.last_alert_time[person_id] = current_time
            return True
        
        return False
    
    def send_desktop_notification(self, distress_info, person_id=0):
        """Send desktop notification"""
        if not config.ENABLE_DESKTOP_NOTIFICATIONS or not self.plyer_available:
            return
        
        try:
            title = "⚠️ DISTRESS DETECTED!"
            message = (
                f"Person {person_id}: {distress_info['distress_type']}\n"
                f"Confidence: {distress_info['confidence']:.1%}\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            self.notification.notify(
                title=title,
                message=message,
                app_name="Vital Sight",
                timeout=10
            )
            
            if config.DEBUG_MODE:
                print(f"✓ Desktop notification sent for person {person_id}")
                
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"Error sending desktop notification: {e}")
    
    def play_alert_sound(self):
        """Play alert sound (system beep)"""
        if not config.ENABLE_SOUND_ALERTS:
            return
        
        try:
            # Use system beep
            print('\a')  # Bell character
            
            if config.DEBUG_MODE:
                print("✓ Alert sound played")
                
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"Error playing alert sound: {e}")
    
    def send_email_alert(self, distress_info, person_id=0, image_path=None):
        """Send email alert"""
        if not config.ENABLE_EMAIL_ALERTS or not self.email_available:
            return
        
        try:
            msg = self.MIMEMultipart()
            msg['From'] = config.EMAIL_SENDER
            msg['To'] = ', '.join(config.EMAIL_RECIPIENTS)
            msg['Subject'] = f"⚠️ DISTRESS ALERT - Person {person_id}"
            
            body = f"""
DISTRESS DETECTED

Person ID: {person_id}
Distress Type: {distress_info['distress_type']}
Confidence: {distress_info['confidence']:.1%}
Body Angle: {distress_info.get('body_angle', 'N/A'):.1f}°
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Location: {image_path if image_path else 'N/A'}

This is an automated alert from Vital Sight CCTV Monitoring System.
"""
            
            msg.attach(self.MIMEText(body, 'plain'))
            
            # Send email
            server = self.smtplib.SMTP(config.EMAIL_SMTP_SERVER, config.EMAIL_SMTP_PORT)
            server.starttls()
            server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            
            if config.DEBUG_MODE:
                print(f"✓ Email alert sent to {len(config.EMAIL_RECIPIENTS)} recipient(s)")
                
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"Error sending email alert: {e}")
    
    def save_alert_image(self, frame, distress_info, person_id=0):
        """Save frame with detected distress"""
        if not config.SAVE_ALERTS:
            return None
        
        try:
            import cv2
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"alert_{person_id}_{timestamp}_{self.alert_count}.jpg"
            filepath = os.path.join(config.OUTPUT_DIR, filename)
            
            # Add text overlay to image
            frame_copy = frame.copy()
            text = f"{distress_info['distress_type']} - {distress_info['confidence']:.1%}"
            cv2.putText(frame_copy, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imwrite(filepath, frame_copy)
            self.alert_count += 1
            
            if config.DEBUG_MODE:
                print(f"✓ Alert image saved: {filepath}")
            
            return filepath
            
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"Error saving alert image: {e}")
            return None
    
    def send_alert(self, frame, distress_info, person_id=0):
        """
        Send all configured alerts
        """
        # Check cooldown
        if not self.should_send_alert(person_id):
            return
        
        # Save alert image first
        image_path = self.save_alert_image(frame, distress_info, person_id)
        
        # Console alert (always enabled)
        self.log_alert(distress_info, person_id)
        
        # Desktop notification
        self.send_desktop_notification(distress_info, person_id)
        
        # Sound alert
        self.play_alert_sound()
        
        # Email alert
        if config.ENABLE_EMAIL_ALERTS:
            self.send_email_alert(distress_info, person_id, image_path)
        
        # SMS alert (placeholder - requires API integration)
        if config.ENABLE_SMS_ALERTS:
            self.send_sms_alert(distress_info, person_id)
    
    def log_alert(self, distress_info, person_id=0):
        """Log alert to console"""
        print("\n" + "="*60)
        print("⚠️  DISTRESS ALERT TRIGGERED")
        print("="*60)
        print(f"Person ID: {person_id}")
        print(f"Type: {distress_info['distress_type']}")
        print(f"Confidence: {distress_info['confidence']:.1%}")
        print(f"Body Angle: {distress_info.get('body_angle', 'N/A'):.1f}°")
        print(f"Vertical Position: {distress_info.get('vertical_position', 'N/A'):.2f}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
    
    def send_sms_alert(self, distress_info, person_id=0):
        """Send SMS alert (placeholder for API integration)"""
        # This is a placeholder - implement with your preferred SMS API
        # Examples: Twilio, Amazon SNS, etc.
        if config.DEBUG_MODE:
            print(f"SMS alert would be sent to: {config.SMS_RECIPIENTS}")
            print(f"Message: Person {person_id} - {distress_info['distress_type']}")
