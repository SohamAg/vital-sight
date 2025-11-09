"""
Test script for voice call notifications
"""
import os
from edge.notifications import NotificationService

def test_notification():
    """Test the notification service"""
    
    # Configuration (same as webapp.py)
    config = {
        'twilio_account_sid': os.environ.get('TWILIO_ACCOUNT_SID', 'AC9482d7139f9d9056cbdf9159f02052db'),
        'twilio_auth_token': os.environ.get('TWILIO_AUTH_TOKEN', '5c9fd678f6689941e0cebcae6cebac35'),
        'twilio_phone_number': os.environ.get('TWILIO_PHONE_NUMBER', '+18663508040'),
        'alert_phone_number': os.environ.get('ALERT_PHONE_NUMBER', '+19297602752'),
        'elevenlabs_api_key': os.environ.get('ELEVENLABS_API_KEY', 'sk_a6abdc87464e5c00d90059b302746c55d005dbe8d29c79df')
    }
    
    print("=" * 80)
    print("VitalSight Notification Service Test")
    print("=" * 80)
    print()
    
    # Initialize service
    print("[1/4] Initializing notification service...")
    notification_service = NotificationService(**config)
    print()
    
    if not notification_service.enabled:
        print("[ERROR] Notification service failed to initialize!")
        print("Please check your Twilio and ElevenLabs credentials.")
        return
    
    print("[✓] Notification service initialized successfully")
    print()
    
    # Test with a sample report (simulating a CRITICAL fire detection)
    print("[2/4] Preparing test alert...")
    sample_report = """
    IMMEDIATE SITUATION:
    A fire has been detected in the northwest section of the facility. 
    Flames are visible in multiple locations with heavy smoke accumulation. 
    Two individuals are seen near the affected area moving toward the exit.
    
    OBSERVABLE DETAILS:
    Multiple flame sources detected across approximately 15-20% of the visible frame.
    Dense smoke reducing visibility in the upper portion of the area.
    Two people located approximately 10 meters from the nearest flame source.
    
    ASSESSMENT:
    Rapidly developing fire situation requiring immediate evacuation and fire suppression.
    Smoke inhalation risk is HIGH due to enclosed space.
    Structural integrity may be compromised.
    
    RECOMMENDED ACTION:
    Immediately activate building-wide evacuation alarm.
    Contact fire department and provide exact location.
    Ensure all personnel evacuate via designated emergency exits.
    Deploy fire suppression systems if available.
    """
    
    print("[✓] Test report prepared")
    print()
    
    # Test categories
    test_cases = [
        ("fall", "Fall detection (should NOT trigger call)"),
        ("violence_panic", "Crowd violence (should NOT trigger call)"),
        ("fire", "Fire detection (SHOULD trigger call)")
    ]
    
    print("[3/4] Testing alert logic...")
    print()
    
    for category, description in test_cases:
        print(f"   Testing: {description}")
        result = notification_service.send_alert(
            category=category,
            report_text=sample_report,
            source_path="test_video.mp4"
        )
        
        if result.get('call'):
            print(f"   ✓ Voice call initiated: {result.get('call_sid')}")
            print(f"   Status: {result.get('status')}")
        else:
            print(f"   - No call needed: {result.get('reason')}")
        print()
    
    print("[4/4] Test completed!")
    print()
    print("=" * 80)
    print("If you received a call for the 'fire' test, the integration is working!")
    print("=" * 80)

if __name__ == "__main__":
    test_notification()

