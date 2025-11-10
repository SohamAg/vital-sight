"""
VitalSight Web Dashboard
Modern web interface for viewing processed videos and detection reports
"""
from flask import Flask, render_template, jsonify, request, send_from_directory, Response, session, redirect, url_for
from pathlib import Path
import json
import re
from werkzeug.utils import secure_filename
import os
import cv2
import threading
import sys
import time
import markdown
from functools import wraps

# Try to import VitalSightV2, but make it optional for web serving
try:
    from edge.detector_v2 import VitalSightV2
    from edge.notifications import NotificationService
    import google.generativeai as genai
    import numpy as np
    DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Detection not available: {e}")
    print("[INFO] Web interface will work, but upload processing disabled")
    DETECTION_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.secret_key = 'vitalsight_secret_key_2024'  # Change this in production

# User credentials (local authentication)
USERS = {
    'sohamkagrawal@gmail.com': {
        'password': 'vitalsight',
        'name': 'Soham',
        'email': 'sohamkagrawal@gmail.com'
    }
}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Configuration
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("data/demo_reports")
UPLOAD_DIR = Path("data/uploads")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

# Camera Name Mapping - Maps video filenames to camera identifiers
CAMERA_MAPPING = {
    "fire_sample1": "Camera 01 - Loading Dock",
    "fire_sample2": "Camera 02 - Main Entrance",
    "fire_sample3": "Camera 03 - Warehouse",
    "fall_sample1": "Camera 04 - Hallway A",
    "fall_sample2": "Camera 05 - Cafeteria",
    "fall_sample3": "Camera 06 - Lobby",
    "distress_sample1": "Camera 07 - Office Floor 2",
    "distress_sample2": "Camera 08 - Parking Garage",
    "crowd_sample1": "Camera 09 - Main Plaza",
    "crowd_sample2": "Camera 10 - Event Hall",
    "crowd_sample4": "Camera 11 - Security Gate",
    "injury_sample1": "Camera 12 - Factory Floor",
    "chill_sample1": "Camera 13 - Break Room",
    "chill_sample2": "Camera 14 - Reception",
    "chill_sample3": "Camera 15 - Corridor B",
}

def get_camera_name(video_path):
    """Convert video path to camera name"""
    if not video_path:
        return "Unknown Camera"
    
    filename = Path(video_path).stem  # Get filename without extension
    
    # Remove _processed suffix if present
    if filename.endswith("_processed"):
        filename = filename[:-10]
    
    return CAMERA_MAPPING.get(filename, f"Camera - {filename}")

# Notification Service Configuration
NOTIFICATION_CONFIG = {
    'twilio_account_sid': os.environ.get('TWILIO_ACCOUNT_SID'),
    'twilio_auth_token': os.environ.get('TWILIO_AUTH_TOKEN'),
    'twilio_phone_number': os.environ.get('TWILIO_PHONE_NUMBER'),
    'alert_phone_number': os.environ.get('ALERT_PHONE_NUMBER'),
    'elevenlabs_api_key': os.environ.get('ELEVENLABS_API_KEY'),
    # SendGrid Email configuration
    'alert_email': os.environ.get('ALERT_EMAIL'),
    'sendgrid_api_key': os.environ.get('SENDGRID_API_KEY'),
    'from_email': os.environ.get('FROM_EMAIL')  # Verified sender in SendGrid
}

# Initialize notification service
notification_service = None
if DETECTION_AVAILABLE:
    try:
        notification_service = NotificationService(**NOTIFICATION_CONFIG)
        print("[INFO] Notification service initialized for voice alerts")
    except Exception as e:
        print(f"[WARNING] Could not initialize notification service: {e}")

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Processing status tracker
processing_status = {}

# Live processing frame buffer for streaming
live_frames = {}  # {job_id: latest_frame}

def get_severity_from_report(report_path):
    """Extract severity from report file"""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if "🔴 CRITICAL" in content:
                return "CRITICAL"
            elif "🟠 HIGH PRIORITY" in content:
                return "HIGH"
            elif "🟡 MEDIUM PRIORITY" in content:
                return "MEDIUM"
            elif "🟢 LOW PRIORITY" in content:
                return "LOW"
    except:
        pass
    return None

def parse_report(report_path):
    """Parse report file into structured data"""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract key information using regex
        alert_type = re.search(r'ALERT TYPE: (.+)', content)
        severity = re.search(r'Severity Level: (.+)', content)
        confidence = re.search(r'Detection Confidence: (.+)', content)
        timestamp = re.search(r'Timestamp: (.+)', content)
        frame = re.search(r'Evidence Frame: (.+)', content)
        
        # Extract situation report
        situation_match = re.search(r'SITUATION REPORT:\n={80}\n\n(.+?)\n\n={80}', content, re.DOTALL)
        situation = situation_match.group(1).strip() if situation_match else "No situation report available"
        
        # Convert situation report to HTML using markdown
        situation_html = markdown.markdown(
            situation,
            extensions=['nl2br', 'sane_lists', 'tables']
        )
        
        # Extract notification protocol
        notification_method = re.search(r'Notification Method: (.+)', content)
        response_time = re.search(r'Expected Response Time: (.+)', content)
        
        return {
            'alert_type': alert_type.group(1) if alert_type else "Unknown",
            'severity': severity.group(1) if severity else "Unknown",
            'confidence': confidence.group(1) if confidence else "N/A",
            'timestamp': timestamp.group(1) if timestamp else "N/A",
            'frame': frame.group(1) if frame else None,
            'situation': situation,
            'situation_html': situation_html,
            'notification_method': notification_method.group(1) if notification_method else "N/A",
            'response_time': response_time.group(1) if response_time else "N/A"
        }
    except Exception as e:
        print(f"Error parsing report: {e}")
        return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if 'user' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email in USERS and USERS[email]['password'] == password:
            session['user'] = USERS[email]
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid email or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout"""
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    """Main grid view showing all processed videos"""
    from flask import make_response
    
    videos = sorted(list(PROCESSED_DIR.glob("*_processed.mp4")))
    video_data = []
    
    for video in videos:
        name = video.stem.replace("_processed", "")
        
        # Check for reports - look for any category
        reports = list(REPORTS_DIR.glob(f"{name}_*_report.txt"))
        has_alert = len(reports) > 0
        severity = None
        categories = []
        
        if reports:
            severity = get_severity_from_report(reports[0])
            # Extract categories from report filenames
            for report in reports:
                category = report.stem.replace(f"{name}_", "").replace("_report", "")
                categories.append(category)
        
        video_data.append({
            'name': name,
            'filename': video.name,
            'has_alert': has_alert,
            'severity': severity,
            'categories': categories,
            'report_count': len(reports)
        })
    
    response = make_response(render_template('grid.html', videos=video_data))
    # Prevent caching to always show latest videos
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/detail/<video_name>')
@login_required
def detail(video_name):
    """Video detail view with report"""
    video_path = PROCESSED_DIR / f"{video_name}_processed.mp4"
    
    if not video_path.exists():
        return "Video not found", 404
    
    # Get all reports for this video
    reports = sorted(list(REPORTS_DIR.glob(f"{video_name}_*_report.txt")))
    
    report_data = []
    for report_path in reports:
        parsed = parse_report(report_path)
        if parsed:
            # Add frame path
            frame_name = report_path.stem.replace("_report", "_frame.jpg")
            frame_path = REPORTS_DIR / frame_name
            if frame_path.exists():
                parsed['frame_path'] = f'/reports/{frame_name}'
            report_data.append(parsed)
    
    return render_template('detail.html', 
                          video_name=video_name,
                          video_file=f"{video_name}_processed.mp4",
                          reports=report_data)

@app.route('/upload')
@login_required
def upload_page():
    """Upload interface"""
    return render_template('upload.html')

@app.route('/webcam')
@login_required
def webcam_page():
    """Live webcam interface"""
    return render_template('webcam.html')

@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve processed video files"""
    return send_from_directory(PROCESSED_DIR, filename)

@app.route('/reports/<path:filename>')
def serve_report_file(filename):
    """Serve report files (images, text)"""
    return send_from_directory(REPORTS_DIR, filename)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    if not DETECTION_AVAILABLE:
        return jsonify({'error': 'Detection module not available'}), 503
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    upload_path = UPLOAD_DIR / filename
    file.save(str(upload_path))
    
    # Generate unique ID for this processing job
    job_id = filename.replace('.', '_')
    processing_status[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Video uploaded, queued for processing',
        'upload_path': str(upload_path)
    }
    
    # Start processing in background thread
    def process_video_background():
        try:
            import time
            processing_status[job_id]['status'] = 'processing'
            processing_status[job_id]['message'] = 'Initializing detector...'
            
            # Initialize detector
            detector = VitalSightV2(cfg_path="config.yaml", gemini_api_key=GEMINI_KEY)
            
            # Set camera name instead of file path
            camera_name = get_camera_name(str(upload_path))
            detector.source_path = camera_name
            print(f"[INFO] Processing video from: {camera_name}")
            
            # Attach notification service to Gemini reporter for email + voice alerts
            if notification_service and detector.gemini_reporter:
                detector.gemini_reporter.notification_service = notification_service
                print("[INFO] Email notifications enabled for ALL detections")
                print("[INFO] Voice call alerts enabled for HIGH/CRITICAL detections")
            
            # Determine output path
            output_name = Path(filename).stem
            output_path = PROCESSED_DIR / f"{output_name}_processed.mp4"
            
            # Open video
            cap = cv2.VideoCapture(str(upload_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            processing_status[job_id]['total_frames'] = total_frames
            processing_status[job_id]['processed_frames'] = 0
            processing_status[job_id]['message'] = 'Running YOLO detection and pose estimation...'
            
            # Define frame callback for live streaming
            def stream_frame(frame_data):
                """Callback function to stream frames during processing"""
                try:
                    # Extract data from dictionary
                    annotated_frame = frame_data['annotated_frame']
                    frame_num = frame_data['frame_id']
                    total = frame_data['total_frames']
                    
                    # Update progress
                    processing_status[job_id]['processed_frames'] = frame_num
                    if total > 0:
                        processing_status[job_id]['progress'] = int((frame_num / total) * 90)  # Up to 90%, reserve 10% for reports
                    
                    # Encode frame for streaming (downsample for faster streaming)
                    h, w = annotated_frame.shape[:2]
                    max_width = 960  # Stream at lower resolution for speed
                    if w > max_width:
                        scale = max_width / w
                        new_w = max_width
                        new_h = int(h * scale)
                        stream_frame_resized = cv2.resize(annotated_frame, (new_w, new_h))
                    else:
                        stream_frame_resized = annotated_frame
                    
                    # Encode to JPEG
                    ret, buffer = cv2.imencode('.jpg', stream_frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        live_frames[job_id] = buffer.tobytes()
                except Exception as e:
                    print(f"[WARNING] Stream frame error: {e}")
            
            # Process video with live streaming callback
            print(f"[INFO] Processing {filename} with VitalSight...")
            detector.process(
                source=str(upload_path),
                display=False,
                save_output=True,
                output_path=str(output_path),
                frame_callback=stream_frame  # Enable live streaming!
            )
            
            print(f"[INFO] Processing complete for {filename}")
            processing_status[job_id]['processed_frames'] = total_frames
            
            processing_status[job_id]['message'] = 'Generating AI reports...'
            processing_status[job_id]['progress'] = 98
            
            # Wait for Gemini reports if any
            if detector.gemini_reporter:
                detector.gemini_reporter.wait_for_all_reports()
            
            processing_status[job_id]['status'] = 'completed'
            processing_status[job_id]['progress'] = 100
            processing_status[job_id]['message'] = 'Processing complete!'
            processing_status[job_id]['output_name'] = output_name
            
            # Processing complete
            time.sleep(0.5)
            
        except Exception as e:
            import traceback
            processing_status[job_id]['status'] = 'failed'
            processing_status[job_id]['message'] = f'Error: {str(e)}'
            print(f"[ERROR] Processing failed: {traceback.format_exc()}")
            if job_id in live_frames:
                del live_frames[job_id]
    
    thread = threading.Thread(target=process_video_background, daemon=True)
    thread.start()
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'message': 'Upload successful, processing started'
    })

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded video files for preview"""
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/api/status/<job_id>')
def get_status(job_id):
    """Get processing status for uploaded video"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(processing_status[job_id])

@app.route('/api/stream/<job_id>')
def stream_processing(job_id):
    """Stream live processing frames"""
    def generate():
        """Generate frames for streaming"""
        import time
        
        while job_id in processing_status:
            status = processing_status[job_id]['status']
            
            # If there's a frame available, send it
            if job_id in live_frames:
                frame = live_frames[job_id]
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # If completed or failed, stop streaming
            if status in ['completed', 'failed']:
                break
            
            time.sleep(0.033)  # ~30 FPS
        
        # Clean up after stream ends
        if job_id in live_frames:
            del live_frames[job_id]
    
    return Response(generate(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# ============================================================================
# WEBCAM LIVE DETECTION SYSTEM
# ============================================================================

# Webcam state management
webcam_state = {
    'active': False,
    'detector': None,
    'cap': None,
    'processing_report': False,  # Lock while generating report/sending email
    'current_detections': {},  # {category: confidence}
    'frame_count': 0,
    'people_count': 0,
    'current_frame': None,  # Latest annotated frame for streaming
    'last_frame_time': 0,
    'gemini_prediction': None,  # Gemini's classification: 'fall', 'distress', or None
    'gemini_checked_frame': -1  # Last frame we checked with Gemini
}

def gemini_check_frame(frame, frame_id):
    """Use Gemini Vision to classify if fall or distress is happening"""
    if not DETECTION_AVAILABLE:
        return
    
    try:
        import tempfile
        import os
        
        # Save frame temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_file.name, frame)
        temp_file.close()
        
        # Quick Gemini classification (async in background thread)
        def classify_async():
            try:
                # Upload frame to Gemini
                image = genai.upload_file(temp_file.name)
                
                # Classification prompt
                classify_prompt = """You are monitoring a live webcam feed for emergency situations.

Analyze this image and classify what you see:

- Is someone's HEAD significantly DOWN or LOWERED (bowing, looking down intensely, collapsed)?
- Does someone have their HAND ON CHEST (clutching, grabbing chest area)?

Respond with ONLY ONE WORD:
- "FALL" if you see head down, lowered posture, or lying position
- "DISTRESS" if you see hand on chest or clutching chest
- "NONE" if everything looks normal

Your classification:"""
                
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                response = model.generate_content([classify_prompt, image])
                prediction = response.text.strip().upper()
                
                print(f"[GEMINI] Frame {frame_id}: {prediction}")
                
                # Clean up
                try:
                    genai.delete_file(image.name)
                    os.unlink(temp_file.name)
                except:
                    pass
                
                # Store prediction to boost YOLO detection scores
                if prediction in ['FALL', 'DISTRESS']:
                    category = 'fall' if prediction == 'FALL' else 'distress'
                    webcam_state['gemini_prediction'] = category
                    webcam_state['gemini_checked_frame'] = frame_id
                    print(f"[GEMINI] ✓ {prediction} detected - boosting {category} score in YOLO pipeline")
                else:
                    # Clear prediction if nothing detected
                    webcam_state['gemini_prediction'] = None
                    
            except Exception as e:
                print(f"[ERROR] Gemini classification failed: {e}")
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
        
        # Run in background thread
        check_thread = threading.Thread(target=classify_async, daemon=True)
        check_thread.start()
        
    except Exception as e:
        print(f"[ERROR] Could not start Gemini classification: {e}")

def generate_full_report_async(frame, category, confidence):
    """Generate full report in background without stopping webcam"""
    def generate():
        try:
            print(f"\n[REPORT] Generating full report for {category}...")
            
            # Save frame
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            frame_filename = f"webcam_{category}_{timestamp}_frame.jpg"
            frame_path = REPORTS_DIR / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            category_names = {
                'fall': 'FALL',
                'distress': 'DISTRESS / CHEST PAIN'
            }
            
            # Generate detailed Gemini report
            report_prompt = f"""You are analyzing a LIVE WEBCAM FEED where an emergency situation has been detected.

DETECTED SITUATION: {category_names.get(category, category.upper())}
Confidence: {confidence:.1%}
Source: Live Webcam Feed
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

Based on the image, provide a detailed report with the following sections:

## SITUATION ANALYSIS
Describe exactly what you observe in the image. What is the person's position, posture, and apparent condition?

## SEVERITY ASSESSMENT
Is this a genuine emergency requiring immediate response? Rate severity as LOW, MEDIUM, HIGH, or CRITICAL.

## RECOMMENDED ACTION
What immediate steps should be taken? Be specific and actionable.

## NOTIFICATION PROTOCOL
Based on the severity:
- LOW/MEDIUM: Email notification only
- HIGH/CRITICAL: Email + phone call to emergency contact

Keep the report concise (3-4 sentences per section) but informative for emergency responders."""
            
            # Call Gemini for full report
            image = genai.upload_file(str(frame_path))
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content([report_prompt, image])
            report_text = response.text
            
            # Clean up
            try:
                genai.delete_file(image.name)
            except:
                pass
            
            # Save report
            report_filename = f"webcam_{category}_{timestamp}_report.txt"
            report_path = REPORTS_DIR / report_filename
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"Live Webcam Detection Report\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Detection: {category_names.get(category, category.upper())}\n")
                f.write(f"Confidence: {confidence:.1%}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source: Live Webcam Feed\n\n")
                f.write(f"{'='*50}\n\n")
                f.write(report_text)
            
            print(f"[REPORT] ✓ Report saved: {report_filename}")
            
            # Store report for UI display
            webcam_state['latest_report'] = {
                'category': category,
                'confidence': confidence,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'report_text': report_text,
                'frame_path': str(frame_path),
                'report_path': str(report_path)
            }
            
            # Send notifications
            if notification_service and notification_service.email_enabled:
                print(f"[REPORT] Sending email notification...")
                notification_service.send_alert(
                    category=category,
                    report_text=report_text,
                    source_path='Live Webcam Feed',
                    frame_path=frame_path,
                    confidence=confidence,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                )
                print(f"[REPORT] ✓ Email sent successfully")
            
            # Clear processing flag after 10 seconds (allows new detections)
            time.sleep(10)
            webcam_state['processing_report'] = False
            
        except Exception as e:
            print(f"[ERROR] Report generation failed: {e}")
            import traceback
            traceback.print_exc()
            webcam_state['processing_report'] = False
    
    # Run in background thread
    report_thread = threading.Thread(target=generate, daemon=True)
    report_thread.start()

def webcam_detection_callback(frame_data):
    """Callback for VitalSightV2 detector to receive annotated frames and detections"""
    try:
        webcam_state['current_frame'] = frame_data['annotated_frame']
        webcam_state['frame_count'] = frame_data['frame_id']
        webcam_state['people_count'] = frame_data.get('people_count', 0)
        
        # Get active detections from frame_data
        active_detections = frame_data.get('active_detections', {})
        webcam_state['current_detections'] = active_detections
        webcam_state['last_frame_time'] = time.time()
        
        # Check if we have a new detection that needs a report
        if active_detections and not webcam_state.get('processing_report', False):
            for category, confidence in active_detections.items():
                if category in ['fall', 'distress']:
                    # Check if we haven't generated a report for this category recently
                    last_reported = webcam_state.get('last_reported_category', None)
                    if last_reported != category:
                        print(f"[WEBCAM] {category.upper()} confirmed by YOLO pipeline!")
                        webcam_state['processing_report'] = True
                        webcam_state['last_reported_category'] = category
                        
                        # Generate report in background (doesn't stop webcam)
                        generate_full_report_async(
                            frame_data['annotated_frame'].copy(),
                            category,
                            confidence
                        )
                        break
        
        # Every 60 frames (~2 seconds), use Gemini to classify what's happening
        if webcam_state['frame_count'] % 60 == 0 and webcam_state.get('active', False):
            # Use Gemini Vision to classify the frame
            gemini_check_frame(frame_data['annotated_frame'], frame_data['frame_id'])
    except Exception as e:
        print(f"[ERROR] Webcam callback error: {e}")
        import traceback
        traceback.print_exc()

def get_gemini_boost():
    """Callback for detector to get Gemini's classification"""
    prediction = webcam_state.get('gemini_prediction')
    if prediction:
        # Clear after use to avoid continuous boosting
        frame_checked = webcam_state.get('gemini_checked_frame', -1)
        current_frame = webcam_state.get('frame_count', 0)
        
        # Only boost if this is recent (within 120 frames / 4 seconds)
        if current_frame - frame_checked < 120:
            return prediction
        else:
            # Too old, clear it
            webcam_state['gemini_prediction'] = None
    return None

def run_webcam_detection():
    """Background thread that runs the FULL VitalSightV2.process() method"""
    detector = webcam_state['detector']
    
    if not detector:
        print("[ERROR] No detector in webcam state")
        return
    
    print("[WEBCAM] Starting full detection pipeline...")
    print("[WEBCAM] Gemini classifications will boost YOLO scores")
    print("[WEBCAM] Opening webcam (source=0)...")
    
    # Set source for camera naming
    detector.source_path = "Live Webcam Feed"
    
    # Run the FULL detection process with Gemini boost callback
    # This includes ALL detection logic: fall, distress (hand-chest), violence, injury
    try:
        print("[WEBCAM] Calling detector.process()...")
        detector.process(
            source=0,  # Webcam
            display=False,  # No OpenCV window
            save_output=False,  # Don't save video
            frame_callback=webcam_detection_callback,
            gemini_boost_callback=get_gemini_boost  # Pass Gemini predictions to boost scores
        )
        print("[WEBCAM] detector.process() returned normally")
    except Exception as e:
        print(f"[ERROR] Webcam detection failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        was_active = webcam_state['active']
        webcam_state['active'] = False
        print(f"[WEBCAM] Detection stopped (was_active={was_active})")

@app.route('/api/webcam/start', methods=['POST'])
@login_required
def start_webcam():
    """Initialize webcam detection system"""
    global webcam_state
    
    if not DETECTION_AVAILABLE:
        return jsonify({'success': False, 'error': 'Detection not available'}), 503
    
    if webcam_state['active']:
        return jsonify({'success': True, 'message': 'Webcam already running'})
    
    try:
        # Initialize detector
        detector = VitalSightV2(cfg_path="config.yaml", gemini_api_key=GEMINI_KEY)
        
        # ============================================================
        # WEBCAM-SPECIFIC CONFIGURATION (MUCH STRICTER THRESHOLDS)
        # ============================================================
        
        # 1. COMPLETELY DISABLE FIRE DETECTION
        if 'logic' in detector.cfg and 'fire' in detector.cfg['logic']:
            detector.cfg['logic']['fire']['enabled'] = False
        
        # 2. BALANCED HYSTERESIS THRESHOLDS (stricter than default, but functional)
        #    Default enter: 0.45, we're making it 0.70-0.75 (stricter but reasonable)
        detector.cfg['hysteresis']['enter'] = 0.70  # General threshold (was 0.45)
        detector.cfg['hysteresis']['exit'] = 0.45   # Exit threshold
        detector.cfg['hysteresis']['fall_enter'] = 0.72  # Fall: detect head down (was 0.65, not too strict)
        detector.cfg['hysteresis']['fall_exit'] = 0.50
        detector.cfg['hysteresis']['injury_enter'] = 0.85  # Injury: keep very strict (was 0.60)
        detector.cfg['hysteresis']['injury_exit'] = 0.65
        
        # 3. MODERATE DEBOUNCE FRAMES (enough to avoid jitter, not too long)
        detector.cfg['logic']['fall']['debounce_f'] = 8   # Was 2, now 8 frames (~0.3s at 30fps)
        detector.cfg['logic']['resp']['debounce_f'] = 10  # Was 3, now 10 frames (distress: hand to chest)
        detector.cfg['logic']['violence']['debounce_f'] = 25  # Keep strict: 25 frames
        detector.cfg['logic']['severe_injury']['debounce_f'] = 30  # Keep very strict: 30 frames
        
        # 4. LOWER MOTION THRESHOLDS for distress (to detect hand movement to chest)
        detector.cfg['logic']['resp']['motion_thresh'] = 8.0   # Was 6.0 default, now 8.0 (slightly stricter)
        detector.cfg['logic']['resp']['prox_thresh'] = 0.25    # Hand-chest proximity threshold (slightly more forgiving)
        detector.cfg['logic']['violence']['flow_mag'] = 25.0   # Keep strict: 25.0
        detector.cfg['logic']['severe_injury']['motion_eps'] = 20.0  # Keep strict: 20.0
        
        # 5. MODERATE CONFIDENCE THRESHOLDS
        detector.conf = 0.45  # YOLO confidence (was 0.35, now 0.45 - slightly stricter but functional)
        
        # 6. REINITIALIZE DEBOUNCERS WITH NEW CONFIG VALUES
        #    (Debouncers are created in __init__, so we must recreate them)
        from edge.detector_v2 import Debouncer
        hys = detector.cfg["hysteresis"]
        detector.db = {
            "fall": Debouncer(
                hys.get("fall_enter", hys["enter"]),
                hys.get("fall_exit", hys["exit"]),
                detector.cfg["logic"]["fall"]["debounce_f"]
            ),
            "distress": Debouncer(
                hys["enter"], 
                hys["exit"], 
                detector.cfg["logic"]["resp"]["debounce_f"]
            ),
            "violence_panic": Debouncer(
                hys["enter"], 
                hys["exit"], 
                detector.cfg["logic"]["violence"]["debounce_f"]
            ),
            "fire": Debouncer(
                0.99,  # Fire needs 99% confidence (essentially disabled)
                0.95, 
                100  # Need 100 consecutive frames (will never happen)
            ),
            "severe_injury": Debouncer(
                hys.get("injury_enter", hys["enter"]),
                hys.get("injury_exit", hys["exit"]),
                detector.cfg["logic"]["severe_injury"]["debounce_f"]
            ),
        }
        
        # 7. CLEAR ANY STALE DETECTIONS AND RESET STATE
        detector.first_detections = {}  # Clear any detections from initialization
        
        # 8. ENSURE FIRE IS REMOVED FROM WEBCAM FOCUS LIST
        if 'runtime' in detector.cfg and 'webcam_focus' in detector.cfg['runtime']:
            detector.cfg['runtime']['webcam_focus'] = ['fall', 'distress', 'violence_panic', 'severe_injury']
            # Explicitly remove 'fire' if present
            if 'fire' in detector.cfg['runtime']['webcam_focus']:
                detector.cfg['runtime']['webcam_focus'].remove('fire')
        
        print("[WEBCAM] ============================================")
        print("[WEBCAM] 🤖 HYBRID DETECTION: GEMINI + YOLO")
        print("[WEBCAM] ============================================")
        print(f"  - Gemini Classification: Every 60 frames (~2 seconds)")
        print(f"  - What Gemini Classifies:")
        print(f"    • FALL: Head down, lying, collapsed posture")
        print(f"    • DISTRESS: Hand on chest, clutching")
        print(f"  - YOLO Pipeline: Uses Gemini classification to boost scores")
        print(f"  - Debouncing: {detector.cfg['logic']['fall']['debounce_f']} frames (fall), {detector.cfg['logic']['resp']['debounce_f']} frames (distress)")
        print(f"  - Report Generation: Full Gemini report + email/call")
        print(f"  - Fire detection: DISABLED")
        print("[WEBCAM] ============================================")
        
        # Attach notification service for email alerts
        if notification_service and detector.gemini_reporter:
            detector.gemini_reporter.notification_service = notification_service
            print("[WEBCAM] Email notifications enabled")
        
        webcam_state['detector'] = detector
        webcam_state['active'] = True
        webcam_state['frame_count'] = 0
        webcam_state['processing_report'] = False
        webcam_state['current_detections'] = {}  # Clear any detections
        
        # Start detection in background thread
        detection_thread = threading.Thread(target=run_webcam_detection, daemon=True)
        detection_thread.start()
        
        print("[WEBCAM] Detection system started with REDUCED sensitivity")
        return jsonify({'success': True, 'message': 'Webcam detection started'})
    
    except Exception as e:
        print(f"[ERROR] Webcam start failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/webcam/stop', methods=['POST'])
@login_required
def stop_webcam():
    """Stop webcam detection system"""
    global webcam_state
    
    webcam_state['active'] = False
    webcam_state['current_detections'] = {}
    webcam_state['frame_count'] = 0
    webcam_state['current_frame'] = None
    webcam_state['gemini_prediction'] = None  # Clear Gemini prediction
    webcam_state['gemini_checked_frame'] = -1
    webcam_state['latest_report'] = None  # Clear report
    webcam_state['last_reported_category'] = None
    webcam_state['processing_report'] = False
    
    # The background thread will exit on its own
    print("[WEBCAM] Detection system stopping...")
    return jsonify({'success': True, 'message': 'Webcam detection stopped'})

@app.route('/api/webcam/stats')
@login_required
def webcam_stats():
    """Get current webcam detection statistics"""
    return jsonify({
        'active': webcam_state['active'],
        'frame_count': webcam_state['frame_count'],
        'people_count': webcam_state['people_count'],
        'detections': webcam_state['current_detections'],
        'gemini_prediction': webcam_state.get('gemini_prediction', None),
        'latest_report': webcam_state.get('latest_report', None),
        'processing_report': webcam_state.get('processing_report', False)
    })

@app.route('/api/webcam/stream')
@login_required
def webcam_stream():
    """MJPEG stream - serves frames from the background detection thread"""
    
    def generate_frames():
        """Serve annotated frames from webcam_state"""
        import time
        
        if not DETECTION_AVAILABLE:
            error_frame = create_error_frame("Detection not available")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            return
        
        if not webcam_state.get('active'):
            error_frame = create_error_frame("Please start monitoring first")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            return
        
        print("[WEBCAM] Stream started")
        last_frame_id = -1
        
        try:
            while webcam_state.get('active'):
                # Get current annotated frame from detection thread
                frame = webcam_state.get('current_frame')
                frame_id = webcam_state.get('frame_count', 0)
                
                if frame is not None and frame_id != last_frame_id:
                    # New frame available
                    last_frame_id = frame_id
                    
                    # Encode and yield
                    ret, buffer = cv2.imencode('.jpg', frame, 
                                              [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    # No new frame, wait a bit
                    time.sleep(0.01)
                
                # Check if detection thread is still alive (increased timeout)
                if webcam_state.get('last_frame_time', 0) > 0:
                    time_since_last_frame = time.time() - webcam_state['last_frame_time']
                    if time_since_last_frame > 10:
                        # No frames for 10 seconds, something's wrong
                        print(f"[WARNING] No frames received for {time_since_last_frame:.1f} seconds")
                        print(f"[DEBUG] Webcam active: {webcam_state.get('active')}")
                        print(f"[DEBUG] Frame count: {webcam_state.get('frame_count')}")
                        break
        
        except Exception as e:
            print(f"[ERROR] Stream generation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[WEBCAM] Stream ended")
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def create_error_frame(message):
    """Create an error message frame"""
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes() if ret else b''

@app.route('/api/webcam/detect', methods=['POST'])
def webcam_detect():
    """Legacy endpoint - kept for compatibility"""
    return jsonify({
        'detections': webcam_state['current_detections'],
        'active': webcam_state['active']
    })

if __name__ == '__main__':
    print("=" * 80)
    print("🚨 VitalSight Web Dashboard")
    print("=" * 80)
    print(f"\nProcessed Videos: {len(list(PROCESSED_DIR.glob('*_processed.mp4')))} found")
    print(f"Detection Reports: {len(list(REPORTS_DIR.glob('*_report.txt')))} found")
    print("\nStarting server at http://localhost:5000")
    print("=" * 80)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

