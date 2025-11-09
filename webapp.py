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
import markdown
from functools import wraps

# Try to import VitalSightV2, but make it optional for web serving
try:
    from edge.detector_v2 import VitalSightV2
    from edge.notifications import NotificationService
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
GEMINI_KEY = "AIzaSyDPE3QNZqVino7KJvFDeZ_nfYcQ627FcMo"

# Notification Service Configuration
NOTIFICATION_CONFIG = {
    'twilio_account_sid': os.environ.get('TWILIO_ACCOUNT_SID', 'AC9482d7139f9d9056cbdf9159f02052db'),
    'twilio_auth_token': os.environ.get('TWILIO_AUTH_TOKEN', '5c9fd678f6689941e0cebcae6cebac35'),
    'twilio_phone_number': os.environ.get('TWILIO_PHONE_NUMBER', '+18663508040'),
    'alert_phone_number': os.environ.get('ALERT_PHONE_NUMBER', '+19297602752'),
    'elevenlabs_api_key': os.environ.get('ELEVENLABS_API_KEY', 'sk_a6abdc87464e5c00d90059b302746c55d005dbe8d29c79df')
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
            detector.source_path = str(upload_path)
            
            # Attach notification service to Gemini reporter for voice alerts
            if notification_service and detector.gemini_reporter:
                detector.gemini_reporter.notification_service = notification_service
                print("[INFO] Voice call alerts enabled for CRITICAL detections")
            
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
            def stream_frame(annotated_frame, frame_num, total):
                """Callback function to stream frames during processing"""
                try:
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

@app.route('/api/webcam/detect', methods=['POST'])
def webcam_detect():
    """Process single frame from webcam"""
    # This would handle real-time webcam frame detection
    # For now, return placeholder
    return jsonify({
        'detections': [],
        'message': 'Webcam detection endpoint ready'
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

