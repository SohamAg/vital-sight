"""
VitalSight Web Dashboard
Modern web interface for viewing processed videos and detection reports
"""
from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from pathlib import Path
import json
import re
from werkzeug.utils import secure_filename
import os
import cv2
import threading
import sys

# Try to import VitalSightV2, but make it optional for web serving
try:
    from edge.detector_v2 import VitalSightV2
    DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Detection not available: {e}")
    print("[INFO] Web interface will work, but upload processing disabled")
    DETECTION_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configuration
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("data/demo_reports")
UPLOAD_DIR = Path("data/uploads")
GEMINI_KEY = "AIzaSyDPE3QNZqVino7KJvFDeZ_nfYcQ627FcMo"

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
            elif "🟡 MEDIUM" in content:
                return "MEDIUM"
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
            'notification_method': notification_method.group(1) if notification_method else "N/A",
            'response_time': response_time.group(1) if response_time else "N/A"
        }
    except Exception as e:
        print(f"Error parsing report: {e}")
        return None

@app.route('/')
def index():
    """Main grid view showing all processed videos"""
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
    
    return render_template('grid.html', videos=video_data)

@app.route('/detail/<video_name>')
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
def upload_page():
    """Upload interface"""
    return render_template('upload.html')

@app.route('/webcam')
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
            processing_status[job_id]['status'] = 'processing'
            processing_status[job_id]['message'] = 'Processing video...'
            
            # Open video to process frame by frame
            cap = cv2.VideoCapture(str(upload_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            # Initialize detector
            detector = VitalSightV2(cfg_path="config.yaml", gemini_api_key=GEMINI_KEY)
            
            # Set up video writer
            output_name = Path(filename).stem
            output_path = PROCESSED_DIR / f"{output_name}_processed.mp4"
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Try H.264 codec for browser compatibility
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Process frame by frame for live streaming
            detector.source_path = str(upload_path)
            detector.first_detections = {}
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process single frame through detector
                results = detector.detect_model.track(frame, persist=True, verbose=False)
                pose_results = detector.pose_model(frame, verbose=False)
                
                # Run situation detection (simplified)
                active = detector._analyze_situations(results, pose_results, frame)
                
                # Annotate frame
                annotated = frame.copy()
                if results and len(results) > 0:
                    annotated = results[0].plot()
                
                # Add detection labels
                if active:
                    for i, (category, conf) in enumerate(active):
                        cat_display = detector.situation_names.get(category, category)
                        label = f"{cat_display}: {conf:.1%}"
                        cv2.putText(annotated, label, (10, 30 + i*30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                # Write to output file
                video_writer.write(annotated)
                
                # Update live frame for streaming
                _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                live_frames[job_id] = buffer.tobytes()
                
                # Update progress
                progress = int((frame_count / total_frames) * 100)
                processing_status[job_id]['progress'] = progress
                processing_status[job_id]['message'] = f'Processing: {frame_count}/{total_frames} frames'
            
            cap.release()
            video_writer.release()
            
            # Generate reports if detections were made
            if detector.gemini_reporter and detector.first_detections:
                processing_status[job_id]['message'] = 'Generating AI reports...'
                for category, detection_data in detector.first_detections.items():
                    if not detection_data["reported"]:
                        detector.gemini_reporter.generate_report_async(
                            detection_data["frame"],
                            category,
                            detection_data["confidence"],
                            str(upload_path)
                        )
                        detection_data["reported"] = True
                
                detector.gemini_reporter.wait_for_all_reports()
            
            processing_status[job_id]['status'] = 'completed'
            processing_status[job_id]['progress'] = 100
            processing_status[job_id]['message'] = 'Processing complete!'
            processing_status[job_id]['output_name'] = output_name
            
            # Clean up live frame
            if job_id in live_frames:
                del live_frames[job_id]
            
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

