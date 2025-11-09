"""
Gemini VLM Reporter - Generates detailed reports for detected situations
"""
import os
import cv2
import time
import threading
import google.generativeai as genai
from pathlib import Path


class GeminiReporter:
    """
    Handles sending frames to Google Gemini VLM for detailed situation analysis.
    """
    
    def __init__(self, api_key, reports_dir="data/demo_reports", notification_service=None):
        """
        Initialize the Gemini reporter.
        
        Args:
            api_key: Google Gemini API key
            reports_dir: Directory to save generated reports
            notification_service: Optional NotificationService for voice alerts
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.active_threads = []  # Track background report generation threads
        self.notification_service = notification_service  # Voice call notifications
        
        # Map detection categories to readable names
        self.category_names = {
            "fall": "Fall",
            "fire": "Fire",
            "distress": "Distress/Respiratory Issue",
            "violence_panic": "Violence/Panic/Crowd Issue",
            "severe_injury": "Severe Injury"
        }
        
    def generate_prompt(self, category, confidence, source_path=None, timestamp=None):
        """
        Generate a prompt for Gemini based on the detected category.
        
        Args:
            category: The detected situation category
            confidence: Confidence score of the detection
            source_path: Source video path
            timestamp: Detection timestamp
            
        Returns:
            A formatted prompt string
        """
        category_display = self.category_names.get(category, category)
        
        # Define priority levels and protocols based on situation type
        priority_config = {
            "fall": {
                "priority": "LOW",
                "severity": "🟢 LOW PRIORITY",
                "notification": "Email and SMS to on-site medical staff",
                "response_time": "15-30 minutes",
                "reason": "Falls typically require medical assessment but are not immediately life-threatening. Responders should check for injuries but can take measured approach."
            },
            "violence_panic": {
                "priority": "MEDIUM",
                "severity": "🟡 MEDIUM PRIORITY",
                "notification": "Email, SMS, and phone alert to security team",
                "response_time": "5-10 minutes",
                "reason": "Crowd incidents and violence can escalate quickly. Security response needed to prevent injuries and restore order. Situation is containable with prompt intervention."
            },
            "distress": {
                "priority": "HIGH",
                "severity": "🟠 HIGH PRIORITY",
                "notification": "Immediate phone call + SMS to medical emergency team",
                "response_time": "2-5 minutes",
                "reason": "Respiratory distress can rapidly deteriorate to cardiac arrest. Immediate medical intervention critical. Every minute without oxygen causes brain damage."
            },
            "severe_injury": {
                "priority": "CRITICAL",
                "severity": "🔴 CRITICAL",
                "notification": "Emergency phone call + SMS to paramedics + 911 dispatch",
                "response_time": "< 2 minutes",
                "reason": "Severe trauma requires immediate life-saving intervention. Blood loss, internal injuries, or head trauma can be fatal within minutes. Emergency medical services must respond immediately."
            },
            "fire": {
                "priority": "CRITICAL",
                "severity": "🔴 CRITICAL",
                "notification": "Immediate 911 call + building-wide evacuation alarm + fire department dispatch",
                "response_time": "< 2 minutes",
                "reason": "Fire spreads exponentially and produces toxic smoke. Every second counts to prevent loss of life and property. Immediate evacuation and fire suppression required."
            }
        }
        
        config = priority_config.get(category, priority_config["violence_panic"])
        
        source_info = f"from {source_path}" if source_path else "from surveillance camera"
        time_info = f"at {timestamp}" if timestamp else "just now"
        
        prompt = f"""You are an emergency response analyst providing a detailed incident report. 

**INCIDENT DETECTED:**
Type: {category_display.upper()}
Priority: {config['priority']}
Confidence: {confidence:.1%}
Location: {source_info}
Time: {time_info}

**YOUR TASK:**
Provide a comprehensive, actionable report describing EXACTLY what is happening in this image. This report will be sent to emergency responders and decision-makers.

**REPORT STRUCTURE - Be specific and detailed:**

1. IMMEDIATE SITUATION (2-3 sentences):
   - Describe what you see happening RIGHT NOW
   - Exact positions, postures, and conditions of all people visible
   - Environmental context and setting

2. OBSERVABLE DETAILS (2-3 sentences):
   - Number of people involved and their states
   - Specific injuries, hazards, or danger signs visible
   - Any objects, obstacles, or environmental factors relevant to response

3. ASSESSMENT (1-2 sentences):
   - What likely just occurred or is currently occurring
   - Immediate risks to life or safety

4. RECOMMENDED ACTION (1-2 sentences):
   - Specific steps responders should take immediately upon arrival
   - Any special equipment, personnel, or precautions needed

**WRITING REQUIREMENTS:**
- Use present tense (this is happening NOW)
- Be factual and specific - avoid speculation
- Include measurable details (distances, numbers, positions)
- Write clearly for someone who cannot see the image
- Keep total response to 8-12 sentences
- Use professional emergency services language

Provide your detailed emergency response report now."""

        return prompt
    
    def save_frame(self, frame, source_path, category):
        """
        Save the detection frame for reference.
        
        Args:
            frame: The frame to save (numpy array)
            source_path: Original video source path
            category: Detection category
            
        Returns:
            Path to saved frame
        """
        # Generate filename from source
        if source_path and isinstance(source_path, str):
            video_name = Path(source_path).stem
        else:
            video_name = f"detection_{int(time.time())}"
        
        frame_filename = f"{video_name}_{category}_frame.jpg"
        frame_path = self.reports_dir / frame_filename
        
        cv2.imwrite(str(frame_path), frame)
        return frame_path
    
    def _generate_report_sync(self, frame, category, confidence, source_path=None):
        """
        Synchronous report generation (internal method called by thread).
        
        Args:
            frame: The frame to analyze (numpy array)
            category: Detection category
            confidence: Confidence score
            source_path: Original video source path
        """
        try:
            # Save frame temporarily for Gemini
            frame_path = self.save_frame(frame, source_path, category)
            
            # Generate timestamp
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Generate prompt with context
            prompt = self.generate_prompt(category, confidence, source_path, timestamp)
            
            # Load image for Gemini
            image = genai.upload_file(str(frame_path))
            
            # Generate content with Gemini
            print(f"[GEMINI] Analyzing {category} situation...")
            response = self.model.generate_content([prompt, image])
            report_text = response.text
            
            # Create report filename
            if source_path and isinstance(source_path, str):
                video_name = Path(source_path).stem
            else:
                video_name = f"detection_{int(time.time())}"
            
            report_filename = f"{video_name}_{category}_report.txt"
            report_path = self.reports_dir / report_filename
            
            # Determine severity and notification protocol based on category
            priority_map = {
                "fall": ("🟢 LOW PRIORITY", "Email + SMS to medical staff", "15-30 minutes"),
                "violence_panic": ("🟡 MEDIUM PRIORITY", "Email + SMS + Phone to security", "5-10 minutes"),
                "distress": ("🟠 HIGH PRIORITY", "Immediate phone + SMS to medical team", "2-5 minutes"),
                "severe_injury": ("🔴 CRITICAL", "Emergency phone + SMS + 911 dispatch", "< 2 minutes"),
                "fire": ("🔴 CRITICAL", "911 call + Evacuation alarm + Fire dept", "< 2 minutes")
            }
            
            severity, notification_method, response_time = priority_map.get(
                category, 
                ("🟡 MEDIUM PRIORITY", "Email + SMS", "5-10 minutes")
            )
            
            # Save report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"🚨 VITALSIGHT EMERGENCY DETECTION REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"ALERT TYPE: {self.category_names.get(category, category).upper()}\n")
                f.write(f"Severity Level: {severity}\n")
                f.write(f"Detection Confidence: {confidence:.1%}\n")
                f.write(f"Source: {source_path or 'Unknown'}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Evidence Frame: {frame_path.name}\n")
                f.write("\n" + "=" * 80 + "\n")
                f.write("SITUATION REPORT:\n")
                f.write("=" * 80 + "\n\n")
                f.write(report_text)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("NOTIFICATION PROTOCOL:\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Notification Method: {notification_method}\n")
                f.write(f"Expected Response Time: {response_time}\n")
                f.write(f"Authorities to be contacted: Emergency Services, Security Team, Medical Response\n")
                f.write("\n" + "=" * 80 + "\n")
            
            print(f"[GEMINI] ✓ Report saved to: {report_path}")
            
            # Send voice call notification for CRITICAL situations
            if self.notification_service:
                print(f"[NOTIFICATION] Checking if voice alert needed for {category}...")
                notification_result = self.notification_service.send_alert(
                    category=category,
                    report_text=report_text,
                    source_path=source_path
                )
                if notification_result.get('call'):
                    print(f"[NOTIFICATION] ✓ Voice call initiated: {notification_result.get('call_sid')}")
                else:
                    print(f"[NOTIFICATION] Voice call not needed: {notification_result.get('reason', 'N/A')}")
            
            # Clean up uploaded file from Gemini
            try:
                genai.delete_file(image.name)
            except:
                pass
            
        except Exception as e:
            error_msg = f"Error generating Gemini report: {str(e)}"
            print(f"[GEMINI ERROR] {error_msg}")
            
            # Save error report
            if source_path and isinstance(source_path, str):
                video_name = Path(source_path).stem
            else:
                video_name = f"detection_{int(time.time())}"
            
            error_report_path = self.reports_dir / f"{video_name}_{category}_error.txt"
            with open(error_report_path, 'w', encoding='utf-8') as f:
                f.write(f"Failed to generate report for {category}\n")
                f.write(f"Error: {error_msg}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def generate_report_async(self, frame, category, confidence, source_path=None):
        """
        Generate a detailed report using Gemini VLM asynchronously.
        This method returns immediately and runs report generation in background.
        
        Args:
            frame: The frame to analyze (numpy array)
            category: Detection category
            confidence: Confidence score
            source_path: Original video source path
        """
        # Create a copy of the frame to avoid issues with frame being modified
        frame_copy = frame.copy()
        
        # Start background thread for report generation
        thread = threading.Thread(
            target=self._generate_report_sync,
            args=(frame_copy, category, confidence, source_path),
            daemon=True
        )
        thread.start()
        self.active_threads.append(thread)
        
        print(f"[GEMINI] Report generation started in background for {category}")
    
    def wait_for_all_reports(self):
        """
        Wait for all background report generation threads to complete.
        Call this before exiting the program to ensure all reports are saved.
        """
        if self.active_threads:
            print(f"[GEMINI] Waiting for {len(self.active_threads)} report(s) to complete...")
            for thread in self.active_threads:
                thread.join()
            self.active_threads.clear()
            print("[GEMINI] All reports completed.")

