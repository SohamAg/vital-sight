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
    
    def __init__(self, api_key, reports_dir="data/demo_reports"):
        """
        Initialize the Gemini reporter.
        
        Args:
            api_key: Google Gemini API key
            reports_dir: Directory to save generated reports
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.active_threads = []  # Track background report generation threads
        
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
        
        # Define notification protocols and urgency
        if category in ["fire", "distress", "severe_injury"]:
            severity = "CRITICAL"
            notification = "Authorities will be notified via EMAIL, SMS, and IMMEDIATE PHONE CALL"
            reason = {
                "fire": "Fire situations are highly imminent and can rapidly escalate, causing widespread damage and loss of life. Immediate emergency response is required.",
                "distress": "Respiratory distress can lead to cardiac arrest within minutes. Immediate medical intervention is critical for survival.",
                "severe_injury": "Severe injuries require immediate medical attention to prevent death or permanent disability. Every second counts in trauma cases."
            }.get(category, "Immediate response required.")
        else:  # fall, violence_panic
            severity = "MEDIUM"
            notification = "Authorities will be notified via EMAIL and SMS alerts"
            reason = {
                "fall": "Falls may result in injuries requiring medical assessment, but immediate life threat is typically lower. Response should be prompt but controlled.",
                "violence_panic": "Crowd incidents require security response to prevent escalation, but can be managed with coordinated intervention."
            }.get(category, "Prompt response needed.")
        
        source_info = f"from source '{source_path}'" if source_path else ""
        time_info = f"at {timestamp}" if timestamp else ""
        
        prompt = f"""You are an emergency response analyst. A {category_display.upper()} situation has been detected in this frame {source_info} {time_info} with {confidence:.1%} confidence.

**Your task:** Describe EXACTLY what is happening in this frame as if you were narrating it to emergency responders over the phone or writing it in an urgent email. Be specific, clear, and factual.

**Describe:**
1. What you see happening to the person(s) - their exact position, posture, condition
2. The immediate environment and any hazards visible
3. Number of people involved and their states
4. Any visible injuries, danger signs, or critical elements
5. What likely just happened or is currently happening

**Critical Guidelines:**
- Write in present tense as if describing a live situation
- Use plain, direct language (this may be read aloud to responders)
- Be specific about locations, positions, and conditions
- Mention any time-sensitive factors
- Keep it focused and under 5 sentences

**NOTIFICATION PROTOCOL:**
Severity: {severity}
Protocol: {notification}
Reason: {reason}

Provide your emergency situation report now."""

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
            
            # Determine severity and notification protocol
            if category in ["fire", "distress", "severe_injury"]:
                severity = "🔴 CRITICAL"
                notification_method = "EMAIL + SMS + IMMEDIATE PHONE CALL"
                response_time = "IMMEDIATE (< 2 minutes)"
            else:
                severity = "🟡 MEDIUM"
                notification_method = "EMAIL + SMS"
                response_time = "PROMPT (< 10 minutes)"
            
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

