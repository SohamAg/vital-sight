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
        
    def generate_prompt(self, category, confidence):
        """
        Generate a prompt for Gemini based on the detected category.
        
        Args:
            category: The detected situation category
            confidence: Confidence score of the detection
            
        Returns:
            A formatted prompt string
        """
        category_display = self.category_names.get(category, category)
        
        prompt = f"""You are a security and safety analysis expert. Our AI detection system has identified a potential {category_display} situation in this image with a confidence of {confidence:.2%}.

**Your task:** Assume our detection is correct and provide a brief, factual report describing what you observe in the image that confirms or explains this situation.

**Focus on:**
1. Visual evidence supporting the detection (body positions, postures, environmental factors)
2. Number of people involved
3. Specific actions or states visible in the frame
4. Any environmental hazards or conditions present
5. Urgency level and immediate concerns

**Format:**
- Keep it concise (3-5 sentences)
- Use objective, professional language
- Focus on observable facts
- Provide actionable insights

Please analyze the image and provide your report now."""

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
            
            # Generate prompt
            prompt = self.generate_prompt(category, confidence)
            
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
            
            # Save report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"VITALSIGHT AI DETECTION REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Category: {self.category_names.get(category, category)}\n")
                f.write(f"Confidence: {confidence:.2%}\n")
                f.write(f"Source: {source_path or 'Unknown'}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Frame Image: {frame_path.name}\n")
                f.write("\n" + "-" * 80 + "\n")
                f.write("GEMINI VLM ANALYSIS:\n")
                f.write("-" * 80 + "\n\n")
                f.write(report_text)
                f.write("\n\n" + "=" * 80 + "\n")
            
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

