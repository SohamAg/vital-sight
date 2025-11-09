"""
MCP Server Interface for Emergency Alert Communication
Handles phone calls and email alerts via mock MCP Server API
"""

import requests
import time
from datetime import datetime
from typing import Dict, Optional
import json
from config import MCP_SERVER_CONFIG, ALERT_ROUTING


class MCPServerInterface:
    """
    Interface for communicating with MCP Server for emergency alerts.
    Implements phone call and email notification capabilities.
    """
    
    def __init__(self):
        """Initialize MCP Server Interface"""
        self.phone_call_url = MCP_SERVER_CONFIG['phone_call_url']
        self.email_url = MCP_SERVER_CONFIG['email_url']
        self.timeout = MCP_SERVER_CONFIG['timeout']
        self.retry_attempts = MCP_SERVER_CONFIG['retry_attempts']
        self.alert_log = []
        
    def send_mcp_alert(self, action_type: str, frame_id: int = 0, 
                       additional_info: Optional[Dict] = None) -> bool:
        """
        Send alert via MCP Server based on detected action type.
        
        Args:
            action_type: Type of emergency action detected
            frame_id: Frame number where action was detected
            additional_info: Optional additional context
            
        Returns:
            bool: True if alert sent successfully, False otherwise
        """
        if action_type not in ALERT_ROUTING:
            print(f"[MCP] Warning: Unknown action type '{action_type}'")
            return False
            
        routing_info = ALERT_ROUTING[action_type]
        action = routing_info['action']
        target = routing_info['target']
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare alert data
        alert_data = {
            'action_type': action_type,
            'timestamp': timestamp,
            'frame_id': frame_id,
            'target': target
        }
        
        if additional_info:
            alert_data.update(additional_info)
        
        # Route to appropriate communication method
        if action == 'phone_call':
            success = self._send_phone_call(target, action_type, alert_data)
        elif action == 'email':
            success = self._send_email(target, action_type, alert_data)
        else:
            print(f"[MCP] Error: Unknown action '{action}'")
            return False
        
        # Log the alert
        if success:
            self.alert_log.append({
                'timestamp': timestamp,
                'action_type': action_type,
                'communication_method': action,
                'target': target,
                'status': 'SUCCESS'
            })
        
        return success
    
    def _send_phone_call(self, phone_number: str, action_type: str, 
                         alert_data: Dict) -> bool:
        """
        Send phone call alert via MCP Server.
        
        Args:
            phone_number: Target phone number
            action_type: Type of emergency
            alert_data: Alert details
            
        Returns:
            bool: Success status
        """
        print(f"\n{'='*60}")
        print(f"[MCP] PHONE CALL ALERT")
        print(f"{'='*60}")
        print(f"Emergency Type: {action_type}")
        print(f"Target Number: {phone_number}")
        print(f"Timestamp: {alert_data['timestamp']}")
        print(f"Frame ID: {alert_data['frame_id']}")
        
        # Mock MCP Server API call
        try:
            payload = {
                'phone_number': phone_number,
                'message': f"EMERGENCY ALERT: {action_type} detected at {alert_data['timestamp']}",
                'priority': 'HIGH',
                'alert_data': alert_data
            }
            
            # In production, this would make actual API call:
            # response = requests.post(self.phone_call_url, json=payload, timeout=self.timeout)
            # response.raise_for_status()
            
            # Mock successful response
            print(f"[MCP] API Endpoint: {self.phone_call_url}")
            print(f"[MCP] Payload: {json.dumps(payload, indent=2)}")
            print(f"[MCP] Status: CALL INITIATED")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"[MCP] Error sending phone call: {str(e)}")
            print(f"{'='*60}\n")
            return False
    
    def _send_email(self, email_address: str, action_type: str, 
                    alert_data: Dict) -> bool:
        """
        Send email alert via MCP Server.
        
        Args:
            email_address: Target email address
            action_type: Type of emergency
            alert_data: Alert details
            
        Returns:
            bool: Success status
        """
        print(f"\n{'='*60}")
        print(f"[MCP] EMAIL ALERT")
        print(f"{'='*60}")
        print(f"Emergency Type: {action_type}")
        print(f"Target Email: {email_address}")
        print(f"Timestamp: {alert_data['timestamp']}")
        print(f"Frame ID: {alert_data['frame_id']}")
        
        # Mock MCP Server API call
        try:
            payload = {
                'email': email_address,
                'subject': f"EMERGENCY ALERT: {action_type}",
                'body': f"""
Emergency Detection System Alert

Type: {action_type}
Timestamp: {alert_data['timestamp']}
Frame ID: {alert_data['frame_id']}

This is an automated alert from the VitalSight Emergency Detection System.
Immediate attention may be required.

Alert Details:
{json.dumps(alert_data, indent=2)}
                """,
                'priority': 'HIGH',
                'alert_data': alert_data
            }
            
            # In production, this would make actual API call:
            # response = requests.post(self.email_url, json=payload, timeout=self.timeout)
            # response.raise_for_status()
            
            # Mock successful response
            print(f"[MCP] API Endpoint: {self.email_url}")
            print(f"[MCP] Subject: {payload['subject']}")
            print(f"[MCP] Status: EMAIL SENT")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"[MCP] Error sending email: {str(e)}")
            print(f"{'='*60}\n")
            return False
    
    def get_alert_log(self) -> list:
        """
        Get the log of all alerts sent.
        
        Returns:
            list: Alert log entries
        """
        return self.alert_log
    
    def clear_alert_log(self):
        """Clear the alert log"""
        self.alert_log = []


# Convenience function for direct use
def send_mcp_alert(action_type: str, frame_id: int = 0, 
                   additional_info: Optional[Dict] = None) -> bool:
    """
    Convenience function to send MCP alert without instantiating class.
    
    Args:
        action_type: Type of emergency action detected
        frame_id: Frame number where action was detected
        additional_info: Optional additional context
        
    Returns:
        bool: Success status
    """
    mcp = MCPServerInterface()
    return mcp.send_mcp_alert(action_type, frame_id, additional_info)
