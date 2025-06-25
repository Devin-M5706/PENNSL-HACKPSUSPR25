from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
import base64
import time
import re
from asl_dataset import get_letter_description, match_hand_to_letter
import math
from collections import deque
from typing import Tuple, Optional
import traceback

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe with optimized settings
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only detect one hand for faster processing
    model_complexity=0,  # Use light model for faster processing
    min_detection_confidence=0.5,  # Higher threshold to reduce false positives
    min_tracking_confidence=0.5
)

# Temporal smoothing parameters
CONFIDENCE_THRESHOLD = 0.5  # Higher threshold for more stable predictions
HISTORY_SIZE = 3  # Slightly larger history for better stability
letter_history = deque(maxlen=HISTORY_SIZE)
confidence_history = deque(maxlen=HISTORY_SIZE)
last_detected_letter = None
last_processed_time = 0
PROCESSING_INTERVAL = 0.1  # Minimum time between processing (100ms)

# Drawing specifications
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)  # Convert frozenset to list
LANDMARK_DRAWING_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
CONNECTION_DRAWING_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)

def get_finger_states(hand_landmarks):
    """Get the state of each finger with improved detection for closed positions"""
    # Get palm orientation with improved accuracy
    wrist = hand_landmarks.landmark[0]
    index_base = hand_landmarks.landmark[5]
    pinky_base = hand_landmarks.landmark[17]
    
    # Calculate palm normal using cross product for better accuracy
    palm_vector1 = np.array([index_base.x - wrist.x,
                            index_base.y - wrist.y,
                            index_base.z - wrist.z])
    palm_vector2 = np.array([pinky_base.x - wrist.x,
                            pinky_base.y - wrist.y,
                            pinky_base.z - wrist.z])
    palm_normal = np.cross(palm_vector1, palm_vector2)
    palm_normal = palm_normal / np.linalg.norm(palm_normal)
    
    # Calculate palm center with more points for better accuracy
    palm_points = [0, 1, 5, 9, 13, 17]  # Include more points
    palm_center = {
        'x': sum(hand_landmarks.landmark[i].x for i in palm_points) / len(palm_points),
        'y': sum(hand_landmarks.landmark[i].y for i in palm_points) / len(palm_points),
        'z': sum(hand_landmarks.landmark[i].z for i in palm_points) / len(palm_points)
    }
    
    # Define finger joints with more precise tracking
    finger_joints = {
        'thumb': [1, 2, 4],    # CMC, MCP, IP, TIP
        'index': [5, 6, 8],    # MCP, PIP, DIP, TIP
        'middle': [9, 10, 12],
        'ring': [13, 14, 16],
        'pinky': [17, 18, 20]
    }
    
    finger_states = {}
    finger_spreads = {}
    
    # Calculate finger states with improved accuracy
    for finger, joints in finger_joints.items():
        base = hand_landmarks.landmark[joints[0]]
        middle = hand_landmarks.landmark[joints[1]]
        tip = hand_landmarks.landmark[joints[2]]
        
        state = calculate_finger_state(
            {'x': base.x, 'y': base.y, 'z': base.z},
            {'x': middle.x, 'y': middle.y, 'z': middle.z},
            {'x': tip.x, 'y': tip.y, 'z': tip.z},
            palm_center,
            palm_normal
        )
        finger_states[finger] = state
    
    # Calculate finger spreads with improved accuracy
    adjacent_fingers = [
        ('index', 'middle'),
        ('middle', 'ring'),
        ('ring', 'pinky')
    ]
    
    for f1, f2 in adjacent_fingers:
        f1_tip = finger_joints[f1][2]  # Tip index
        f2_tip = finger_joints[f2][2]  # Tip index
        f1_base = finger_joints[f1][0]  # Base index
        f2_base = finger_joints[f2][0]  # Base index
        
        # Calculate spread between adjacent fingers using both tip and base
        tip_spread = calculate_distance(
            {'x': hand_landmarks.landmark[f1_tip].x, 'y': hand_landmarks.landmark[f1_tip].y, 'z': hand_landmarks.landmark[f1_tip].z},
            {'x': hand_landmarks.landmark[f2_tip].x, 'y': hand_landmarks.landmark[f2_tip].y, 'z': hand_landmarks.landmark[f2_tip].z}
        )
        base_spread = calculate_distance(
            {'x': hand_landmarks.landmark[f1_base].x, 'y': hand_landmarks.landmark[f1_base].y, 'z': hand_landmarks.landmark[f1_base].z},
            {'x': hand_landmarks.landmark[f2_base].x, 'y': hand_landmarks.landmark[f2_base].y, 'z': hand_landmarks.landmark[f2_base].z}
        )
        
        # Normalize by palm size
        palm_size = calculate_distance(
            {'x': hand_landmarks.landmark[0].x, 'y': hand_landmarks.landmark[0].y, 'z': hand_landmarks.landmark[0].z},
            {'x': hand_landmarks.landmark[5].x, 'y': hand_landmarks.landmark[5].y, 'z': hand_landmarks.landmark[5].z}
        )
        
        # Use both tip and base spread for more accurate classification
        spread_ratio = (tip_spread + base_spread) / (2 * palm_size)
        spread_name = f"{f1}_{f2}"
        
        # Classify spread with more categories
        if spread_ratio < 0.15:
            finger_spreads[spread_name] = 'touch'
        elif spread_ratio < 0.25:
            finger_spreads[spread_name] = 'close'
        elif spread_ratio < 0.35:
            finger_spreads[spread_name] = 'slight_spread'
        elif spread_ratio < 0.45:
            finger_spreads[spread_name] = 'spread'
        else:
            finger_spreads[spread_name] = 'wide_spread'
    
    return finger_states, finger_spreads

def apply_temporal_smoothing(letter, confidence):
    """Apply temporal smoothing to stabilize predictions"""
    letter_history.append(letter)
    confidence_history.append(confidence)
    
    if len(letter_history) < HISTORY_SIZE:
        return letter, confidence
    
    # Calculate smoothed confidence
    smoothed_confidence = sum(confidence_history) / len(confidence_history)
    
    # Get most common letter from history
    from collections import Counter
    letter_counts = Counter(letter_history)
    most_common_letter = letter_counts.most_common(1)[0][0]
    
    # Only return prediction if confidence is above threshold
    if smoothed_confidence >= CONFIDENCE_THRESHOLD:
        return most_common_letter, smoothed_confidence
    else:
        return None, 0.0

def calculate_angle_3d(p1, p2, p3):
    """Calculate angle between three points in 3D space with improved accuracy."""
    v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
    
    # Normalize vectors for more stable angle calculation
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate angle using dot product of normalized vectors
    dot_product = np.dot(v1_norm, v2_norm)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return math.degrees(angle)

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points in 3D space."""
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)

def normalize_coordinates(landmarks, palm_center):
    """Normalize landmark coordinates relative to palm center."""
    normalized = []
    for landmark in landmarks:
        normalized.append({
            'x': landmark['x'] - palm_center['x'],
            'y': landmark['y'] - palm_center['y'],
            'z': landmark['z'] - palm_center['z']
        })
    return normalized

def calculate_finger_state(base, middle, tip, palm_center, palm_normal):
    """Calculate finger state with improved accuracy."""
    # Normalize coordinates relative to palm center
    base_norm = {
        'x': base['x'] - palm_center['x'],
        'y': base['y'] - palm_center['y'],
        'z': base['z'] - palm_center['z']
    }
    middle_norm = {
        'x': middle['x'] - palm_center['x'],
        'y': middle['y'] - palm_center['y'],
        'z': middle['z'] - palm_center['z']
    }
    tip_norm = {
        'x': tip['x'] - palm_center['x'],
        'y': tip['y'] - palm_center['y'],
        'z': tip['z'] - palm_center['z']
    }

    # Calculate angles between joints
    base_middle_angle = calculate_angle_3d(base_norm, middle_norm, tip_norm)
    
    # Calculate finger direction vector
    finger_vector = np.array([tip_norm['x'] - base_norm['x'],
                            tip_norm['y'] - base_norm['y'],
                            tip_norm['z'] - base_norm['z']])
    finger_vector = finger_vector / np.linalg.norm(finger_vector)
    
    # Calculate angle between finger and palm normal
    palm_angle = np.arccos(np.clip(np.dot(finger_vector, palm_normal), -1.0, 1.0))
    palm_angle_deg = math.degrees(palm_angle)

    # Calculate distances for closed fist detection
    tip_to_palm_dist = math.sqrt(sum((tip_norm[k])**2 for k in ['x', 'y', 'z']))
    middle_to_palm_dist = math.sqrt(sum((middle_norm[k])**2 for k in ['x', 'y', 'z']))
    
    # Enhanced state detection with more states
    if tip_to_palm_dist < middle_to_palm_dist * 0.7:  # Tighter threshold for closed
        return 'down'
    elif tip_to_palm_dist < middle_to_palm_dist * 0.9:  # Slightly curved
        return 'slightly_curved'
    elif base_middle_angle > 160:  # More precise angle for straight
        if palm_angle_deg < 45:  # More precise angle for "up"
            return 'up'
        elif palm_angle_deg > 135:  # More precise angle for "down"
            return 'down'
        else:
            return 'side'
    elif base_middle_angle > 90:  # Slightly curved
        return 'slightly_curved'
    else:  # Fully curved
        return 'curved'

def get_hand_bounds(hand_landmarks, image_width, image_height):
    """Calculate bounding box for detected hand"""
    x_min = y_min = float('inf')
    x_max = y_max = float('-inf')
    
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
    
    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_width, x_max + padding)
    y_max = min(image_height, y_max + padding)
    
    return {
        'x_min': x_min,
        'y_min': y_min,
        'x_max': x_max,
        'y_max': y_max
    }

def get_hand_landmarks_info(landmarks):
    """Extract detailed information about hand landmarks."""
    info = []
    for i, landmark in enumerate(landmarks.landmark):
        info.append({
            'id': i,
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility
        })
    return info

def validate_image_data(image_data: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Validate and decode base64 image data."""
    try:
        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64 string
        decoded_data = base64.b64decode(image_data)
        return decoded_data, None
    except Exception as e:
        return None, f"Invalid image data: {str(e)}"

def preprocess_image(frame):
    """Enhanced image preprocessing for better hand detection"""
    try:
        # Convert to RGB first for consistent color processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if too large (keep aspect ratio)
        max_dimension = 640
        height, width = frame_rgb.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            frame_rgb = cv2.resize(frame_rgb, (int(width * scale), int(height * scale)))
        
        # Basic preprocessing - keep it simple for faster processing
        frame_rgb = cv2.GaussianBlur(frame_rgb, (5, 5), 0)
        
        # Enhance contrast
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        frame_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Increase brightness slightly
        frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.3, beta=15)
        
        return frame_rgb
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return frame

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points in 3D space."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
    
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate angle using dot product
    dot_product = np.dot(v1_norm, v2_norm)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return math.degrees(angle)

def calculate_finger_spread(p1, p2):
    """Calculate distance between two points."""
    return np.linalg.norm(p1 - p2)

def find_best_matching_letter(finger_states, finger_spreads):
    """Match finger states and spreads to ASL letters."""
    letter_patterns = {
        'A': {
            'fingers': {
                'thumb': 'side',      # Thumb out to side
                'index': 'down',      # All fingers in fist
                'middle': 'down',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'touch',
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'B': {
            'fingers': {
                'thumb': 'down',      # Thumb across palm
                'index': 'up',        # All fingers straight up
                'middle': 'up',
                'ring': 'up',
                'pinky': 'up'
            },
            'spreads': {
                'index_middle': 'touch',  # Fingers together
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'C': {
            'fingers': {
                'thumb': 'curved',    # Curved C shape
                'index': 'curved',
                'middle': 'curved',
                'ring': 'curved',
                'pinky': 'curved'
            },
            'spreads': {
                'index_middle': 'slight_spread',
                'middle_ring': 'slight_spread',
                'ring_pinky': 'slight_spread'
            }
        },
        'D': {
            'fingers': {
                'thumb': 'up',        # Thumb up
                'index': 'up',        # Index up
                'middle': 'down',     # Others curved in
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'spread',
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'E': {
            'fingers': {
                'thumb': 'curved',    # Thumb curved in
                'index': 'curved',    # All fingers curved in
                'middle': 'curved',
                'ring': 'curved',
                'pinky': 'curved'
            },
            'spreads': {
                'index_middle': 'touch',
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'F': {
            'fingers': {
                'thumb': 'side',      # Thumb and index touch
                'index': 'curved',    # Other fingers up
                'middle': 'up',
                'ring': 'up',
                'pinky': 'up'
            },
            'spreads': {
                'index_middle': 'spread',
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'G': {
            'fingers': {
                'thumb': 'side',      # Thumb out straight
                'index': 'side',      # Index pointing sideways
                'middle': 'down',     # Others down
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'spread',
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'H': {
            'fingers': {
                'thumb': 'side',      # Thumb out
                'index': 'side',      # Index and middle out
                'middle': 'side',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'touch',
                'middle_ring': 'spread',
                'ring_pinky': 'touch'
            }
        },
        'I': {
            'fingers': {
                'thumb': 'side',      # Just pinky up
                'index': 'down',
                'middle': 'down',
                'ring': 'down',
                'pinky': 'up'
            },
            'spreads': {
                'index_middle': 'touch',
                'middle_ring': 'touch',
                'ring_pinky': 'spread'
            }
        },
        'K': {
            'fingers': {
                'thumb': 'up',        # Index and middle up in V
                'index': 'up',
                'middle': 'up',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'spread',
                'middle_ring': 'spread',
                'ring_pinky': 'touch'
            }
        },
        'L': {
            'fingers': {
                'thumb': 'side',      # L shape
                'index': 'up',
                'middle': 'down',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'spread',
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'M': {
            'fingers': {
                'thumb': 'down',      # Thumb between fingers
                'index': 'down',
                'middle': 'down',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'touch',
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'N': {
            'fingers': {
                'thumb': 'down',      # Similar to M
                'index': 'down',
                'middle': 'down',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'touch',
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'O': {
            'fingers': {
                'thumb': 'curved',    # All fingers tightly curved to make O
                'index': 'curved',
                'middle': 'curved',
                'ring': 'curved',
                'pinky': 'curved'
            },
            'spreads': {
                'index_middle': 'touch',
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'P': {
            'fingers': {
                'thumb': 'side',      # Index pointing down
                'index': 'down',
                'middle': 'down',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'spread',
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'Q': {
            'fingers': {
                'thumb': 'side',      # Index pointing down
                'index': 'down',
                'middle': 'down',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'spread',
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'R': {
            'fingers': {
                'thumb': 'side',      # Index and middle crossed
                'index': 'up',
                'middle': 'up',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'cross',
                'middle_ring': 'spread',
                'ring_pinky': 'touch'
            }
        },
        'S': {
            'fingers': {
                'thumb': 'curved',    # Fist with thumb in front
                'index': 'down',
                'middle': 'down',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'touch',
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'T': {
            'fingers': {
                'thumb': 'up',        # Thumb between index and middle
                'index': 'curved',
                'middle': 'curved',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'spread',
                'middle_ring': 'spread',
                'ring_pinky': 'touch'
            }
        },
        'U': {
            'fingers': {
                'thumb': 'side',      # Index and middle together
                'index': 'up',
                'middle': 'up',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'touch',
                'middle_ring': 'spread',
                'ring_pinky': 'touch'
            }
        },
        'V': {
            'fingers': {
                'thumb': 'side',      # Peace sign
                'index': 'up',
                'middle': 'up',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'spread',
                'middle_ring': 'spread',
                'ring_pinky': 'touch'
            }
        },
        'W': {
            'fingers': {
                'thumb': 'side',      # Three fingers up
                'index': 'up',
                'middle': 'up',
                'ring': 'up',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'spread',
                'middle_ring': 'spread',
                'ring_pinky': 'spread'
            }
        },
        'X': {
            'fingers': {
                'thumb': 'side',      # Index hook
                'index': 'curved',
                'middle': 'down',
                'ring': 'down',
                'pinky': 'down'
            },
            'spreads': {
                'index_middle': 'spread',
                'middle_ring': 'touch',
                'ring_pinky': 'touch'
            }
        },
        'Y': {
            'fingers': {
                'thumb': 'side',      # Thumb and pinky out
                'index': 'down',
                'middle': 'down',
                'ring': 'down',
                'pinky': 'up'
            },
            'spreads': {
                'index_middle': 'touch',
                'middle_ring': 'touch',
                'ring_pinky': 'spread'
            }
        }
    }

    best_match = {'letter': None, 'confidence': 0.0}
    
    for letter, pattern in letter_patterns.items():
        # Calculate finger position match
        finger_matches = 0
        total_fingers = len(pattern['fingers'])
        
        for finger, expected_state in pattern['fingers'].items():
            if finger in finger_states:
                actual_state = finger_states[finger]
                if actual_state == expected_state:
                    finger_matches += 1
                elif (expected_state == 'down' and actual_state == 'curved') or \
                     (expected_state == 'curved' and actual_state == 'down'):
                    finger_matches += 0.8
                elif (expected_state == 'up' and actual_state == 'curved') or \
                     (expected_state == 'curved' and actual_state == 'up'):
                    finger_matches += 0.7
                elif (expected_state == 'side' and actual_state == 'curved') or \
                     (expected_state == 'curved' and actual_state == 'side'):
                    finger_matches += 0.6
                elif (expected_state == 'side' and actual_state == 'up') or \
                     (expected_state == 'up' and actual_state == 'side'):
                    finger_matches += 0.5
        
        # Calculate spread match
        spread_matches = 0
        total_spreads = len(pattern['spreads'])
        
        for spread, expected_state in pattern['spreads'].items():
            if spread in finger_spreads:
                actual_state = finger_spreads[spread]
                if actual_state == expected_state:
                    spread_matches += 1
                elif (expected_state == 'close' and actual_state == 'slight_spread') or \
                     (expected_state == 'slight_spread' and actual_state == 'close'):
                    spread_matches += 0.8
                elif (expected_state == 'spread' and actual_state == 'slight_spread') or \
                     (expected_state == 'slight_spread' and actual_state == 'spread'):
                    spread_matches += 0.7
                elif (expected_state == 'touch' and actual_state == 'close') or \
                     (expected_state == 'close' and actual_state == 'touch'):
                    spread_matches += 0.9
                elif (expected_state == 'cross' and actual_state == 'touch') or \
                     (expected_state == 'touch' and actual_state == 'cross'):
                    spread_matches += 0.6
        
        # Calculate overall confidence
        finger_confidence = finger_matches / total_fingers if total_fingers > 0 else 0
        spread_confidence = spread_matches / total_spreads if total_spreads > 0 else 0
        
        # Letter-specific confidence adjustments
        if letter == 'B':  # B requires straight fingers
            if any(state == 'curved' for state in finger_states.values()):
                overall_confidence *= 0.5  # Heavily penalize curved fingers for B
            overall_confidence = (finger_confidence * 0.9) + (spread_confidence * 0.1)  # Emphasize finger position
        elif letter == 'O':  # O requires curved fingers
            if any(state == 'up' for state in finger_states.values()):
                overall_confidence *= 0.5  # Heavily penalize straight fingers for O
            overall_confidence = (finger_confidence * 0.7) + (spread_confidence * 0.3)
        elif letter in ['F', 'W']:  # Letters with multiple straight fingers
            overall_confidence = (finger_confidence * 0.8) + (spread_confidence * 0.2)
        elif letter in ['C', 'E']:  # Letters with specific curved shapes
            overall_confidence = (finger_confidence * 0.6) + (spread_confidence * 0.4)
        elif letter in ['K', 'P', 'Q']:  # Letters with specific angles
            overall_confidence = (finger_confidence * 0.7) + (spread_confidence * 0.3)
        elif letter in ['R', 'X']:  # Letters with crossed or hooked fingers
            overall_confidence = (finger_confidence * 0.75) + (spread_confidence * 0.25)
        else:
            overall_confidence = (finger_confidence * 0.7) + (spread_confidence * 0.3)
        
        # Letter-specific thresholds based on complexity
        base_threshold = 0.75
        if letter in ['A', 'S', 'T', 'L', 'O']:  # Simpler letters
            threshold = 0.65
        elif letter in ['R', 'X', 'M', 'N']:  # Complex letters
            threshold = 0.80
        else:
            threshold = base_threshold
        
        if overall_confidence > threshold and overall_confidence > best_match['confidence']:
            best_match = {'letter': letter, 'confidence': overall_confidence}
    
    return best_match

@app.route('/translate', methods=['POST', 'OPTIONS'])
def translate():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        current_time = time.time()
        if current_time - last_processed_time < PROCESSING_INTERVAL:
            return jsonify({
                'letter': last_detected_letter,
                'confidence': confidence_history[-1] if confidence_history else 0
            })
            
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize image for faster processing
        image = cv2.resize(image, (640, 480))
        
        # Process image
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_hand_landmarks:
            last_processed_time = current_time
            return jsonify({'letter': '?', 'confidence': 0})
            
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get finger states and spreads
        finger_states, finger_spreads = get_finger_states(hand_landmarks)
        
        # Match to letter
        letter, confidence = match_hand_to_letter(finger_states, finger_spreads)
        
        # Apply temporal smoothing
        letter, confidence = apply_temporal_smoothing(letter, confidence)
        
        last_detected_letter = letter
        last_processed_time = current_time
        
        return jsonify({
            'letter': letter if letter else '?',
            'confidence': confidence
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

def calculate_finger_states(landmarks):
    """Calculate finger states with more lenient thresholds"""
    finger_states = {}
    
    # Define more lenient thresholds
    CLOSED_THRESHOLD = 0.15  # Increased from 0.1
    OPEN_THRESHOLD = 0.3     # Increased from 0.2
    
    # Thumb
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    
    # Calculate thumb angle with more lenient thresholds
    thumb_angle = calculate_angle(thumb_tip, thumb_ip, thumb_mcp)
    if thumb_angle < 30:  # More lenient angle threshold
        finger_states['thumb'] = 'down'
    elif thumb_angle > 60:  # More lenient angle threshold
        finger_states['thumb'] = 'up'
    else:
        finger_states['thumb'] = 'side'
    
    # Index finger
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    
    # Calculate index finger angle with more lenient thresholds
    index_angle = calculate_angle(index_tip, index_pip, index_mcp)
    if index_angle < 30:  # More lenient angle threshold
        finger_states['index'] = 'down'
    elif index_angle > 60:  # More lenient angle threshold
        finger_states['index'] = 'up'
    else:
        finger_states['index'] = 'curved'
    
    # Middle finger
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    
    # Calculate middle finger angle with more lenient thresholds
    middle_angle = calculate_angle(middle_tip, middle_pip, middle_mcp)
    if middle_angle < 30:  # More lenient angle threshold
        finger_states['middle'] = 'down'
    elif middle_angle > 60:  # More lenient angle threshold
        finger_states['middle'] = 'up'
    else:
        finger_states['middle'] = 'curved'
    
    # Ring finger
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    
    # Calculate ring finger angle with more lenient thresholds
    ring_angle = calculate_angle(ring_tip, ring_pip, ring_mcp)
    if ring_angle < 30:  # More lenient angle threshold
        finger_states['ring'] = 'down'
    elif ring_angle > 60:  # More lenient angle threshold
        finger_states['ring'] = 'up'
    else:
        finger_states['ring'] = 'curved'
    
    # Pinky
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]
    
    # Calculate pinky angle with more lenient thresholds
    pinky_angle = calculate_angle(pinky_tip, pinky_pip, pinky_mcp)
    if pinky_angle < 30:  # More lenient angle threshold
        finger_states['pinky'] = 'down'
    elif pinky_angle > 60:  # More lenient angle threshold
        finger_states['pinky'] = 'up'
    else:
        finger_states['pinky'] = 'curved'
    
    return finger_states

def calculate_finger_spreads(landmarks):
    """Calculate finger spreads with more lenient thresholds"""
    spreads = {}
    
    # Define more lenient thresholds
    CLOSE_THRESHOLD = 0.15  # Increased from 0.1
    SPREAD_THRESHOLD = 0.3  # Increased from 0.2
    WIDE_SPREAD_THRESHOLD = 0.5  # Increased from 0.4
    
    # Calculate spreads between fingers with more lenient thresholds
    # Index-Middle spread
    index_middle_spread = calculate_finger_spread(landmarks[8], landmarks[12])
    if index_middle_spread < CLOSE_THRESHOLD:
        spreads['index_middle'] = 'close'
    elif index_middle_spread < SPREAD_THRESHOLD:
        spreads['index_middle'] = 'slight_spread'
    elif index_middle_spread < WIDE_SPREAD_THRESHOLD:
        spreads['index_middle'] = 'spread'
    else:
        spreads['index_middle'] = 'wide_spread'
    
    # Middle-Ring spread
    middle_ring_spread = calculate_finger_spread(landmarks[12], landmarks[16])
    if middle_ring_spread < CLOSE_THRESHOLD:
        spreads['middle_ring'] = 'close'
    elif middle_ring_spread < SPREAD_THRESHOLD:
        spreads['middle_ring'] = 'slight_spread'
    elif middle_ring_spread < WIDE_SPREAD_THRESHOLD:
        spreads['middle_ring'] = 'spread'
    else:
        spreads['middle_ring'] = 'wide_spread'
    
    # Ring-Pinky spread
    ring_pinky_spread = calculate_finger_spread(landmarks[16], landmarks[20])
    if ring_pinky_spread < CLOSE_THRESHOLD:
        spreads['ring_pinky'] = 'close'
    elif ring_pinky_spread < SPREAD_THRESHOLD:
        spreads['ring_pinky'] = 'slight_spread'
    elif ring_pinky_spread < WIDE_SPREAD_THRESHOLD:
        spreads['ring_pinky'] = 'spread'
    else:
        spreads['ring_pinky'] = 'wide_spread'
    
    return spreads

if __name__ == '__main__':
    app.run(debug=True) 