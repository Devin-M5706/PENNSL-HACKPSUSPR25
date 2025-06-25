"""
ASL Letter Dataset
This file contains predefined hand landmark positions for ASL letters.
Each letter is defined by its finger positions, hand orientation, and allowed variations.
"""

letter_patterns = {
    'A': {
        'description': 'Make a fist with thumb resting on the side',
        'fingers': {
            'thumb': ['side', 'up', 'curved'],  # Allow more thumb positions
            'index': ['down', 'curved'],
            'middle': ['down', 'curved'],
            'ring': ['down', 'curved'],
            'pinky': ['down', 'curved']
        },
        'spreads': {
            'index_middle': ['close', 'touch'],  # Allow touching fingers
            'middle_ring': ['close', 'touch'],
            'ring_pinky': ['close', 'touch']
        },
        'angles': {
            'thumb_palm': [30, 90],  # Angle range between thumb and palm
            'index_middle': [0, 20],  # Small angle between fingers
            'palm_vertical': [60, 120]  # Palm can be tilted
        },
        'threshold': 0.3  # Even lower threshold for A
    },
    'B': {
        'description': 'All fingers extended and together',
        'fingers': {
            'thumb': ['up', 'side'],  # Allow thumb to be up or to the side
            'index': ['up', 'slightly_curved'],
            'middle': ['up', 'slightly_curved'],
            'ring': ['up', 'slightly_curved'],
            'pinky': ['up', 'slightly_curved']
        },
        'spreads': {
            'index_middle': ['close', 'touch'],
            'middle_ring': ['close', 'touch'],
            'ring_pinky': ['close', 'touch']
        },
        'angles': {
            'thumb_palm': [30, 90],
            'fingers_vertical': [70, 110],  # Fingers should be mostly vertical
            'palm_vertical': [70, 110]
        },
        'threshold': 0.4
    },
    'C': {
        'description': 'Curved hand forming a C shape',
        'fingers': {
            'thumb': ['curved', 'side'],
            'index': ['curved'],
            'middle': ['curved'],
            'ring': ['curved'],
            'pinky': ['curved']
        },
        'spreads': {
            'index_middle': ['close', 'slight_spread'],
            'middle_ring': ['close', 'slight_spread'],
            'ring_pinky': ['close', 'slight_spread']
        },
        'angles': {
            'curve_angle': [30, 60],  # Angle of finger curvature
            'palm_vertical': [60, 120]
        },
        'threshold': 0.4
    },
    'D': {
        'description': 'Index finger pointing up',
        'fingers': {
            'thumb': 'up',
            'index': 'up',
            'middle': 'down',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'spread',
            'middle_ring': 'close',
            'ring_pinky': 'close'
        }
    },
    'E': {
        'description': 'All fingers slightly curved',
        'fingers': {
            'thumb': 'up',
            'index': 'curved',
            'middle': 'curved',
            'ring': 'curved',
            'pinky': 'curved'
        },
        'spreads': {
            'index_middle': 'close',
            'middle_ring': 'close',
            'ring_pinky': 'close'
        }
    },
    'F': {
        'description': 'Index and middle fingers crossed',
        'fingers': {
            'thumb': 'up',
            'index': 'up',
            'middle': 'up',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'crossed',
            'middle_ring': 'spread',
            'ring_pinky': 'close'
        }
    },
    'G': {
        'description': 'Index finger pointing to the side',
        'fingers': {
            'thumb': 'up',
            'index': 'side',
            'middle': 'down',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'spread',
            'middle_ring': 'close',
            'ring_pinky': 'close'
        }
    },
    'H': {
        'description': 'Index and middle fingers extended',
        'fingers': {
            'thumb': 'up',
            'index': 'up',
            'middle': 'up',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'spread',
            'middle_ring': 'spread',
            'ring_pinky': 'close'
        }
    },
    'I': {
        'description': 'Pinky finger extended',
        'fingers': {
            'thumb': 'up',
            'index': 'down',
            'middle': 'down',
            'ring': 'down',
            'pinky': 'up'
        },
        'spreads': {
            'index_middle': 'close',
            'middle_ring': 'close',
            'ring_pinky': 'spread'
        }
    },
    'J': {
        'description': 'Index finger making a J shape',
        'fingers': {
            'thumb': 'up',
            'index': 'curved',
            'middle': 'down',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'spread',
            'middle_ring': 'close',
            'ring_pinky': 'close'
        }
    },
    'K': {
        'description': 'Index and middle fingers spread like a K',
        'fingers': {
            'thumb': 'up',
            'index': 'up',
            'middle': 'up',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'wide_spread',
            'middle_ring': 'spread',
            'ring_pinky': 'close'
        }
    },
    'L': {
        'description': 'Index finger and thumb forming an L',
        'fingers': {
            'thumb': 'up',
            'index': 'up',
            'middle': 'down',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'spread',
            'middle_ring': 'close',
            'ring_pinky': 'close'
        }
    },
    'M': {
        'description': 'Three fingers down (thumb, index, middle)',
        'fingers': {
            'thumb': 'down',
            'index': 'down',
            'middle': 'down',
            'ring': 'up',
            'pinky': 'up'
        },
        'spreads': {
            'index_middle': 'close',
            'middle_ring': 'spread',
            'ring_pinky': 'spread'
        }
    },
    'N': {
        'description': 'Two fingers down (thumb, index)',
        'fingers': {
            'thumb': 'down',
            'index': 'down',
            'middle': 'up',
            'ring': 'up',
            'pinky': 'up'
        },
        'spreads': {
            'index_middle': 'spread',
            'middle_ring': 'close',
            'ring_pinky': 'close'
        }
    },
    'O': {
        'description': 'All fingers curved to form an O',
        'fingers': {
            'thumb': 'curved',
            'index': 'curved',
            'middle': 'curved',
            'ring': 'curved',
            'pinky': 'curved'
        },
        'spreads': {
            'index_middle': 'close',
            'middle_ring': 'close',
            'ring_pinky': 'close'
        }
    },
    'P': {
        'description': 'Index finger pointing down',
        'fingers': {
            'thumb': 'up',
            'index': 'down',
            'middle': 'down',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'spread',
            'middle_ring': 'close',
            'ring_pinky': 'close'
        }
    },
    'Q': {
        'description': 'All fingers down',
        'fingers': {
            'thumb': 'down',
            'index': 'down',
            'middle': 'down',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'close',
            'middle_ring': 'close',
            'ring_pinky': 'close'
        }
    },
    'R': {
        'description': 'Index and middle fingers crossed',
        'fingers': {
            'thumb': 'up',
            'index': 'up',
            'middle': 'up',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'crossed',
            'middle_ring': 'spread',
            'ring_pinky': 'close'
        }
    },
    'S': {
        'description': 'Fist with thumb over fingers',
        'fingers': {
            'thumb': 'over',
            'index': 'down',
            'middle': 'down',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'close',
            'middle_ring': 'close',
            'ring_pinky': 'close'
        }
    },
    'T': {
        'description': 'Thumb between index and middle fingers',
        'fingers': {
            'thumb': 'between',
            'index': 'up',
            'middle': 'up',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'close',
            'middle_ring': 'spread',
            'ring_pinky': 'close'
        }
    },
    'U': {
        'description': 'Index and middle fingers together',
        'fingers': {
            'thumb': 'up',
            'index': 'up',
            'middle': 'up',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'close',
            'middle_ring': 'spread',
            'ring_pinky': 'close'
        }
    },
    'V': {
        'description': 'Index and middle fingers spread in a V',
        'fingers': {
            'thumb': 'up',
            'index': 'up',
            'middle': 'up',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'spread',
            'middle_ring': 'spread',
            'ring_pinky': 'close'
        }
    },
    'W': {
        'description': 'Three fingers up (index, middle, ring)',
        'fingers': {
            'thumb': 'up',
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
        'description': 'Index finger bent',
        'fingers': {
            'thumb': 'up',
            'index': 'bent',
            'middle': 'down',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'spread',
            'middle_ring': 'close',
            'ring_pinky': 'close'
        }
    },
    'Y': {
        'description': 'Thumb and pinky extended',
        'fingers': {
            'thumb': 'up',
            'index': 'down',
            'middle': 'down',
            'ring': 'down',
            'pinky': 'up'
        },
        'spreads': {
            'index_middle': 'close',
            'middle_ring': 'close',
            'ring_pinky': 'spread'
        }
    },
    'Z': {
        'description': 'Index finger making a Z shape',
        'fingers': {
            'thumb': 'up',
            'index': 'curved',
            'middle': 'down',
            'ring': 'down',
            'pinky': 'down'
        },
        'spreads': {
            'index_middle': 'spread',
            'middle_ring': 'close',
            'ring_pinky': 'close'
        }
    }
}

def calculate_match_score(pattern, positions, spreads):
    """Calculate how well the current hand position matches a letter pattern with enhanced scoring."""
    # Initialize scores
    finger_position_score = 0
    finger_spread_score = 0
    angle_score = 0
    
    # Check finger positions with multiple allowed positions
    finger_count = 0
    for finger, expected_positions in pattern['fingers'].items():
        if not isinstance(expected_positions, list):
            expected_positions = [expected_positions]
        
        actual = positions[finger]
        max_score = 0
        
        for expected in expected_positions:
            score = 0
            if actual == expected:
                score = 1.0
            elif actual in ['down', 'curved'] and expected in ['down', 'curved']:
                score = 0.9
            elif actual in ['side', 'up'] and expected in ['side', 'up']:
                score = 0.9
            elif actual == 'slightly_curved' and expected in ['up', 'curved']:
                score = 0.8
            max_score = max(max_score, score)
        
        finger_position_score += max_score
        finger_count += 1
    
    finger_position_score /= finger_count if finger_count > 0 else 1
    
    # Check spreads with multiple allowed positions
    spread_count = 0
    for spread_type, expected_spreads in pattern['spreads'].items():
        if not isinstance(expected_spreads, list):
            expected_spreads = [expected_spreads]
        
        actual = spreads[spread_type]
        max_score = 0
        
        for expected in expected_spreads:
            score = 0
            if actual == expected:
                score = 1.0
            elif (actual == 'close' and expected in ['touch', 'slight_spread']) or \
                 (actual == 'touch' and expected in ['close', 'slight_spread']):
                score = 0.9
            elif actual == 'slight_spread' and expected in ['close', 'spread']:
                score = 0.8
            max_score = max(max_score, score)
        
        finger_spread_score += max_score
        spread_count += 1
    
    finger_spread_score /= spread_count if spread_count > 0 else 1
    
    # Calculate final score with weighted components
    total_score = (finger_position_score * 0.6) + (finger_spread_score * 0.4)
    
    # Apply letter-specific boosts
    if pattern.get('description', '').startswith('Make a fist'):
        total_score *= 1.4  # 40% boost for letter A
    
    return total_score

def match_hand_to_letter(finger_positions, finger_spreads, history=None):
    """Match hand position to ASL letter with improved accuracy."""
    best_match = None
    best_score = 0
    
    for letter, pattern in letter_patterns.items():
        score = calculate_match_score(pattern, finger_positions, finger_spreads)
        threshold = pattern.get('threshold', 0.5)  # Default threshold of 0.5
        
        if score >= threshold:
            if score > best_score:
                best_score = score
                best_match = letter
    
    # Apply temporal smoothing if history is provided
    if history and len(history) > 0 and best_match:
        recent_occurrences = sum(1 for h in history[-3:] if h == best_match)
        if recent_occurrences > 0:
            best_score *= (1 + 0.1 * recent_occurrences)  # Up to 30% boost
    
    return best_match, best_score

def get_letter_description(letter):
    """Get the description for a specific ASL letter."""
    if letter in letter_patterns:
        return letter_patterns[letter]['description']
    return None

def get_letter_finger_positions(letter):
    """Get the expected finger positions for a specific ASL letter."""
    if letter in letter_patterns:
        return letter_patterns[letter]['fingers']
    return None

def get_letter_finger_spreads(letter):
    """Get the expected finger spreads for a specific ASL letter."""
    if letter in letter_patterns:
        return letter_patterns[letter]['spreads']
    return None 