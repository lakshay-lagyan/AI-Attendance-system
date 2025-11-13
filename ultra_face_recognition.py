"""
Ultra-Advanced Face Recognition System
State-of-the-art face detection and recognition with robust tracking
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import dlib
from scipy.spatial.distance import cosine
from collections import defaultdict, deque
import time
import math
from typing import List, Dict, Tuple, Optional, Any

class UltraFaceRecognizer:
    def __init__(self):
        """Initialize ultra-advanced face recognition system"""
        # MediaPipe components
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Multiple detection models for robustness
        self.face_detector_close = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.3  # Short range, lower threshold
        )
        self.face_detector_far = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.4  # Full range
        )
        
        # Face mesh for detailed landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=15,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.3
        )
        
        # Enhanced preprocessing
        self.clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        
        # Face tracking
        self.face_trackers = {}
        self.next_face_id = 0
        self.face_history = defaultdict(lambda: deque(maxlen=50))
        
    def enhance_frame(self, image: np.ndarray) -> np.ndarray:
        """Advanced frame enhancement for challenging conditions"""
        # Multi-step enhancement pipeline
        enhanced = image.copy().astype(np.float32)
        
        # 1. Gamma correction based on brightness
        gray_temp = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray_temp)
        
        if mean_brightness < 80:  # Dark image
            gamma = 0.6
        elif mean_brightness > 180:  # Bright image  
            gamma = 1.4
        else:
            gamma = 1.0
            
        if gamma != 1.0:
            enhanced = 255 * np.power(enhanced / 255, gamma)
        
        enhanced = enhanced.astype(np.uint8)
        
        # 2. CLAHE for local contrast
        lab = cv.cvtColor(enhanced, cv.COLOR_BGR2LAB)
        lab[:,:,0] = self.clahe.apply(lab[:,:,0])
        enhanced = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
        
        # 3. Bilateral filter for noise reduction
        enhanced = cv.bilateralFilter(enhanced, 9, 75, 75)
        
        # 4. Sharpening for blurred images
        if self._detect_blur(enhanced) > 0.5:  # If significantly blurred
            kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
            enhanced = cv.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def _detect_blur(self, image: np.ndarray) -> float:
        """Detect blur level (0=sharp, 1=very blurred)"""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
        
        # Normalize blur score (lower variance = more blur)
        max_variance = 2000
        blur_level = max(0, 1 - (laplacian_var / max_variance))
        return min(1, blur_level)
    
    def detect_faces_ultra(self, image: np.ndarray) -> List[Dict]:
        """Ultra-robust face detection with multiple models"""
        h, w = image.shape[:2]
        
        # Enhance image
        enhanced = self.enhance_frame(image)
        rgb_enhanced = cv.cvtColor(enhanced, cv.COLOR_BGR2RGB)
        
        all_detections = []
        
        # Detection 1: Close-range model
        results_close = self.face_detector_close.process(rgb_enhanced)
        if results_close.detections:
            for det in results_close.detections:
                bbox = det.location_data.relative_bounding_box
                x = max(0, int((bbox.xmin - 0.1) * w))
                y = max(0, int((bbox.ymin - 0.1) * h))  
                width = min(w - x, int((bbox.width + 0.2) * w))
                height = min(h - y, int((bbox.height + 0.2) * h))
                
                all_detections.append({
                    'bbox': (x, y, width, height),
                    'confidence': float(det.score[0]),
                    'source': 'mp_close'
                })
        
        # Detection 2: Far-range model
        results_far = self.face_detector_far.process(rgb_enhanced)
        if results_far.detections:
            for det in results_far.detections:
                bbox = det.location_data.relative_bounding_box
                x = max(0, int((bbox.xmin - 0.1) * w))
                y = max(0, int((bbox.ymin - 0.1) * h))
                width = min(w - x, int((bbox.width + 0.2) * w))
                height = min(h - y, int((bbox.height + 0.2) * h))
                
                all_detections.append({
                    'bbox': (x, y, width, height),
                    'confidence': float(det.score[0]),
                    'source': 'mp_far'
                })
        
        # Detection 3: Multi-scale detection for small/distant faces
        scales = [1.0, 0.8, 1.2]  # Different scales
        for scale in scales:
            if scale != 1.0:
                scaled_w, scaled_h = int(w * scale), int(h * scale)
                scaled_img = cv.resize(enhanced, (scaled_w, scaled_h))
                rgb_scaled = cv.cvtColor(scaled_img, cv.COLOR_BGR2RGB)
                
                scale_results = self.face_detector_far.process(rgb_scaled)
                if scale_results.detections:
                    for det in scale_results.detections:
                        bbox = det.location_data.relative_bounding_box
                        # Scale back to original coordinates
                        x = max(0, int((bbox.xmin / scale - 0.1) * w))
                        y = max(0, int((bbox.ymin / scale - 0.1) * h))
                        width = min(w - x, int((bbox.width / scale + 0.2) * w))
                        height = min(h - y, int((bbox.height / scale + 0.2) * h))
                        
                        all_detections.append({
                            'bbox': (x, y, width, height),
                            'confidence': float(det.score[0]) * 0.9,  # Slightly lower confidence for scaled
                            'source': f'mp_scale_{scale}'
                        })
        
        # Non-maximum suppression
        final_faces = self._advanced_nms(all_detections, iou_thresh=0.4)
        
        # Add landmarks and pose analysis
        for face in final_faces:
            landmarks = self._extract_landmarks_robust(enhanced, face['bbox'])
            face['landmarks'] = landmarks
            face['pose'] = self._analyze_pose(landmarks) if landmarks is not None else None
        
        return final_faces
    
    def _advanced_nms(self, detections: List[Dict], iou_thresh: float = 0.4) -> List[Dict]:
        """Advanced Non-Maximum Suppression with confidence weighting"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        kept = []
        while detections:
            current = detections.pop(0)
            kept.append(current)
            
            # Remove overlapping detections
            remaining = []
            for det in detections:
                iou = self._calculate_iou(current['bbox'], det['bbox'])
                if iou < iou_thresh:
                    remaining.append(det)
            
            detections = remaining
        
        return kept
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_landmarks_robust(self, image: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        """Robust landmark extraction"""
        try:
            x, y, w, h = bbox
            
            # Expand ROI slightly
            padding = int(min(w, h) * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_roi = image[y1:y2, x1:x2]
            if face_roi.size == 0:
                return None
            
            rgb_roi = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)
            mesh_results = self.face_mesh.process(rgb_roi)
            
            if mesh_results.multi_face_landmarks:
                # Get the first face's landmarks
                face_landmarks = mesh_results.multi_face_landmarks[0]
                landmarks = []
                
                # Convert relative coordinates to absolute
                roi_h, roi_w = face_roi.shape[:2]
                for landmark in face_landmarks.landmark:
                    abs_x = landmark.x * roi_w + x1
                    abs_y = landmark.y * roi_h + y1
                    landmarks.append([abs_x, abs_y])
                
                return np.array(landmarks)
        
        except Exception:
            pass
        
        return None
    
    def _analyze_pose(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Advanced pose analysis from landmarks"""
        if landmarks is None or len(landmarks) < 100:
            return {'pose': 'unknown', 'quality': 0.0}
        
        try:
            # Key points for pose estimation (MediaPipe face mesh indices)
            nose_tip = landmarks[1]      # Nose tip
            chin = landmarks[175]        # Chin
            left_eye = landmarks[33]     # Left eye
            right_eye = landmarks[263]   # Right eye
            left_mouth = landmarks[61]   # Left mouth corner
            right_mouth = landmarks[291] # Right mouth corner
            
            # Calculate face center and eye center
            eye_center = (left_eye + right_eye) / 2
            mouth_center = (left_mouth + right_mouth) / 2
            face_center = (nose_tip + chin) / 2
            
            # Calculate angles
            # Yaw (left-right rotation)
            eye_distance = np.linalg.norm(right_eye - left_eye)
            nose_deviation = abs(nose_tip[0] - eye_center[0])
            yaw = (nose_deviation / (eye_distance / 2)) * 45  # Normalize to degrees
            
            # Pitch (up-down rotation)  
            face_height = np.linalg.norm(nose_tip - chin)
            eye_nose_vertical = abs(nose_tip[1] - eye_center[1])
            pitch = (eye_nose_vertical / face_height) * 60
            
            # Roll (tilt)
            roll = abs(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi)
            
            # Determine pose category
            if yaw > 25:
                pose_cat = 'profile_right'
            elif yaw > 15:
                pose_cat = 'three_quarter_right'
            elif yaw > 10:
                pose_cat = 'slight_right'
            elif pitch > 20:
                pose_cat = 'looking_down' 
            elif pitch > 15:
                pose_cat = 'slight_down'
            elif roll > 25:
                pose_cat = 'tilted'
            else:
                pose_cat = 'frontal'
            
            # Quality score (0-1, higher is better for recognition)
            angle_penalty = (min(yaw, 45) + min(pitch, 45) + min(roll, 45)) / 135
            quality = max(0, 1 - angle_penalty)
            
            return {
                'pose': pose_cat,
                'angles': {'yaw': yaw, 'pitch': pitch, 'roll': roll},
                'quality': quality,
                'description': f"{pose_cat} (Y:{yaw:.1f}° P:{pitch:.1f}° R:{roll:.1f}°)"
            }
        
        except Exception:
            return {'pose': 'unknown', 'quality': 0.0}
    
    def track_faces_advanced(self, image: np.ndarray, faces: List[Dict]) -> List[Dict]:
        """Advanced face tracking with temporal smoothing"""
        current_time = time.time()
        
        # Clean old trackers (5 second timeout)
        for face_id in list(self.face_trackers.keys()):
            if current_time - self.face_trackers[face_id]['last_seen'] > 5.0:
                del self.face_trackers[face_id]
                if face_id in self.face_history:
                    del self.face_history[face_id]
        
        tracked_faces = []
        
        for face in faces:
            bbox = face['bbox']
            x, y, w, h = bbox
            face_center = (x + w//2, y + h//2)
            
            # Find best matching existing tracker
            best_match_id = None
            best_distance = float('inf')
            
            for face_id, tracker in self.face_trackers.items():
                last_center = tracker['center']
                distance = np.sqrt((face_center[0] - last_center[0])**2 + 
                                 (face_center[1] - last_center[1])**2)
                
                # Consider size similarity too
                last_w, last_h = tracker['size']
                size_diff = abs(w - last_w) + abs(h - last_h)
                combined_distance = distance + size_diff * 0.1
                
                if combined_distance < 150 and combined_distance < best_distance:
                    best_distance = combined_distance
                    best_match_id = face_id
            
            # Assign or create face ID
            if best_match_id is not None:
                face['face_id'] = best_match_id
                # Update tracker
                self.face_trackers[best_match_id].update({
                    'center': face_center,
                    'size': (w, h),
                    'last_seen': current_time,
                    'bbox': bbox
                })
            else:
                # New face
                face['face_id'] = self.next_face_id
                self.face_trackers[self.next_face_id] = {
                    'center': face_center,
                    'size': (w, h),
                    'last_seen': current_time,
                    'first_seen': current_time,
                    'bbox': bbox
                }
                self.next_face_id += 1
            
            # Add to history for temporal smoothing
            self.face_history[face['face_id']].append({
                'bbox': bbox,
                'confidence': face['confidence'],
                'pose': face.get('pose'),
                'timestamp': current_time
            })
            
            # Calculate stability score
            history = list(self.face_history[face['face_id']])
            if len(history) > 1:
                # Measure consistency over time
                recent_bboxes = [h['bbox'] for h in history[-5:]]  # Last 5 frames
                bbox_variance = self._calculate_bbox_variance(recent_bboxes)
                stability = max(0, 1 - bbox_variance / 1000)  # Normalize
                face['stability'] = stability
            else:
                face['stability'] = 0.5  # New face, medium stability
            
            tracked_faces.append(face)
        
        return tracked_faces
    
    def _calculate_bbox_variance(self, bboxes: List[Tuple]) -> float:
        """Calculate variance in bounding box positions"""
        if len(bboxes) < 2:
            return 0
        
        centers = [(x + w//2, y + h//2) for x, y, w, h in bboxes]
        mean_x = np.mean([c[0] for c in centers])
        mean_y = np.mean([c[1] for c in centers])
        
        variance = np.mean([(c[0] - mean_x)**2 + (c[1] - mean_y)**2 for c in centers])
        return variance
    
    def extract_robust_features(self, image: np.ndarray, face_bbox: Tuple, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """Extract robust face features for recognition"""
        try:
            x, y, w, h = face_bbox
            
            # Extract face with generous padding
            padding = int(min(w, h) * 0.3)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_roi = image[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
            
            # Align face if landmarks available
            if landmarks is not None and len(landmarks) > 50:
                face_roi = self._align_face_advanced(face_roi, landmarks, (x1, y1))
            
            # Multiple preprocessing approaches for robustness
            processed_faces = []
            
            # Original
            face_224 = cv.resize(face_roi, (224, 224))
            processed_faces.append(face_224)
            
            # Enhanced contrast
            enhanced = self.enhance_frame(face_roi)
            face_enhanced = cv.resize(enhanced, (224, 224))
            processed_faces.append(face_enhanced)
            
            # Slightly different crops for robustness
            h_roi, w_roi = face_roi.shape[:2]
            crop_margin = int(min(h_roi, w_roi) * 0.05)
            if crop_margin > 0:
                cropped = face_roi[crop_margin:h_roi-crop_margin, crop_margin:w_roi-crop_margin]
                face_cropped = cv.resize(cropped, (224, 224))
                processed_faces.append(face_cropped)
            
            # Extract embeddings from all variants
            embeddings = []
            for proc_face in processed_faces:
                try:
                    emb = DeepFace.represent(proc_face, model_name='Facenet512', enforce_detection=False)[0]['embedding']
                    embeddings.append(np.array(emb))
                except:
                    continue
            
            if embeddings:
                # Average and normalize
                final_embedding = np.mean(embeddings, axis=0)
                final_embedding = final_embedding / (np.linalg.norm(final_embedding) + 1e-8)
                return final_embedding
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
        
        return None
    
    def _align_face_advanced(self, face_image: np.ndarray, landmarks: np.ndarray, offset: Tuple) -> np.ndarray:
        """Advanced face alignment using landmarks"""
        try:
            # Adjust landmarks to face ROI coordinates
            adjusted_landmarks = landmarks - np.array(offset)
            
            # Use eye landmarks for alignment (MediaPipe indices)
            left_eye = adjusted_landmarks[33]   # Left eye center
            right_eye = adjusted_landmarks[263] # Right eye center
            
            # Calculate rotation to make eyes horizontal
            dy = right_eye[1] - left_eye[1] 
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Limit rotation to prevent extreme transformations
            angle = np.clip(angle, -45, 45)
            
            # Rotate image
            center = tuple(np.array(face_image.shape[1::-1]) / 2)
            rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
            aligned = cv.warpAffine(face_image, rotation_matrix, 
                                  (face_image.shape[1], face_image.shape[0]))
            
            return aligned
        
        except:
            return face_image

# Global instance
ultra_recognizer = UltraFaceRecognizer()

def process_frame_ultra(image: np.ndarray) -> List[Dict]:
    """Process frame with ultra-advanced recognition"""
    # Detect faces with ultra-robust method
    faces = ultra_recognizer.detect_faces_ultra(image)
    
    # Track faces for temporal consistency  
    tracked_faces = ultra_recognizer.track_faces_advanced(image, faces)
    
    # Enhance each face with full analysis
    enhanced_faces = []
    for face in tracked_faces:
        # Calculate comprehensive quality metrics
        quality = calculate_face_quality_ultra(image, face['bbox'], face.get('landmarks'))
        
        # Extract robust features
        embedding = ultra_recognizer.extract_robust_features(
            image, face['bbox'], face.get('landmarks')
        )
        
        # Create enhanced face info
        enhanced_face = {
            **face,
            'quality': quality,
            'embedding': embedding,
            'description': generate_face_description(face, quality)
        }
        
        enhanced_faces.append(enhanced_face)
    
    return enhanced_faces

def calculate_face_quality_ultra(image: np.ndarray, bbox: Tuple, landmarks: np.ndarray) -> Dict[str, float]:
    """Ultra-comprehensive face quality assessment"""
    x, y, w, h = bbox
    face_roi = image[y:y+h, x:x+w] if y+h <= image.shape[0] and x+w <= image.shape[1] else image
    
    if face_roi.size == 0:
        return {'overall': 0.0}
    
    # Size quality (minimum 60px for good recognition)
    min_size = 60
    size_quality = min(1.0, min(w, h) / min_size)
    
    # Sharpness quality
    gray_roi = cv.cvtColor(face_roi, cv.COLOR_BGR2GRAY)
    laplacian_var = cv.Laplacian(gray_roi, cv.CV_64F).var()
    sharpness_quality = min(1.0, laplacian_var / 300.0)
    
    # Brightness quality (optimal range 80-180)
    brightness = np.mean(gray_roi)
    if 80 <= brightness <= 180:
        brightness_quality = 1.0
    else:
        brightness_quality = max(0, 1 - abs(brightness - 130) / 130.0)
    
    # Contrast quality
    contrast = np.std(gray_roi)
    contrast_quality = min(1.0, contrast / 50.0)
    
    # Pose quality from landmarks
    pose_quality = 0.5
    if landmarks is not None:
        pose_info = ultra_recognizer._analyze_pose(landmarks)
        pose_quality = pose_info.get('quality', 0.5)
    
    # Overall quality (weighted combination)
    weights = [0.25, 0.25, 0.2, 0.15, 0.15]
    qualities = [size_quality, sharpness_quality, brightness_quality, contrast_quality, pose_quality]
    
    overall = sum(w * q for w, q in zip(weights, qualities))
    
    return {
        'overall': overall,
        'size': size_quality, 
        'sharpness': sharpness_quality,
        'brightness': brightness_quality,
        'contrast': contrast_quality,
        'pose': pose_quality
    }

def generate_face_description(face: Dict, quality: Dict) -> str:
    """Generate human-readable face description"""
    pose = face.get('pose', {})
    pose_desc = pose.get('pose', 'unknown') if pose else 'unknown'
    
    quality_level = 'excellent' if quality['overall'] > 0.8 else \
                   'good' if quality['overall'] > 0.6 else \
                   'fair' if quality['overall'] > 0.4 else 'poor'
    
    stability = face.get('stability', 0.5)
    stability_desc = 'stable' if stability > 0.7 else 'moving' if stability > 0.3 else 'unstable'
    
    return f"{pose_desc} face, {quality_level} quality, {stability_desc} tracking"
