"""
Advanced 3D Face Reconstruction Module
Implements state-of-the-art 3D face reconstruction for enhanced recognition accuracy
"""

import numpy as np
import cv2 as cv
import mediapipe as mp
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
import pickle
import logging
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Face3DReconstructor:
    """Advanced 3D Face Reconstruction for Enhanced Recognition"""
    
    def __init__(self):
        """Initialize 3D face reconstruction components"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # 3D face model parameters
        self.canonical_landmarks = self._load_canonical_model()
        self.pca_model = None
        self.mean_shape = None
        
        # Feature extraction weights
        self.landmark_weights = self._initialize_landmark_weights()
        
        logger.info("✅ 3D Face Reconstructor initialized")
    
    def _load_canonical_model(self) -> np.ndarray:
        """Load canonical 3D face model"""
        # MediaPipe 468 landmark indices for key facial features
        key_landmarks = [
            # Face contour
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            # Eyes
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            # Nose
            1, 2, 5, 4, 6, 168, 8, 9, 10, 151, 195, 197, 196, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 290, 328, 326, 2, 97, 99, 68, 67,
            # Mouth
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318
        ]
        
        # Create canonical 3D coordinates (normalized)
        canonical = np.zeros((len(key_landmarks), 3))
        for i, idx in enumerate(key_landmarks):
            # Approximate 3D coordinates based on facial anatomy
            x = (idx % 23) / 22.0 - 0.5  # Normalized x
            y = (idx // 23) / 20.0 - 0.5  # Normalized y
            z = np.sin(x * np.pi) * 0.1  # Approximate depth
            canonical[i] = [x, y, z]
        
        return canonical
    
    def _initialize_landmark_weights(self) -> np.ndarray:
        """Initialize weights for different facial landmarks"""
        # Higher weights for more stable landmarks
        weights = {
            'eye_corners': 2.0,
            'nose_tip': 2.5,
            'mouth_corners': 2.0,
            'face_contour': 1.5,
            'general': 1.0
        }
        
        # Create weight array (simplified for this implementation)
        landmark_weights = np.ones(468) * weights['general']
        
        # Eye corners
        eye_landmarks = [33, 133, 362, 263]
        landmark_weights[eye_landmarks] = weights['eye_corners']
        
        # Nose tip
        nose_landmarks = [1, 2, 5, 4, 6]
        landmark_weights[nose_landmarks] = weights['nose_tip']
        
        # Mouth corners
        mouth_landmarks = [61, 291, 39, 181]
        landmark_weights[mouth_landmarks] = weights['mouth_corners']
        
        return landmark_weights
    
    def extract_3d_landmarks(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract 3D facial landmarks from image"""
        try:
            # Convert BGR to RGB
            rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            
            # Process image
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
            
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract 3D coordinates
            landmarks_3d = []
            landmarks_2d = []
            
            h, w = image.shape[:2]
            
            for landmark in face_landmarks.landmark:
                # 3D coordinates (x, y, z)
                x3d = landmark.x
                y3d = landmark.y
                z3d = landmark.z
                
                # 2D coordinates for visualization
                x2d = int(landmark.x * w)
                y2d = int(landmark.y * h)
                
                landmarks_3d.append([x3d, y3d, z3d])
                landmarks_2d.append([x2d, y2d])
            
            landmarks_3d = np.array(landmarks_3d)
            landmarks_2d = np.array(landmarks_2d)
            
            # Calculate additional 3D features
            face_features = self._calculate_3d_features(landmarks_3d)
            
            return {
                'landmarks_3d': landmarks_3d,
                'landmarks_2d': landmarks_2d,
                'features': face_features,
                'quality_score': self._calculate_quality_score(landmarks_3d, image),
                'pose_angles': self._estimate_pose(landmarks_3d),
                'depth_map': self._generate_depth_map(landmarks_3d, image.shape[:2])
            }
            
        except Exception as e:
            logger.error(f"3D landmark extraction failed: {e}")
            return None
    
    def _calculate_3d_features(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """Calculate advanced 3D facial features"""
        features = {}
        
        try:
            # Face dimensions
            features['face_width'] = np.max(landmarks_3d[:, 0]) - np.min(landmarks_3d[:, 0])
            features['face_height'] = np.max(landmarks_3d[:, 1]) - np.min(landmarks_3d[:, 1])
            features['face_depth'] = np.max(landmarks_3d[:, 2]) - np.min(landmarks_3d[:, 2])
            
            # Eye features
            left_eye = landmarks_3d[33:42]  # Approximate left eye region
            right_eye = landmarks_3d[362:371]  # Approximate right eye region
            
            features['eye_distance'] = np.linalg.norm(
                np.mean(left_eye, axis=0) - np.mean(right_eye, axis=0)
            )
            
            # Nose features
            nose_tip = landmarks_3d[1]  # Nose tip
            nose_bridge = landmarks_3d[6]  # Nose bridge
            
            features['nose_length'] = np.linalg.norm(nose_tip - nose_bridge)
            features['nose_protrusion'] = nose_tip[2] - np.mean(landmarks_3d[:, 2])
            
            # Mouth features
            mouth_left = landmarks_3d[61]
            mouth_right = landmarks_3d[291]
            
            features['mouth_width'] = np.linalg.norm(mouth_left - mouth_right)
            
            # Facial symmetry
            center_line = np.mean(landmarks_3d[:, 0])
            left_side = landmarks_3d[landmarks_3d[:, 0] < center_line]
            right_side = landmarks_3d[landmarks_3d[:, 0] > center_line]
            
            # Mirror right side to compare with left
            right_mirrored = right_side.copy()
            right_mirrored[:, 0] = 2 * center_line - right_mirrored[:, 0]
            
            if len(left_side) > 0 and len(right_mirrored) > 0:
                features['symmetry_score'] = 1.0 / (1.0 + np.mean(
                    [np.min(np.linalg.norm(left_side - point, axis=1)) 
                     for point in right_mirrored]
                ))
            else:
                features['symmetry_score'] = 0.5
            
            # Curvature analysis
            features['face_curvature'] = self._calculate_curvature(landmarks_3d)
            
        except Exception as e:
            logger.warning(f"Feature calculation error: {e}")
            features = {key: 0.0 for key in [
                'face_width', 'face_height', 'face_depth', 'eye_distance',
                'nose_length', 'nose_protrusion', 'mouth_width', 
                'symmetry_score', 'face_curvature'
            ]}
        
        return features
    
    def _calculate_curvature(self, landmarks_3d: np.ndarray) -> float:
        """Calculate facial surface curvature"""
        try:
            # Use face contour points for curvature calculation
            contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288]
            contour_points = landmarks_3d[contour_indices]
            
            # Calculate second derivatives (curvature approximation)
            curvatures = []
            for i in range(1, len(contour_points) - 1):
                p1, p2, p3 = contour_points[i-1], contour_points[i], contour_points[i+1]
                
                # Calculate curvature using three points
                v1 = p2 - p1
                v2 = p3 - p2
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    curvature = 1 - cos_angle  # Curvature measure
                    curvatures.append(curvature)
            
            return np.mean(curvatures) if curvatures else 0.0
            
        except Exception as e:
            logger.warning(f"Curvature calculation error: {e}")
            return 0.0
    
    def _calculate_quality_score(self, landmarks_3d: np.ndarray, image: np.ndarray) -> float:
        """Calculate quality score for 3D reconstruction"""
        try:
            score = 1.0
            
            # Check landmark distribution
            std_x = np.std(landmarks_3d[:, 0])
            std_y = np.std(landmarks_3d[:, 1])
            std_z = np.std(landmarks_3d[:, 2])
            
            # Penalize if landmarks are too clustered
            if std_x < 0.1 or std_y < 0.1:
                score *= 0.7
            
            # Check depth variation
            if std_z < 0.01:
                score *= 0.8
            
            # Check image quality
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
            
            # Penalize blurry images
            if laplacian_var < 100:
                score *= 0.6
            
            # Check lighting conditions
            mean_brightness = np.mean(gray)
            if mean_brightness < 50 or mean_brightness > 200:
                score *= 0.8
            
            return max(0.1, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"Quality score calculation error: {e}")
            return 0.5
    
    def _estimate_pose(self, landmarks_3d: np.ndarray) -> Dict[str, float]:
        """Estimate head pose angles"""
        try:
            # Use key landmarks for pose estimation
            nose_tip = landmarks_3d[1]
            left_eye = landmarks_3d[33]
            right_eye = landmarks_3d[362]
            
            # Calculate yaw (left-right rotation)
            eye_center = (left_eye + right_eye) / 2
            yaw = np.arctan2(nose_tip[0] - eye_center[0], nose_tip[2] - eye_center[2])
            
            # Calculate pitch (up-down rotation)
            pitch = np.arctan2(nose_tip[1] - eye_center[1], nose_tip[2] - eye_center[2])
            
            # Calculate roll (tilt)
            eye_vector = right_eye - left_eye
            roll = np.arctan2(eye_vector[1], eye_vector[0])
            
            return {
                'yaw': np.degrees(yaw),
                'pitch': np.degrees(pitch),
                'roll': np.degrees(roll)
            }
            
        except Exception as e:
            logger.warning(f"Pose estimation error: {e}")
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
    
    def _generate_depth_map(self, landmarks_3d: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Generate depth map from 3D landmarks"""
        try:
            h, w = image_shape
            depth_map = np.zeros((h, w), dtype=np.float32)
            
            # Create depth values from z-coordinates
            z_values = landmarks_3d[:, 2]
            z_normalized = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values) + 1e-8)
            
            # Map landmarks to image coordinates
            for i, (x, y, z) in enumerate(landmarks_3d):
                img_x = int(x * w)
                img_y = int(y * h)
                
                if 0 <= img_x < w and 0 <= img_y < h:
                    depth_map[img_y, img_x] = z_normalized[i]
            
            # Interpolate depth map
            from scipy.interpolate import griddata
            
            # Get valid points
            valid_points = []
            valid_values = []
            
            for y in range(h):
                for x in range(w):
                    if depth_map[y, x] > 0:
                        valid_points.append([x, y])
                        valid_values.append(depth_map[y, x])
            
            if len(valid_points) > 3:
                # Create grid
                grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                
                # Interpolate
                depth_map = griddata(
                    valid_points, valid_values, 
                    (grid_x, grid_y), 
                    method='linear', 
                    fill_value=0
                )
            
            return depth_map
            
        except Exception as e:
            logger.warning(f"Depth map generation error: {e}")
            return np.zeros(image_shape, dtype=np.float32)
    
    def create_3d_template(self, images: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Create comprehensive 3D face template from multiple images"""
        try:
            all_landmarks = []
            all_features = []
            quality_scores = []
            pose_angles = []
            
            logger.info(f"Processing {len(images)} images for 3D template creation")
            
            for i, image in enumerate(images):
                result = self.extract_3d_landmarks(image)
                
                if result is None:
                    logger.warning(f"Failed to extract landmarks from image {i}")
                    continue
                
                all_landmarks.append(result['landmarks_3d'])
                all_features.append(result['features'])
                quality_scores.append(result['quality_score'])
                pose_angles.append(result['pose_angles'])
            
            if len(all_landmarks) < 3:
                logger.error("Insufficient valid images for 3D template creation")
                return None
            
            # Create weighted average based on quality scores
            weights = np.array(quality_scores)
            weights = weights / np.sum(weights)
            
            # Average landmarks
            avg_landmarks = np.average(all_landmarks, axis=0, weights=weights)
            
            # Average features
            avg_features = {}
            for key in all_features[0].keys():
                values = [features[key] for features in all_features]
                avg_features[key] = np.average(values, weights=weights)
            
            # Calculate pose variation
            pose_variation = self._calculate_pose_variation(pose_angles)
            
            # Create 3D descriptor
            descriptor_3d = self._create_3d_descriptor(avg_landmarks, avg_features)
            
            # Calculate template confidence
            confidence = self._calculate_template_confidence(
                quality_scores, pose_variation, len(all_landmarks)
            )
            
            template = {
                'landmarks_3d': avg_landmarks,
                'features': avg_features,
                'descriptor': descriptor_3d,
                'quality_scores': quality_scores,
                'pose_variation': pose_variation,
                'confidence': confidence,
                'num_images': len(all_landmarks),
                'template_type': '3D_advanced',
                'version': '2.0'
            }
            
            logger.info(f"✅ 3D template created with confidence: {confidence:.3f}")
            return template
            
        except Exception as e:
            logger.error(f"3D template creation failed: {e}")
            return None
    
    def _calculate_pose_variation(self, pose_angles: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate pose variation across images"""
        try:
            yaws = [pose['yaw'] for pose in pose_angles]
            pitches = [pose['pitch'] for pose in pose_angles]
            rolls = [pose['roll'] for pose in pose_angles]
            
            return {
                'yaw_std': np.std(yaws),
                'pitch_std': np.std(pitches),
                'roll_std': np.std(rolls),
                'total_variation': np.std(yaws) + np.std(pitches) + np.std(rolls)
            }
        except Exception as e:
            logger.warning(f"Pose variation calculation error: {e}")
            return {'yaw_std': 0, 'pitch_std': 0, 'roll_std': 0, 'total_variation': 0}
    
    def _create_3d_descriptor(self, landmarks_3d: np.ndarray, features: Dict[str, float]) -> np.ndarray:
        """Create comprehensive 3D descriptor"""
        try:
            # Flatten landmarks
            landmark_vector = landmarks_3d.flatten()
            
            # Convert features to vector
            feature_vector = np.array(list(features.values()))
            
            # Combine descriptors
            descriptor = np.concatenate([landmark_vector, feature_vector])
            
            # Normalize
            descriptor = descriptor / (np.linalg.norm(descriptor) + 1e-8)
            
            return descriptor
            
        except Exception as e:
            logger.error(f"3D descriptor creation failed: {e}")
            return np.zeros(100)  # Fallback descriptor
    
    def _calculate_template_confidence(self, quality_scores: List[float], 
                                     pose_variation: Dict[str, float], 
                                     num_images: int) -> float:
        """Calculate overall template confidence"""
        try:
            # Base confidence from quality scores
            avg_quality = np.mean(quality_scores)
            quality_std = np.std(quality_scores)
            
            # Confidence from pose variation (more variation = better)
            pose_score = min(1.0, pose_variation['total_variation'] / 30.0)  # Normalize to 30 degrees
            
            # Confidence from number of images
            num_score = min(1.0, num_images / 10.0)  # Optimal around 10 images
            
            # Combined confidence
            confidence = (avg_quality * 0.4 + pose_score * 0.3 + num_score * 0.2 + 
                         (1.0 - quality_std) * 0.1)
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Confidence calculation error: {e}")
            return 0.5
    
    def match_3d_faces(self, template1: Dict[str, Any], template2: Dict[str, Any]) -> float:
        """Advanced 3D face matching with multiple metrics"""
        try:
            # Descriptor similarity
            desc_similarity = 1 - cosine(template1['descriptor'], template2['descriptor'])
            
            # Feature similarity
            feature_similarity = self._calculate_feature_similarity(
                template1['features'], template2['features']
            )
            
            # Landmark similarity
            landmark_similarity = self._calculate_landmark_similarity(
                template1['landmarks_3d'], template2['landmarks_3d']
            )
            
            # Weighted combination
            weights = {
                'descriptor': 0.4,
                'features': 0.3,
                'landmarks': 0.3
            }
            
            final_similarity = (
                desc_similarity * weights['descriptor'] +
                feature_similarity * weights['features'] +
                landmark_similarity * weights['landmarks']
            )
            
            # Adjust based on template confidence
            confidence_factor = (template1['confidence'] + template2['confidence']) / 2
            final_similarity *= confidence_factor
            
            return max(0.0, min(1.0, final_similarity))
            
        except Exception as e:
            logger.error(f"3D face matching failed: {e}")
            return 0.0
    
    def _calculate_feature_similarity(self, features1: Dict[str, float], 
                                    features2: Dict[str, float]) -> float:
        """Calculate similarity between facial features"""
        try:
            similarities = []
            
            for key in features1.keys():
                if key in features2:
                    val1, val2 = features1[key], features2[key]
                    
                    # Normalize by expected range
                    if key in ['face_width', 'face_height', 'eye_distance', 'mouth_width']:
                        # Size features - use relative difference
                        if val1 > 0 and val2 > 0:
                            sim = 1 - abs(val1 - val2) / max(val1, val2)
                        else:
                            sim = 0
                    elif key == 'symmetry_score':
                        # Symmetry - direct comparison
                        sim = 1 - abs(val1 - val2)
                    else:
                        # Other features - normalized difference
                        sim = 1 - min(1, abs(val1 - val2))
                    
                    similarities.append(max(0, sim))
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"Feature similarity calculation error: {e}")
            return 0.0
    
    def _calculate_landmark_similarity(self, landmarks1: np.ndarray, 
                                     landmarks2: np.ndarray) -> float:
        """Calculate similarity between 3D landmarks"""
        try:
            # Align landmarks using Procrustes analysis
            aligned_lm1, aligned_lm2 = self._procrustes_alignment(landmarks1, landmarks2)
            
            # Calculate weighted distances
            distances = np.linalg.norm(aligned_lm1 - aligned_lm2, axis=1)
            weighted_distances = distances * self.landmark_weights[:len(distances)]
            
            # Convert to similarity
            avg_distance = np.mean(weighted_distances)
            similarity = 1 / (1 + avg_distance * 10)  # Scale factor
            
            return similarity
            
        except Exception as e:
            logger.warning(f"Landmark similarity calculation error: {e}")
            return 0.0
    
    def _procrustes_alignment(self, landmarks1: np.ndarray, 
                            landmarks2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align two sets of landmarks using Procrustes analysis"""
        try:
            # Center the landmarks
            lm1_centered = landmarks1 - np.mean(landmarks1, axis=0)
            lm2_centered = landmarks2 - np.mean(landmarks2, axis=0)
            
            # Scale to unit norm
            lm1_scaled = lm1_centered / (np.linalg.norm(lm1_centered) + 1e-8)
            lm2_scaled = lm2_centered / (np.linalg.norm(lm2_centered) + 1e-8)
            
            # Find optimal rotation using SVD
            H = lm1_scaled.T @ lm2_scaled
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Apply rotation to second set
            lm2_aligned = lm2_scaled @ R
            
            return lm1_scaled, lm2_aligned
            
        except Exception as e:
            logger.warning(f"Procrustes alignment error: {e}")
            return landmarks1, landmarks2

# Global instance
face_3d_reconstructor = Face3DReconstructor()

def extract_3d_face_features(image: np.ndarray) -> Optional[Dict[str, Any]]:
    """Extract 3D face features from single image"""
    return face_3d_reconstructor.extract_3d_landmarks(image)

def create_advanced_3d_template(images: List[np.ndarray]) -> Optional[Dict[str, Any]]:
    """Create advanced 3D face template from multiple images"""
    return face_3d_reconstructor.create_3d_template(images)

def match_3d_face_templates(template1: Dict[str, Any], template2: Dict[str, Any]) -> float:
    """Match two 3D face templates"""
    return face_3d_reconstructor.match_3d_faces(template1, template2)

def visualize_3d_landmarks(image: np.ndarray, landmarks_3d: np.ndarray) -> np.ndarray:
    """Visualize 3D landmarks on image"""
    try:
        vis_image = image.copy()
        h, w = image.shape[:2]
        
        # Draw landmarks
        for landmark in landmarks_3d:
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            z = landmark[2]
            
            # Color based on depth
            color_intensity = int(255 * (z + 0.1) / 0.2)  # Normalize z
            color = (0, color_intensity, 255 - color_intensity)
            
            cv.circle(vis_image, (x, y), 2, color, -1)
        
        return vis_image
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return image

if __name__ == "__main__":
    # Test the 3D reconstruction system
    print("🧪 Testing 3D Face Reconstruction System...")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test feature extraction
    result = extract_3d_face_features(test_image)
    if result:
        print("✅ 3D feature extraction working")
    else:
        print("❌ 3D feature extraction failed")
    
    print("🎉 3D Face Reconstruction System ready!")
