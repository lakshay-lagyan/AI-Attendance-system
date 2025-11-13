import cv2 as cv
import numpy as np
from deepface import DeepFace
import os
import faiss
import datetime
import pickle
from pymongo import MongoClient
from collections import deque
from typing import Dict, List, Optional, Any, Tuple

# Import 3D reconstruction module
try:
    from face_3d_reconstruction import (
        extract_3d_face_features, 
        create_advanced_3d_template, 
        match_3d_face_templates,
        visualize_3d_landmarks
    )
    USE_3D_RECONSTRUCTION = True
    print("✅ 3D Face Reconstruction enabled")
except ImportError as e:
    print(f"⚠️ 3D Face Reconstruction disabled: {e}")
    USE_3D_RECONSTRUCTION = False

# MongoDB connection
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(MONGODB_URI)

# Database references
transactional_db = client["transactional_db"]
attendance_col = transactional_db["attendance"]

core = client['secure_db']
persons_col = core["persons"]

# Enhanced FAISS setup with HNSW for better performance
EMBEDDING_DIM = 512
faiss_index = None
person_id_map = []

# Performance tracking
detection_stats = {
    'total_detections': 0,
    'successful_recognitions': 0,
    'unknown_persons': 0,
    'occluded_faces': 0
}

def cosine_sim(a, b) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def detect_occlusion(face_img):
    """Detect if face is occluded (mask, hand, object)"""
    try:
        gray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Check lower face area (likely occluded by mask)
        lower_face = gray[int(h*0.6):, :]
        upper_face = gray[:int(h*0.5), :]
        
        # Calculate variance - occluded areas have less variance
        lower_var = np.var(lower_face)
        upper_var = np.var(upper_face)
        
        # If lower face has significantly less variance, likely occluded
        occlusion_score = upper_var / (lower_var + 1e-8)
        
        return occlusion_score > 2.0, occlusion_score
    except:
        return False, 0.0

def is_real_face(face_img):
    """Enhanced liveness detection with occlusion tolerance"""
    try:
        if face_img is None or face_img.size == 0:
            return False
        
        gray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
        
        # 1. Laplacian variance (blur detection) - relaxed for occluded faces
        laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
        if laplacian_var < 30:  # Lowered from 50
            return False
        
        # 2. Check brightness
        brightness = np.mean(gray)
        if brightness < 20 or brightness > 235:  # More tolerant
            return False
        
        # 3. Check contrast - relaxed for occluded faces
        contrast = np.std(gray)
        if contrast < 15:  # Lowered from 20
            return False
        
        # 4. Edge detection - tolerant to partial occlusion
        edges = cv.Canny(gray, 30, 120)  # Lowered thresholds
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density < 0.005:  # More tolerant
            return False
        
        return True
    except Exception as e:
        print(f"[Liveness Error] {e}")
        return False

def get_face_embedding(image, handle_occlusion=True):
    """Extract face embedding with occlusion handling"""
    try:
        if hasattr(image, 'read'):
            file_bytes = np.frombuffer(image.read(), np.uint8)
            image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        
        if image is None or image.size == 0:
            return None
        
        # Check for occlusion
        is_occluded, occlusion_score = detect_occlusion(image)
        
        if not is_real_face(image):
            print("[Warning] Fake face or low quality detected")
            return None
        
        # Use RetinaFace for better detection with occlusions
        detector_backend = 'retinaface' if handle_occlusion else 'opencv'
        
        embedding_obj = DeepFace.represent(
            img_path=image,
            model_name='ArcFace',
            detector_backend=detector_backend,
            enforce_detection=False
        )
        
        if not embedding_obj:
            return None
        
        embedding = np.array(embedding_obj[0]['embedding'])
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Store occlusion info
        if is_occluded:
            detection_stats['occluded_faces'] += 1
        
        return embedding
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None

def extract_multi_vector_embeddings(images):
    """Extract multiple embeddings from different angles"""
    embeddings_data = []
    
    for idx, image in enumerate(images):
        try:
            if hasattr(image, 'read'):
                image.seek(0)
                file_bytes = np.frombuffer(image.read(), np.uint8)
                img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
            else:
                img = image
            
            if img is None:
                continue
            
            if not is_real_face(img):
                print(f"[Skip] Image {idx}: Failed liveness check")
                continue
            
            embedding = get_face_embedding(img)
            if embedding is None:
                continue
            
            # Calculate quality score
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
            brightness = np.mean(gray)
            
            quality_score = min(100, (laplacian_var / 100) * 50 + (brightness / 255) * 50)
            
            embeddings_data.append({
                'embedding': embedding,
                'quality': quality_score,
                'index': idx
            })
        except Exception as e:
            print(f"[Error] Processing image {idx}: {e}")
            continue
    
    return embeddings_data

def create_3d_template(embeddings_data, images=None):
    """Create advanced 3D template from multiple embeddings and images"""
    if not embeddings_data:
        return None
    
    # Sort by quality and select top embeddings
    embeddings_data.sort(key=lambda x: x['quality'], reverse=True)
    top_embeddings = embeddings_data[:min(len(embeddings_data), 10)]
    
    # Traditional embedding processing
    embeddings = np.array([item['embedding'] for item in top_embeddings])
    weights = np.array([item['quality'] for item in top_embeddings])
    weights = weights / np.sum(weights)
    
    weighted_centroid = np.average(embeddings, axis=0, weights=weights)
    weighted_centroid = weighted_centroid / (np.linalg.norm(weighted_centroid) + 1e-8)
    
    # Base template
    template = {
        'centroid': weighted_centroid,
        'individual_embeddings': embeddings,
        'weights': weights,
        'quality_scores': [item['quality'] for item in top_embeddings],
        'template_type': 'standard'
    }
    
    # Add 3D reconstruction if available and images provided
    if USE_3D_RECONSTRUCTION and images is not None and len(images) >= 3:
        try:
            print("🔧 Creating advanced 3D face template...")
            
            # Create 3D template
            template_3d = create_advanced_3d_template(images)
            
            if template_3d:
                # Merge 3D features with traditional template
                template.update({
                    'landmarks_3d': template_3d['landmarks_3d'],
                    'features_3d': template_3d['features'],
                    'descriptor_3d': template_3d['descriptor'],
                    'pose_variation': template_3d['pose_variation'],
                    'confidence_3d': template_3d['confidence'],
                    'template_type': '3D_advanced',
                    'version': '2.0'
                })
                
                print(f"✅ 3D template created with confidence: {template_3d['confidence']:.3f}")
            else:
                print("⚠️ 3D template creation failed, using standard template")
                
        except Exception as e:
            print(f"⚠️ 3D reconstruction error: {e}, falling back to standard template")
    
    return template

def continuous_learning_update(name, new_embedding, confidence):
    """Continuously improve embeddings with new successful recognitions"""
    try:
        person = persons_col.find_one({"name": name})
        if not person:
            return False
        
        if confidence < 0.70:  # Lowered threshold for occluded faces
            return False
        
        stored_template = pickle.loads(person['embedding'])
        
        if isinstance(stored_template, dict) and 'centroid' in stored_template:
            current_centroid = stored_template['centroid']
        else:
            current_centroid = stored_template
        
        # Exponential moving average
        alpha = 0.15  # Increased for faster adaptation to occlusions
        updated_centroid = (1 - alpha) * current_centroid + alpha * new_embedding
        updated_centroid = updated_centroid / (np.linalg.norm(updated_centroid) + 1e-8)
        
        update_count = person.get('update_count', 0) + 1
        
        updated_template = {
            'centroid': updated_centroid,
            'last_updated': datetime.datetime.now(),
            'update_count': update_count
        }
        
        # Update in MongoDB
        persons_col.update_one(
            {"name": name},
            {
                "$set": {
                    "embedding": pickle.dumps(updated_template),
                    "last_updated": datetime.datetime.now(),
                    "update_count": update_count
                }
            }
        )
        
        # Rebuild FAISS periodically
        if update_count % 30 == 0:  # More frequent rebuilds
            rebuild_faiss_index()
        
        print(f"[Continuous Learning] Updated template for {name} (confidence: {confidence:.3f})")
        return True
    except Exception as e:
        print(f"[Continuous Learning Error] {e}")
        return False

def ensemble_matching(query_embedding, stored_template, threshold=0.55):
    """Enhanced matching with lower threshold for occluded faces"""
    scores = []
    
    if isinstance(stored_template, dict) and 'centroid' in stored_template:
        centroid = stored_template['centroid']
        centroid_score = cosine_sim(query_embedding, centroid)
        scores.append(centroid_score)
        
        if 'individual_embeddings' in stored_template:
            individual_embeddings = stored_template['individual_embeddings']
            for emb in individual_embeddings:
                score = cosine_sim(query_embedding, emb)
                scores.append(score)
    else:
        score = cosine_sim(query_embedding, stored_template)
        scores.append(score)
    
    if len(scores) >= 3:
        top_3 = sorted(scores, reverse=True)[:3]
        return np.mean(top_3)
    else:
        return max(scores) if scores else 0.0

def initialize_faiss_index():
    """Initialize FAISS index with HNSW for better performance"""
    global faiss_index, person_id_map
    
    index_path = "faiss_index.bin"
    map_path = "person_id_map.pkl"
    
    if os.path.exists(index_path) and os.path.exists(map_path):
        try:
            faiss_index = faiss.read_index(index_path)
            with open(map_path, 'rb') as f:
                person_id_map = pickle.load(f)
            print(f"✅ Loaded HNSW FAISS index with {faiss_index.ntotal} faces")
        except Exception as e:
            print(f"[FAISS Load Error] {e}, creating new index")
            # Use HNSW for faster approximate search
            faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)  # 32 neighbors
            faiss_index.hnsw.efConstruction = 200  # Build quality
            faiss_index.hnsw.efSearch = 64  # Search quality
            person_id_map = []
    else:
        # Create HNSW index for better performance with large datasets
        faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
        faiss_index.hnsw.efConstruction = 200
        faiss_index.hnsw.efSearch = 64
        person_id_map = []
        print("✅ Created new HNSW FAISS index")

def save_faiss_index():
    """Save FAISS index to disk"""
    try:
        faiss.write_index(faiss_index, "faiss_index.bin")
        with open("person_id_map.pkl", 'wb') as f:
            pickle.dump(person_id_map, f)
        print(f"💾 FAISS index saved ({len(person_id_map)} persons)")
    except Exception as e:
        print(f"[FAISS Save Error] {e}")

def rebuild_faiss_index():
    """Rebuild FAISS HNSW index from MongoDB"""
    global faiss_index, person_id_map
    
    # Use HNSW for better performance
    faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
    faiss_index.hnsw.efConstruction = 200
    faiss_index.hnsw.efSearch = 64
    person_id_map = []
    
    # Get persons from MongoDB (exclude blocked users)
    persons = list(persons_col.find({"status": {"$ne": "blocked"}}))
    
    for person in persons:
        try:
            stored_data = pickle.loads(person['embedding'])
            
            if isinstance(stored_data, dict) and 'centroid' in stored_data:
                embedding = stored_data['centroid']
            else:
                embedding = stored_data
            
            # Ensure embedding dimension matches
            if len(embedding) > EMBEDDING_DIM:
                embedding = embedding[:EMBEDDING_DIM]
            elif len(embedding) < EMBEDDING_DIM:
                padding = np.zeros(EMBEDDING_DIM - len(embedding))
                embedding = np.concatenate([embedding, padding])
            
            # Normalize embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            # Add to FAISS index
            faiss_index.add(np.array([embedding], dtype=np.float32))
            person_id_map.append(person['name'])
        except Exception as e:
            print(f"[Index Rebuild] Error processing {person.get('name', 'unknown')}: {e}")
            continue
    
    save_faiss_index()
    print(f"[FAISS] Rebuilt HNSW index with {len(person_id_map)} persons")
    
    return len(person_id_map)

def search_face_faiss(embedding, threshold=0.55, image=None):
    """Enhanced FAISS search with 3D reconstruction support"""
    if faiss_index is None or faiss_index.ntotal == 0:
        return None, 0.0
    
    # Normalize embedding
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    
    # Search in FAISS index - get top 5 for better matching
    k = min(5, faiss_index.ntotal)
    distances, indices = faiss_index.search(
        np.array([embedding], dtype=np.float32),
        k=k
    )
    
    best_name = None
    best_score = 0.0
    
    # Extract 3D features if image provided and 3D reconstruction enabled
    face_3d_features = None
    if USE_3D_RECONSTRUCTION and image is not None:
        try:
            face_3d_features = extract_3d_face_features(image)
            if face_3d_features:
                print("🔧 Using 3D face analysis for enhanced matching")
        except Exception as e:
            print(f"⚠️ 3D feature extraction failed: {e}")
    
    try:
        for i in range(len(distances[0])):
            if distances[0][i] < threshold:
                continue
            
            matched_name = person_id_map[indices[0][i]]
            
            # Get person from MongoDB
            person = persons_col.find_one({"name": matched_name})
            if person:
                stored_template = pickle.loads(person['embedding'])
                
                # Traditional ensemble matching
                ensemble_score = ensemble_matching(embedding, stored_template, threshold)
                
                # Enhanced 3D matching if available
                if (face_3d_features and 
                    isinstance(stored_template, dict) and 
                    stored_template.get('template_type') == '3D_advanced'):
                    
                    try:
                        # Create temporary 3D template for current face
                        current_3d_template = {
                            'landmarks_3d': face_3d_features['landmarks_3d'],
                            'features': face_3d_features['features'],
                            'descriptor': face_3d_features.get('descriptor', np.zeros(100)),
                            'confidence': face_3d_features['quality_score']
                        }
                        
                        # 3D template matching
                        similarity_3d = match_3d_face_templates(current_3d_template, stored_template)
                        
                        # Combine 2D and 3D scores
                        combined_score = (ensemble_score * 0.6 + similarity_3d * 0.4)
                        
                        print(f"🔍 Enhanced matching: 2D={ensemble_score:.3f}, 3D={similarity_3d:.3f}, Combined={combined_score:.3f}")
                        
                        # Use combined score if 3D matching was successful
                        if similarity_3d > 0.3:  # Minimum 3D confidence
                            ensemble_score = combined_score
                            
                    except Exception as e:
                        print(f"⚠️ 3D matching error: {e}, using 2D score")
                
                if ensemble_score > best_score:
                    best_score = ensemble_score
                    best_name = matched_name
        
        # Continuous learning with relaxed threshold
        if best_name and best_score >= 0.70:
            continuous_learning_update(best_name, embedding, best_score)
        
        if best_score >= threshold:
            detection_stats['successful_recognitions'] += 1
            return best_name, best_score
        else:
            detection_stats['unknown_persons'] += 1
    except Exception as e:
        print(f"[Search Error] {e}")
    
    return None, 0.0

def generate_camera_stream():
    """Enhanced camera stream with CROWD DETECTION and OCCLUSION handling"""
    cap = cv.VideoCapture(0)
    threshold = 0.55  # Lowered for occluded faces
    marked_attendance = {}
    cooldown_period = 300  # 5 minutes
    
    # Performance tracking
    fps_queue = deque(maxlen=30)
    last_time = datetime.datetime.now()
    
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return
    
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break
        
        frame_count += 1
        
        # Calculate FPS
        current_time = datetime.datetime.now()
        time_diff = (current_time - last_time).total_seconds()
        if time_diff > 0:
            fps = 1.0 / time_diff
            fps_queue.append(fps)
        last_time = current_time
        
        avg_fps = np.mean(fps_queue) if fps_queue else 0
        
        # Process every frame for crowd detection
        try:
            # Detect ALL faces in frame (CROWD SUPPORT)
            face_objs = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="retinaface",  # Better for crowded scenes
                enforce_detection=False,
                align=True
            )
            
            detection_stats['total_detections'] += len(face_objs)
            detected_count = len(face_objs)
            recognized_count = 0
            unknown_count = 0
            
            # Process MULTIPLE faces simultaneously
            for face_obj in face_objs:
                facial_area = face_obj["facial_area"]
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                
                # Ensure valid coordinates
                x, y = max(0, x), max(0, y)
                w, h = min(w, frame.shape[1]-x), min(h, frame.shape[0]-y)
                
                face = frame[y:y+h, x:x+w]
                
                if face.size == 0:
                    continue
                
                # Check for occlusion
                is_occluded, occlusion_score = detect_occlusion(face)
                
                # Liveness detection (relaxed for occluded faces)
                if not is_real_face(face):
                    cv.putText(frame, "Fake/Poor Quality", (x, y-10),
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    continue
                
                # Get embedding with occlusion handling
                embedding = get_face_embedding(face, handle_occlusion=True)
                if embedding is None:
                    continue
                
                # Search in FAISS
                matched_name, confidence = search_face_faiss(embedding, threshold)
                
                if matched_name:
                    recognized_count += 1
                    # Get person details from MongoDB
                    person = persons_col.find_one({"name": matched_name})
                    
                    # Check if blocked
                    if person and person.get("status") == "blocked":
                        label = f"{matched_name} - BLOCKED"
                        color = (0, 0, 255)
                        cv.putText(frame, label, (x, y-10),
                                  cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                        continue
                    
                    template_type = person.get("template_type", "Standard") if person else "Standard"
                    
                    label = f"{matched_name} ({confidence:.2f})"
                    if is_occluded:
                        label += " [Masked]"
                    
                    color = (0, 255, 0)
                    
                    # Mark attendance with cooldown
                    current_time = datetime.datetime.now()
                    last_marked = marked_attendance.get(matched_name)
                    
                    if not last_marked or (current_time - last_marked).total_seconds() > cooldown_period:
                        # Insert attendance to MongoDB
                        attendance_col.insert_one({
                            "name": matched_name,
                            "timestamp": current_time.isoformat(),
                            "confidence": confidence,
                            "template_type": template_type,
                            "occluded": is_occluded,
                            "occlusion_score": float(occlusion_score),
                            "method": "auto"
                        })
                        marked_attendance[matched_name] = current_time
                        print(f"✓ Attendance marked for {matched_name} (Conf: {confidence:.3f}, Occluded: {is_occluded})")
                    
                    # Draw on frame
                    cv.putText(frame, label, (x, y-10),
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Green corner markers for recognized
                    cv.circle(frame, (x+5, y+5), 5, color, -1)
                    cv.circle(frame, (x+w-5, y+5), 5, color, -1)
                else:
                    unknown_count += 1
                    # Unknown person
                    label = "Unknown"
                    if is_occluded:
                        label += " [Masked]"
                    color = (0, 165, 255)
                    cv.putText(frame, label, (x, y-10),
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
        except Exception as e:
            print(f"[Recognition Error] {e}")
        
        # Enhanced System Info Overlay
        total_faces = faiss_index.ntotal if faiss_index else 0
        info_y = 30
        
        # Background panel for info
        cv.rectangle(frame, (5, 5), (400, 140), (0, 0, 0), -1)
        cv.rectangle(frame, (5, 5), (400, 140), (100, 100, 100), 2)
        
        cv.putText(frame, f"FAISS Index: {total_faces} faces", (10, info_y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += 25
        cv.putText(frame, f"FPS: {avg_fps:.1f} | Detected: {detected_count}", (10, info_y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        info_y += 25
        cv.putText(frame, f"Recognized: {recognized_count} | Unknown: {unknown_count}", (10, info_y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        info_y += 25
        cv.putText(frame, "Continuous Learning: ACTIVE", (10, info_y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        info_y += 25
        cv.putText(frame, "Crowd & Occlusion Support: ON", (10, info_y),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
        
        # Encode and yield frame
        ret, buffer = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    
    cap.release()

# Initialize on import
initialize_faiss_index()
