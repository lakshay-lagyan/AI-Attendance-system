import cv2 as cv
import numpy as np
from deepface import DeepFace
import os
import faiss
import datetime
import pickle
from pymongo import MongoClient

# MongoDB connection (use environment variable)
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(MONGODB_URI)

# Database references
transactional_db = client["transactional_db"]
attendance_col = transactional_db["attendance"]

core = client['secure_db']
persons_col = core["persons"]

# FAISS setup
EMBEDDING_DIM = 512
faiss_index = None
person_id_map = []

def cosine_sim(a, b) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def is_real_face(face_img):
    """Enhanced liveness detection"""
    try:
        if face_img is None or face_img.size == 0:
            return False
        
        gray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
        
        # 1. Laplacian variance (blur detection)
        laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
        if laplacian_var < 50:
            return False
        
        # 2. Check brightness
        brightness = np.mean(gray)
        if brightness < 30 or brightness > 225:
            return False
        
        # 3. Check contrast
        contrast = np.std(gray)
        if contrast < 20:
            return False
        
        # 4. Edge detection
        edges = cv.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density < 0.01:
            return False
        
        return True
    except Exception as e:
        print(f"[Liveness Error] {e}")
        return False

def get_face_embedding(image):
    """Extract face embedding using DeepFace ArcFace"""
    try:
        if hasattr(image, 'read'):
            file_bytes = np.frombuffer(image.read(), np.uint8)
            image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        
        if image is None or image.size == 0:
            return None
        
        if not is_real_face(image):
            print("[Warning] Fake face or low quality detected")
            return None
        
        embedding_obj = DeepFace.represent(
            img_path=image,
            model_name='ArcFace',
            detector_backend='opencv',
            enforce_detection=False
        )
        
        if not embedding_obj:
            return None
        
        embedding = np.array(embedding_obj[0]['embedding'])
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
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

def create_3d_template(embeddings_data):
    """Create 3D template from multiple embeddings"""
    if not embeddings_data:
        return None
    
    embeddings_data.sort(key=lambda x: x['quality'], reverse=True)
    top_embeddings = embeddings_data[:min(len(embeddings_data), 10)]
    
    embeddings = np.array([item['embedding'] for item in top_embeddings])
    weights = np.array([item['quality'] for item in top_embeddings])
    weights = weights / np.sum(weights)
    
    weighted_centroid = np.average(embeddings, axis=0, weights=weights)
    weighted_centroid = weighted_centroid / (np.linalg.norm(weighted_centroid) + 1e-8)
    
    return {
        'centroid': weighted_centroid,
        'individual_embeddings': embeddings,
        'weights': weights,
        'quality_scores': [item['quality'] for item in top_embeddings]
    }

def continuous_learning_update(name, new_embedding, confidence):
    """Continuously improve embeddings with new successful recognitions - MongoDB version"""
    try:
        person = persons_col.find_one({"name": name})
        if not person:
            return False
        
        if confidence < 0.75:
            return False
        
        stored_template = pickle.loads(person['embedding'])
        
        if isinstance(stored_template, dict) and 'centroid' in stored_template:
            current_centroid = stored_template['centroid']
        else:
            current_centroid = stored_template
        
        # Exponential moving average
        alpha = 0.1
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
        if update_count % 50 == 0:
            rebuild_faiss_index()
        
        print(f"[Continuous Learning] Updated template for {name} (confidence: {confidence:.3f})")
        return True
    except Exception as e:
        print(f"[Continuous Learning Error] {e}")
        return False

def ensemble_matching(query_embedding, stored_template, threshold=0.6):
    """Enhanced matching using ensemble of embeddings"""
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
    """Initialize FAISS index from disk or create new"""
    global faiss_index, person_id_map
    
    index_path = "faiss_index.bin"
    map_path = "person_id_map.pkl"
    
    if os.path.exists(index_path) and os.path.exists(map_path):
        try:
            faiss_index = faiss.read_index(index_path)
            with open(map_path, 'rb') as f:
                person_id_map = pickle.load(f)
            print(f"✅ Loaded FAISS index with {faiss_index.ntotal} faces")
        except Exception as e:
            print(f"[FAISS Load Error] {e}, creating new index")
            faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            person_id_map = []
    else:
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        person_id_map = []
        print("✅ Created new FAISS index")

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
    """Rebuild FAISS index from MongoDB"""
    global faiss_index, person_id_map
    
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
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
    print(f"[FAISS] Rebuilt index with {len(person_id_map)} persons")
    
    return len(person_id_map)

def search_face_faiss(embedding, threshold=0.6):
    """Enhanced FAISS search with continuous learning - MongoDB version"""
    if faiss_index is None or faiss_index.ntotal == 0:
        return None, 0.0
    
    # Normalize embedding
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    
    # Search in FAISS index
    distances, indices = faiss_index.search(
        np.array([embedding], dtype=np.float32),
        k=min(3, faiss_index.ntotal)
    )
    
    best_name = None
    best_score = 0.0
    
    try:
        for i in range(len(distances[0])):
            if distances[0][i] < threshold:
                continue
            
            matched_name = person_id_map[indices[0][i]]
            
            # Get person from MongoDB
            person = persons_col.find_one({"name": matched_name})
            if person:
                stored_template = pickle.loads(person['embedding'])
                ensemble_score = ensemble_matching(embedding, stored_template, threshold)
                
                if ensemble_score > best_score:
                    best_score = ensemble_score
                    best_name = matched_name
        
        # Continuous learning
        if best_name and best_score >= 0.75:
            continuous_learning_update(best_name, embedding, best_score)
        
        if best_score >= threshold:
            return best_name, best_score
    except Exception as e:
        print(f"[Search Error] {e}")
    
    return None, 0.0

def generate_camera_stream():
    """Enhanced camera stream with continuous learning - MongoDB version"""
    cap = cv.VideoCapture(0)
    threshold = 0.6
    marked_attendance = {}
    cooldown_period = 300  # 5 minutes
    
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return
    
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break
        
        frame_count += 1
        # Process every 2nd frame for performance
        if frame_count % 2 != 0:
            ret, buffer = cv.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            continue
        
        try:
            # Detect faces
            face_objs = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=False,
                align=False
            )
            
            for face_obj in face_objs:
                facial_area = face_obj["facial_area"]
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                face = frame[y:y+h, x:x+w]
                
                if face.size == 0:
                    continue
                
                # Liveness detection
                if not is_real_face(face):
                    cv.putText(frame, "Fake Face!", (x, y-10),
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    continue
                
                # Get embedding
                embedding = get_face_embedding(face)
                if embedding is None:
                    continue
                
                # Search in FAISS
                matched_name, confidence = search_face_faiss(embedding, threshold)
                
                if matched_name:
                    # Get person details from MongoDB
                    person = persons_col.find_one({"name": matched_name})
                    
                    # Check if blocked
                    if person and person.get("status") == "blocked":
                        label = f"{matched_name} - BLOCKED"
                        color = (0, 0, 255)
                        cv.putText(frame, label, (x, y-10),
                                  cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Red overlay
                        overlay = frame.copy()
                        cv.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), -1)
                        frame = cv.addWeighted(overlay, 0.3, frame, 0.7, 0)
                        continue
                    
                    template_type = person.get("template_type", "Standard") if person else "Standard"
                    update_count = person.get("update_count", 0) if person else 0
                    
                    label = f"{matched_name} ({confidence:.2f})"
                    label2 = f"Updates: {update_count} | {template_type}"
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
                            "continuous_learning_active": "true",
                            "method": "auto"
                        })
                        marked_attendance[matched_name] = current_time
                        print(f"✓ Attendance marked for {matched_name} (Conf: {confidence:.3f})")
                    
                    # Draw on frame
                    cv.putText(frame, label, (x, y-30),
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv.putText(frame, label2, (x, y-10),
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                else:
                    # Unknown person
                    label = "Unknown Person"
                    color = (0, 165, 255)
                    cv.putText(frame, label, (x, y-10),
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        except Exception as e:
            print(f"[Recognition Error] {e}")
        
        # System info overlay
        total_faces = faiss_index.ntotal if faiss_index else 0
        cv.putText(frame, f"FAISS Index: {total_faces} faces", (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(frame, "Continuous Learning: ACTIVE", (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Encode and yield frame
        ret, buffer = cv.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    
    cap.release()