from flask import Flask, request, jsonify, render_template, send_from_directory, redirect
from flask_cors import CORS
import os
import sqlite3
import face_recognition
import numpy as np
import pickle
import base64
from io import BytesIO
from PIL import Image
import csv
import logging

app = Flask(__name__)

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_FOLDER = os.path.join(BASE_DIR, "pre_registered", "photos")
DB_PATH = os.path.join(BASE_DIR, "pre_registered", "database.db")
CSV_PATH = os.path.join(BASE_DIR, "pre_registered", "users.csv")

# Max upload size (10 MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Matching threshold: can be overridden by env var or per-request
DEFAULT_THRESHOLD = float(os.environ.get("THRESHOLD", 0.65))

# Enable CORS — tighten origins in production!
CORS(app, resources={r"/*": {"origins": "*"}})

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_scan")

os.makedirs(PHOTO_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# === Initialize database ===
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    organization TEXT,
                    phone TEXT,
                    email TEXT,
                    photo_path TEXT,
                    face_encoding BLOB
                )''')
    conn.commit()
    conn.close()

init_db()

# --------------------
# Routes
# --------------------
@app.route('/')
def home():
    return redirect('/preregister')


@app.route('/pre_registered/photos/<path:filename>')
def serve_photos(filename):
    return send_from_directory(PHOTO_FOLDER, filename)


# ===========================
# REGISTER NEW USER
# ===========================
@app.route('/preregister', methods=['GET', 'POST'])
def preregister():
    if request.method == 'GET':
        return render_template('pre_register.html')

    try:
        name = request.form.get('name', '').strip()
        organization = request.form.get('organization', '').strip()
        phone = request.form.get('phone', '').strip()
        email = request.form.get('email', '').strip()
        photo = request.files.get('photo')

        # === Validate all fields ===
        if not all([name, organization, phone, email, photo]):
            return jsonify({'success': False, 'message': 'All fields are required.'})

        # === Save uploaded photo ===
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        filename = f"{safe_name}_{phone}.png"
        filepath = os.path.join(PHOTO_FOLDER, filename)
        photo.save(filepath)

        # === Force convert to RGB (important fix) ===
        try:
            img = Image.open(filepath).convert('RGB')
            img.save(filepath)
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            logger.exception("Image conversion failed")
            return jsonify({'success': False, 'message': 'Invalid or unsupported image format.'})

        # === Extract face encoding ===
        img_array = np.array(img)
        encodings = face_recognition.face_encodings(img_array)
        if not encodings:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'message': 'No face detected in photo. Please upload a clear face photo.'})
        face_encoding = encodings[0]

        # === Store in database ===
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO users (name, organization, phone, email, photo_path, face_encoding) VALUES (?, ?, ?, ?, ?, ?)',
                  (name, organization, phone, email, os.path.join("pre_registered", "photos", filename), pickle.dumps(face_encoding)))
        conn.commit()
        conn.close()

        # === Save in CSV file ===
        file_exists = os.path.isfile(CSV_PATH)
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Name', 'Organization', 'Phone', 'Email', 'Photo Path'])
            writer.writerow([name, organization, phone, email, os.path.join("pre_registered", "photos", filename)])

        return jsonify({'success': True, 'message': f'{name} registered successfully!'})

    except Exception as e:
        logger.exception("Error in /preregister")
        return jsonify({'success': False, 'message': 'Registration failed. Check server logs for details.'})


# ===========================
# SCAN FACE AND MATCH
# - Accepts multipart/form-data (file) OR JSON with data URL
# - Picks the largest detected face if multiple
# - threshold override via ?threshold=0.7 or form field 'threshold'
# ===========================
def _load_image_from_request(req):
    """
    Returns PIL Image or raises ValueError.
    Accepts:
      - multipart/form-data with 'file' field
      - JSON body with {'image': 'data:image/png;base64,...'}
    """
    # 1) check files (preferred)
    if 'file' in req.files:
        file = req.files.get('file')
        img_bytes = file.read()
        try:
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
            return img
        except Exception as e:
            raise ValueError("Uploaded file is not a valid image") from e

    # 2) check JSON data
    if req.is_json:
        data = req.get_json(silent=True) or {}
        img_b64 = data.get('image')
        if not img_b64:
            raise ValueError("JSON must include 'image' field with data URL")
        # remove prefix if present
        if ',' in img_b64:
            img_b64 = img_b64.split(',', 1)[1]
        try:
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
            return img
        except Exception as e:
            raise ValueError("Invalid base64 image data") from e

    # 3) no supported body
    raise ValueError("No image found in request (expecting file upload or JSON base64)")

def _downscale_if_needed(img, max_size=1200):
    # reduce very large images to speed up detection, maintain aspect ratio
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        logger.info(f"Downscaling image from {(w,h)} to {new_size} for faster detection")
        return img.resize(new_size, Image.LANCZOS)
    return img

@app.route('/scan_face', methods=['POST'])
def scan_face():
    try:
        # threshold override order: query param -> form field -> default env var
        try:
            q_threshold = request.args.get('threshold')
            f_threshold = request.form.get('threshold')
            if q_threshold:
                threshold = float(q_threshold)
            elif f_threshold:
                threshold = float(f_threshold)
            else:
                threshold = DEFAULT_THRESHOLD
        except Exception:
            threshold = DEFAULT_THRESHOLD

        # parse image
        try:
            img = _load_image_from_request(request)
        except ValueError as ve:
            logger.warning("Bad request image: %s", ve)
            return jsonify({'found': False, 'error': str(ve)}), 400

        # optional downscale (keeps face pixels reasonable but speeds up)
        img = _downscale_if_needed(img, max_size=1200)
        img_array = np.array(img)

        # detect face locations; pick largest face if multiple
        face_locations = face_recognition.face_locations(img_array)
        if not face_locations:
            logger.info("No faces found in frame")
            return jsonify({'found': False})

        # choose the largest face region
        def box_area(box):
            top, right, bottom, left = box
            return max(0, bottom - top) * max(0, right - left)

        chosen_box = max(face_locations, key=box_area)
        # compute encoding for chosen face only
        encodings_in_frame = face_recognition.face_encodings(img_array, [chosen_box])
        if not encodings_in_frame:
            logger.info("Face located but could not compute encoding")
            return jsonify({'found': False})
        frame_encoding = encodings_in_frame[0]

        # load users
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name, organization, phone, email, photo_path, face_encoding FROM users")
        users = c.fetchall()
        conn.close()

        # matching loop
        best_match = None
        best_distance = threshold  # start at threshold (smaller is better)

        for user in users:
            try:
                name, org, phone, email, photo_path, face_enc_blob = user
                face_enc = pickle.loads(face_enc_blob)
            except Exception as e:
                logger.warning("Skipping user row due to decode error: %s", e)
                continue

            # compute distance
            try:
                distance = float(face_recognition.face_distance([face_enc], frame_encoding)[0])
            except Exception as e:
                logger.exception("Error computing distance")
                continue

            # If distance is strictly smaller than best_distance, it's a better match
            if distance < best_distance:
                best_distance = distance
                best_match = {
                    'name': name,
                    'organization': org,
                    'phone': phone,
                    'email': email,
                    'photo_path': '/' + photo_path.replace("\\", "/"),
                    'distance': round(distance, 4)
                }

        if best_match:
            logger.info("Match found: %s (distance=%.4f, threshold=%.3f)", best_match['name'], best_distance, threshold)
            return jsonify({'found': True, **best_match})
        else:
            logger.info("No match below threshold %.3f (best distance seen: %.4f)", threshold, best_distance if best_distance != threshold else 1.0)
            return jsonify({'found': False})

    except Exception as e:
        logger.exception("Error in /scan_face")
        return jsonify({'found': False, 'error': 'Internal server error'}), 500


# ===========================
# EVENT PAGE
# ===========================
@app.route('/event')
def event_page():
    return render_template('event.html')


# ===========================
# LATEST ATTENDEE
# ===========================
@app.route('/latest_attendee')
def latest_attendee():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name, organization, photo_path FROM users ORDER BY id DESC LIMIT 1")
        user = c.fetchone()
        conn.close()

        if user:
            name, org, photo = user
            return jsonify({
                'name': name,
                'organization': org,
                'photo_path': '/' + photo.replace("\\", "/")
            })
        return jsonify({})
    except Exception as e:
        logger.exception("Error in /latest_attendee")
        return jsonify({})


# ===========================
# RUN FLASK APP
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = True if os.environ.get("PORT") is None else False
    logger.info("Starting app on 0.0.0.0:%d (debug=%s) threshold_default=%.3f", port, debug_mode, DEFAULT_THRESHOLD)
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
