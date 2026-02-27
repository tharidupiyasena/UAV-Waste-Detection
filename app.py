from flask import Flask, render_template, request, jsonify, send_from_directory
import os, uuid, json, shutil, subprocess, sys
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# ── Folders ──────────────────────────────────────────────────────────────────
UPLOAD_FOLDER       = 'static/uploads'
RESULTS_FOLDER      = 'static/results'
ANNOTATION_FOLDER   = 'annotation_images'
LABEL_STUDIO_DATA   = 'label_studio_data'
DB_FILE             = 'db.json'          # lightweight JSON "database"

for d in [UPLOAD_FOLDER, RESULTS_FOLDER, ANNOTATION_FOLDER, LABEL_STUDIO_DATA]:
    os.makedirs(d, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024   # 64 MB

ALLOWED = {'png', 'jpg', 'jpeg', 'webp'}

# ── Model + SAHI ─────────────────────────────────────────────────────────────
detection_model = None

def load_model():
    global detection_model
    try:
        from sahi import AutoDetectionModel
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path='weights/best.pt',
            confidence_threshold=0.25,
            device='cpu'
        )
        print("✅  SAHI + YOLOv8 model loaded")
    except Exception as e:
        print(f"⚠️  Model load failed: {e}")
        detection_model = None

load_model()

# ── Tiny JSON DB helpers ──────────────────────────────────────────────────────
def db_load():
    if not os.path.exists(DB_FILE):
        return {"images": {}}
    with open(DB_FILE) as f:
        return json.load(f)

def db_save(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def db_upsert(record: dict):
    d = db_load()
    d["images"][record["id"]] = record
    db_save(d)

# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

def run_sahi(image_path: str):
    """Run SAHI sliced inference, return (annotated_path | None, detections list)."""
    from sahi.predict import get_sliced_prediction
    import cv2

    result = get_sliced_prediction(
        image          = image_path,
        detection_model= detection_model,
        slice_height   = 640,
        slice_width    = 640,
        overlap_height_ratio = 0.25,
        overlap_width_ratio  = 0.25,
        verbose        = 0
    )

    detections = []
    for obj in result.object_prediction_list:
        detections.append({
            "class":      obj.category.name,
            "confidence": round(float(obj.score.value), 4),
            "bbox":       [
                obj.bbox.minx, obj.bbox.miny,
                obj.bbox.maxx, obj.bbox.maxy
            ]
        })

    if not detections:
        return None, []

    # Draw boxes with OpenCV
    img = cv2.imread(image_path)
    for d in detections:
        x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,80), 3)
        label = f"{d['class']} {d['confidence']:.2f}"
        cv2.putText(img, label, (x1, max(y1-8,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,80), 2)

    out_name = f"res_{Path(image_path).name}"
    out_path = os.path.join(RESULTS_FOLDER, out_name)
    cv2.imwrite(out_path, img)
    return out_path, detections

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/images')
def api_images():
    d = db_load()
    images = sorted(d["images"].values(),
                    key=lambda x: x.get("created_at",""), reverse=True)
    return jsonify(images)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image received"}), 400

    file = request.files['image']
    if not file.filename or not allowed(file.filename):
        return jsonify({"success": False, "error": "Invalid file type"}), 400

    if detection_model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    img_id   = str(uuid.uuid4())
    filename = f"{img_id}.jpg"
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    try:
        ann_path, detections = run_sahi(img_path)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

    annotated_url = None
    if ann_path:
        annotated_url = f"/{ann_path}"

    record = {
        "id":              img_id,
        "filename":        filename,
        "original_url":    f"/static/uploads/{filename}",
        "annotated_url":   annotated_url,
        "total_detections":len(detections),
        "detections":      detections,
        "annotation_status": "not_annotated",
        "created_at":      datetime.utcnow().isoformat()
    }
    db_upsert(record)

    return jsonify({
        "success":            True,
        "id":                 img_id,
        "total_detections":   len(detections),
        "original_image_url": record["original_url"],
        "annotated_image_url":annotated_url,
        "detections":         detections
    })

@app.route('/annotate/<img_id>', methods=['POST'])
def annotate(img_id):
    """Copy image to annotation_images/ and open Label Studio."""
    d = db_load()
    if img_id not in d["images"]:
        return jsonify({"success": False, "error": "Image not found"}), 404

    record   = d["images"][img_id]
    src_path = record["original_url"].lstrip('/')          # static/uploads/xxx.jpg
    dst_path = os.path.join(ANNOTATION_FOLDER, record["filename"])

    if not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)

    record["annotation_status"] = "pending"
    db_upsert(record)

    # Return the Label Studio URL so the frontend can open it
    ls_url = "http://localhost:8080"
    return jsonify({"success": True, "label_studio_url": ls_url,
                    "image_copied_to": dst_path})

@app.route('/api/mark_annotated/<img_id>', methods=['POST'])
def mark_annotated(img_id):
    d = db_load()
    if img_id not in d["images"]:
        return jsonify({"success": False, "error": "Not found"}), 404
    d["images"][img_id]["annotation_status"] = "annotated"
    db_save(d)
    return jsonify({"success": True})

@app.route('/static/<path:p>')
def static_files(p):
    return send_from_directory('static', p)

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
