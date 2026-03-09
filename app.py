from flask import Flask, render_template, request, jsonify, send_from_directory
import os, uuid, json, shutil, requests, traceback
from pathlib import Path
from datetime import datetime
import cv2
import psutil

# ── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Label Studio Cloud Configuration ─────────────────────────────────────────
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "https://label-studio-production-9ece.up.railway.app")
LABEL_STUDIO_API_KEY = os.environ.get("LABEL_STUDIO_TOKEN", "")
LABEL_STUDIO_PROJECT_ID = 1

# ── Folders ──────────────────────────────────────────────────────────────────
UPLOAD_FOLDER       = 'static/uploads'
RESULTS_FOLDER      = 'static/results'
ANNOTATION_FOLDER   = 'annotation_images'
LABEL_STUDIO_DATA   = 'label_studio_data'
DB_FILE             = 'db.json'

for d in [UPLOAD_FOLDER, RESULTS_FOLDER, ANNOTATION_FOLDER, LABEL_STUDIO_DATA]:
    os.makedirs(d, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

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
    """Run SAHI sliced inference"""
    from sahi.predict import get_sliced_prediction

    print(f"[MEM] Before SAHI inference: {psutil.Process().memory_info().rss / 1024**2:.1f} MB")

    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.15,
        overlap_width_ratio=0.15,
        verbose=0,
        postprocess_type="GREEDYNMM",
        postprocess_match_threshold=0.5
    )

    print(f"[MEM] After SAHI inference:  {psutil.Process().memory_info().rss / 1024**2:.1f} MB")

    detections = []
    for obj in result.object_prediction_list:
        detections.append({
            "class":      obj.category.name,
            "confidence": round(float(obj.score.value), 4),
            "bbox":       [obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy]
        })

    if not detections:
        return None, []

    img = cv2.imread(image_path)
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 80), 3)
        label = f"{d['class']} {d['confidence']:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2)

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
    images = sorted(d["images"].values(), key=lambda x: x.get("created_at",""), reverse=True)
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
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

    annotated_url = f"/{ann_path}" if ann_path else None

    record = {
        "id":              img_id,
        "filename":        filename,
        "original_url":    f"/static/uploads/{filename}",
        "annotated_url":   annotated_url,
        "total_detections": len(detections),
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
        "annotated_image_url": annotated_url,
        "detections":         detections
    })

# ── Cloud Label Studio Annotate Route ────────────────────────────────────────
@app.route('/annotate/<img_id>', methods=['POST'])
def annotate(img_id):
    d = db_load()
    if img_id not in d["images"]:
        return jsonify({"success": False, "error": "Image not found"}), 404

    record = d["images"][img_id]
    src_path = record["original_url"].lstrip('/')
    dst_path = os.path.join(ANNOTATION_FOLDER, record["filename"])

    if not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)

    public_image_url = f"{request.host_url.rstrip('/')}{record['original_url']}"

    # Get real image dimensions
    try:
        img = cv2.imread(src_path)
        h, w = img.shape[:2]
    except:
        w, h = 1000, 1000

    results = []
    for det in record.get("detections", []):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        results.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": (x1 / w) * 100,
                "y": (y1 / h) * 100,
                "width": ((x2 - x1) / w) * 100,
                "height": ((y2 - y1) / h) * 100,
                "rectanglelabels": [det["class"]]
            }
        })

    headers = {"Authorization": f"Bearer {LABEL_STUDIO_API_KEY}"}
    task_data = {
        "project": LABEL_STUDIO_PROJECT_ID,
        "data": {"image": public_image_url},
        "predictions": [{"result": results}] if results else []
    }

    try:
        r = requests.post(
            f"{LABEL_STUDIO_URL}/api/projects/{LABEL_STUDIO_PROJECT_ID}/tasks",
            json=task_data,
            headers=headers,
            timeout=10
        )
        r.raise_for_status()
        task = r.json()
        task_url = f"{LABEL_STUDIO_URL}/tasks/{task['id']}"
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Label Studio error: {str(e)}"}), 500

    record["annotation_status"] = "pending"
    db_upsert(record)

    return jsonify({
        "success": True,
        "label_studio_url": task_url,
        "task_id": task["id"]
    })

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
