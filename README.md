# 🛸 UAV Waste Detection — Dengue Breeding Point Identifier

> A research-driven web application that leverages **UAV (drone) imagery** and **YOLOv8** deep learning to detect waste and identify potential **dengue mosquito breeding points** from an aerial perspective.

---

## 📌 Overview

Dengue fever remains a critical public health concern in many tropical regions. One of the primary contributors to mosquito breeding is stagnant water accumulation in improperly disposed waste. This project automates the detection of such waste using drone-captured images processed by a fine-tuned YOLOv8 object detection model, served through a lightweight Flask web application.

This repository is the **research component** of a larger system aimed at enabling rapid, scalable environmental surveillance without requiring on-ground inspection.

---

## ✨ Features

- 🔍 **YOLOv8-powered detection** — Fine-tuned model (`best.pt`) trained on UAV imagery for waste detection
- 🌐 **Flask web interface** — Upload images and receive detection results directly in the browser
- 🏷️ **Label Studio integration** — Annotation config included (`label_studio_config.xml`) for dataset labeling workflow
- ☁️ **Railway deployment ready** — Pre-configured with `railway.json`, `railway.toml`, `nixpacks.toml`, and `Procfile` for one-click cloud deployment
- 📦 **Minimal dependencies** — Runs on standard Python environment with clear `requirements.txt`

---

## 🗂️ Project Structure

```
UAV-Waste-Detection/
│
├── app.py                        # Flask application entry point
├── best.pt                       # Trained YOLOv8 model weights
├── label_studio_config.xml       # Label Studio annotation configuration
│
├── templates/                    # HTML templates for Flask UI
├── static/                       # Static assets (CSS, JS, images)
├── weights/                      # Additional model weight files
│
├── requirements.txt              # Python dependencies
├── packages.txt                  # System-level package dependencies
├── Procfile                      # Process file for deployment
├── nixpacks.toml                 # Nixpacks build configuration
├── railway.json                  # Railway deployment config
├── railway.toml                  # Railway deployment settings
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/tharidupiyasena/UAV-Waste-Detection.git
cd UAV-Waste-Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> If any system packages are required (e.g., `libgl1`), install them via:
> ```bash
> cat packages.txt | xargs sudo apt-get install -y
> ```

### 3. Run the Application

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`.

---

## 🧠 Model

The detection model (`best.pt`) is a custom-trained **YOLOv8** model fine-tuned on UAV aerial images annotated for waste objects that can serve as dengue mosquito breeding sites (e.g., discarded containers, tyres, plastic waste with water accumulation potential).

Annotations were managed using **Label Studio** with the configuration provided in `label_studio_config.xml`.

---

## ☁️ Deployment

This project is configured for deployment on **[Railway](https://railway.app/)**.

### Deploy to Railway

1. Fork this repository
2. Connect your GitHub account to Railway
3. Create a new project and select this repository
4. Railway will automatically detect the configuration and deploy

The app is served via **Gunicorn**:

```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Web Framework | Flask |
| WSGI Server | Gunicorn |
| Annotation Tool | Label Studio |
| Deployment | Railway |
| Languages | Python, HTML, CSS |

---

## 📊 Use Case

This system is designed to support:

- **Public health authorities** monitoring dengue risk areas
- **Environmental agencies** conducting aerial waste surveys
- **Researchers** studying computer vision applications in epidemiology

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open-source. Please check the repository for license details or contact the author for usage permissions.

---

## 👤 Author

**Tharidu Piyasena**
- GitHub: [@tharidupiyasena](https://github.com/tharidupiyasena)

---

> *This project is part of a research initiative on using UAV technology and deep learning for dengue vector surveillance and environmental monitoring.*
