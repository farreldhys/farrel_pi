# 🦸‍♂️ Pahlawan Indonesia Detection App

![Streamlit](https://img.shields.io/badge/Streamlit-1.33.0-FF4B4B.svg?style=flat&logo=Streamlit)
![YOLOv9](https://img.shields.io/badge/YOLO-v9-00FFFF.svg?style=flat&logo=YOLO)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?style=flat&logo=Docker)
![Python](https://img.shields.io/badge/Python-3.12.3-3776AB.svg?style=flat&logo=Python)

A professional web application for detecting and classifying Indonesian National Heroes from images. Powered by the state-of-the-art **YOLOv9** object detection model and wrapped in a beautiful, responsive **Streamlit** user interface.

---

## 🌟 Features

- **High Accuracy Detection:** Utilizes a custom-trained YOLOv9 model (`best.pt`) optimized for the Pahlawan Indonesia dataset.
- **Real-time Inference:** Fast and efficient image processing.
- **User-Friendly Interface:** Intuitive Streamlit web interface allowing drag-and-drop image uploads.
- **Dockerized Deployment:** Fully containerized environment for seamless deployment to any Ubuntu server.
- **Detailed Results:** Displays visual bounding boxes alongside prediction confidence scores.

## 🛠️ Technology Stack

- **Frontend/UI:** [Streamlit](https://streamlit.io/)
- **Computer Vision Model:** [Ultralytics YOLOv9](https://docs.ultralytics.com/)
- **Containerization:** Docker
- **Language:** Python 3.12.3

## 🚀 Quick Start (Local Development)

### Prerequisites
Make sure you have Python 3.12.3 installed on your machine.

1. **Navigate to the project directory:**
   ```bash
   cd WEB
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py --server.port=8053
   ```
   The app will be available at `http://localhost:8053`.

## 🐳 Deployment (Ubuntu Server)

Deploying to a production server is streamlined using the provided Docker and bash scripts. 

### Prerequisites
- Ubuntu Server
- Docker installed and running

### Deployment Steps

1. **Transfer the project files** (`app.py`, `requirements.txt`, `Dockerfile`, `deploy.sh`, and the `best.pt` model) to your target server.
2. **Make the deployment script executable:**
   ```bash
   chmod +x deploy.sh
   ```
3. **Execute the deployment script:**
   ```bash
   ./deploy.sh
   ```

The script will automatically build the Docker image and deploy the container in the background. Your application will be live at `http://<YOUR_SERVER_IP>:8053`.

## 📁 Project Structure

```text
WEB/
├── app.py               # Main Streamlit application script
├── best.pt              # Custom-trained YOLOv9 model weights
├── deploy.sh            # Automated Docker deployment script
├── Dockerfile           # Docker configuration file
├── README.md            # Project documentation
└── requirements.txt     # Python package dependencies
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

## 📝 License

This project is licensed under the MIT License.
