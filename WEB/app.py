import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Hero Information Dictionary from Notebook
HERO_HISTORY = [
  {"nama": "Cut Meutia", "kategori": "Pahlawan Kemerdekaan Nasional", "cerita": "Cut Meutia adalah pahlawan wanita dari Aceh yang berjuang melawan penjajahan Belanda dan gugur dalam pertempuran pada tahun 1910."},
  {"nama": "Sisingamangaraja", "kategori": "Pahlawan Kemerdekaan Nasional", "cerita": "Sisingamangaraja adalah raja Batak yang memimpin perlawanan terhadap Belanda di Sumatera Utara hingga gugur pada tahun 1907."},
  {"nama": "Dr. Muwardi", "kategori": "Pahlawan Kemerdekaan Nasional", "cerita": "Dr. Muwardi adalah tokoh pejuang kemerdekaan yang berperan dalam menjaga keamanan saat proklamasi Indonesia."},
  {"nama": "Jenderal Ahmad Yani", "kategori": "Pahlawan Revolusi", "cerita": "Jenderal Ahmad Yani adalah Panglima Angkatan Darat yang gugur dalam peristiwa G30S 1965."},
  {"nama": "Letjen R. Suprapto", "kategori": "Pahlawan Revolusi", "cerita": "Letjen R. Suprapto adalah perwira tinggi TNI yang gugur dalam peristiwa G30S 1965."},
  {"nama": "Letjen MT Haryono", "kategori": "Pahlawan Revolusi", "cerita": "Letjen MT Haryono adalah perwira TNI yang gugur dalam peristiwa G30S 1965."},
  {"nama": "Letjen S. Parman", "kategori": "Pahlawan Revolusi", "cerita": "Letjen S. Parman adalah perwira intelijen TNI yang gugur dalam peristiwa G30S 1965."},
  {"nama": "Mayjen DI Panjaitan", "kategori": "Pahlawan Revolusi", "cerita": "Mayjen DI Panjaitan adalah perwira TNI yang gugur dalam peristiwa G30S 1965."},
  {"nama": "Mayjen Sutoyo Siswomiharjo", "kategori": "Pahlawan Revolusi", "cerita": "Mayjen Sutoyo Siswomiharjo adalah perwira TNI yang gugur dalam peristiwa G30S 1965."},
  {"nama": "Brigjen Katamso Darmokusumo", "kategori": "Pahlawan Revolusi", "cerita": "Brigjen Katamso Darmokusumo adalah komandan TNI yang gugur dalam peristiwa G30S 1965 di Yogyakarta."},
  {"nama": "Kolonel Sugiyono Mangunwiyoto", "kategori": "Pahlawan Revolusi", "cerita": "Kolonel Sugiyono Mangunwiyoto adalah perwira TNI yang gugur dalam peristiwa G30S 1965."},
  {"nama": "Kapten Pierre Tendean", "kategori": "Pahlawan Revolusi", "cerita": "Kapten Pierre Tendean adalah ajudan Ahmad Yani yang gugur dalam peristiwa G30S 1965."},
  {"nama": "A.I.P. II Karel Satsuit Tubun", "kategori": "Pahlawan Revolusi", "cerita": "A.I.P. II Karel Satsuit Tubun adalah anggota polisi yang gugur saat menjalankan tugas dalam peristiwa G30S 1965."},
  {"nama": "Soekarno", "kategori": "Pahlawan Proklamator", "cerita": "Soekarno adalah proklamator kemerdekaan Indonesia dan Presiden pertama Indonesia."},
  {"nama": "Mohammad Hatta", "kategori": "Pahlawan Proklamator", "cerita": "Mohammad Hatta adalah proklamator kemerdekaan Indonesia dan Wakil Presiden pertama Indonesia."},
  {"nama": "Kapitan Pattimura", "kategori": "Pahlawan Perintis Kemerdekaan", "cerita": "Kapitan Pattimura memimpin perlawanan rakyat Maluku terhadap Belanda pada tahun 1817."},
  {"nama": "Laksamana Malahayati", "kategori": "Pahlawan Perintis Kemerdekaan", "cerita": "Laksamana Malahayati adalah laksamana wanita pertama yang memimpin armada laut Aceh melawan Belanda."},
  {"nama": "Cut Nyak Dien", "kategori": "Pahlawan Perintis Kemerdekaan", "cerita": "Cut Nyak Dien adalah pahlawan wanita Aceh yang berjuang melawan penjajahan Belanda."},
  {"nama": "Pangeran Diponegoro", "kategori": "Pahlawan Perintis Kemerdekaan", "cerita": "Pangeran Diponegoro memimpin Perang Jawa melawan Belanda pada 1825–1830."},
  {"nama": "Dr. Wahidin Soedirohusodo", "kategori": "Pahlawan Kebangkitan Nasional", "cerita": "Dr. Wahidin Soedirohusodo adalah pelopor kebangkitan nasional Indonesia."},
  {"nama": "Dr. Soetomo", "kategori": "Pahlawan Kebangkitan Nasional", "cerita": "Dr. Soetomo adalah pendiri Budi Utomo yang berperan dalam kebangkitan nasional."},
  {"nama": "Douwes Dekker", "kategori": "Pahlawan Kebangkitan Nasional", "cerita": "Douwes Dekker adalah tokoh pergerakan nasional yang memperjuangkan kemerdekaan Indonesia."},
  {"nama": "Dr. Cipto Mangunkusumo", "kategori": "Pahlawan Kebangkitan Nasional", "cerita": "Dr. Cipto Mangunkusumo adalah tokoh pergerakan nasional yang aktif melawan penjajahan."},
  {"nama": "Ki Hajar Dewantara", "kategori": "Pahlawan Kebangkitan Nasional", "cerita": "Ki Hajar Dewantara adalah pelopor pendidikan nasional Indonesia."},
  {"nama": "HOS Tjokroaminoto", "kategori": "Pahlawan Kebangkitan Nasional", "cerita": "HOS Tjokroaminoto adalah pemimpin Sarekat Islam yang berperan dalam pergerakan nasional."},
  {"nama": "R.A. Kartini", "kategori": "Pahlawan Nasional", "cerita": "R.A. Kartini adalah pelopor emansipasi wanita Indonesia."},
  {"nama": "Jendral Sudirman", "kategori": "Pahlawan Nasional", "cerita": "Jendral Sudirman adalah panglima besar TNI yang memimpin perang gerilya."},
  {"nama": "B.J. Habibie", "kategori": "Pahlawan Nasional", "cerita": "B.J. Habibie adalah Presiden ke-3 Indonesia dan tokoh penting dalam teknologi nasional."}
]

st.set_page_config(page_title="Pahlawan Indonesia Detection", page_icon="🦸‍♂️", layout="wide")

st.title("🦸‍♂️ Pahlawan Indonesia Detection")
st.write("Upload an image to detect Indonesian national heroes using the trained YOLOv9 model.")

@st.cache_resource
def load_model():
    # Load the model relative to the current directory
    return YOLO("best.pt")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Convert uploaded image to PIL Image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with st.spinner("Detecting..."):
        # Run inference
        results = model(image)
        
    with col2:
        st.subheader("Detection Results")
        for r in results:
            # Plot results
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            st.image(im, use_column_width=True)
            
            # Display detection summary
            boxes = r.boxes
            if len(boxes) > 0:
                st.write(f"**Detected {len(boxes)} object(s):**")
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    conf = float(box.conf[0])
                    
                    st.markdown(f"### {class_name}")
                    st.write(f"**Confidence:** {conf:.2f}")
                    
                    # Look up history
                    hero_info = next((h for h in HERO_HISTORY if h['nama'].lower().strip() == class_name.lower().strip()), None)
                    
                    if hero_info:
                        st.info(f"**Kategori:** {hero_info['kategori']}\n\n**Cerita:** {hero_info['cerita']}")
                    else:
                        st.warning("Informasi pahlawan tidak ditemukan di database.")
            else:
                st.write("No objects detected.")
