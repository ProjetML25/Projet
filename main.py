import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd
import cv2

# Définir le CSS pour personnaliser le style de la barre latérale
page_bg_img = """
<style>
/* Style de la barre latérale */
[data-testid="stSidebar"] {
    background-color: #e5e5f7; /* Couleur de fond */
    background-image: url('https://cdn.wallpapersafari.com/48/25/K50kE3.png'); /* Image de fond */
    background-size: cover; /* L'image remplit toute la zone */
    background-position: center; /* Centrer l'image */
    background-repeat: no-repeat; /* Pas de répétition */
    background-attachment: fixed; /* L'image reste fixe lors du défilement */
}

/* Style du conteneur principal */
[data-testid="stAppViewContainer"] {
    background-color: #e5e5f7; /* Couleur de fond */
    background-image: url('https://wallpapers.com/images/hd/dark-animals-tf7l0vmfmweg2jkv.jpg'); /* Image de fond */
    background-size: 100% 100%; /* L'image s'ajuste pour couvrir toute la zone */
    background-position: center; /* Centrer l'image */
    background-repeat: no-repeat; /* Pas de répétition */
    background-attachment: fixed; /* L'image reste fixe lors du défilement */
}

/* Style de l'en-tête */
[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0); /* En-tête transparent */
}

/* Media queries pour la responsivité */
@media (max-width: 768px) {
    [data-testid="stSidebar"], [data-testid="stAppViewContainer"] {
        background-size: cover; /* L'image s'adapte pour s'ajuster à l'écran */
        background-attachment: scroll; /* Permet un défilement fluide sur mobile */
    }
}
</style>
"""

# Appliquer le CSS à la barre latérale
st.markdown(page_bg_img, unsafe_allow_html=True)
    
# --- Functions ---
def load_model():
    """Load the YOLO model trained on the custom Wild Animals dataset."""
    return YOLO("C:/Users/anasp/runs/detect/train/weights/best.pt")

def preprocess_image(image):
    """Convert PIL image to numpy array for processing."""
    return np.array(image)

def classify_animal(category):
    """Classify whether the detected animal is wild or not based on its category."""
    wild_animals = [
        "lion", "tiger", "elephant", "bear", "wolf", "leopard", "cheetah", 
        "zebra", "giraffe", "hyena", "rhinoceros", "crocodile"
    ]
    return "Sauvage" if category.lower() in wild_animals else "Non Sauvage"

def detect_objects(image, model):
    """Run object detection on the image using YOLOv8."""
    results = model.predict(image, save=False, conf=0.25)
    result_img = results[0].plot()  # Render results on image

    # Extract detection details
    detections = results[0].boxes
    if detections is None:
        return result_img, pd.DataFrame()  # No detections

    data = {
        "name": [results[0].names[int(cls)] for cls in detections.cls],
        "confidence": detections.conf.cpu().numpy(),
        "xmin": detections.xyxy[:, 0].cpu().numpy(),
        "ymin": detections.xyxy[:, 1].cpu().numpy(),
        "xmax": detections.xyxy[:, 2].cpu().numpy(),
        "ymax": detections.xyxy[:, 3].cpu().numpy(),
    }
    detected_objects = pd.DataFrame(data)

    # Add classification (wild or not wild)
    detected_objects["Classification"] = detected_objects["name"].apply(classify_animal)

    return result_img, detected_objects

def display_results(image, detected_objects):
    """Display the detected objects in a table and classify them."""
    st.image(image, caption="Image avec détections", use_container_width=True)

    if detected_objects.empty:
        st.write("Aucun animal détecté.")
    else:
        st.write("Détails des détections :")
        st.dataframe(detected_objects[["name", "confidence", "Classification"]])

def save_results(image, detected_objects, output_dir="output"):
    """Save the detection results to a file and provide download links."""
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder l'image avec les détections
    image_path = os.path.join(output_dir, "detection_result.jpg")
    # Convertir l'image en format approprié pour sauvegarde
    result_image = Image.fromarray(image)  # Convertir NumPy array en image PIL
    result_image.save(image_path)

    # Sauvegarder les détails des détections dans un fichier CSV
    csv_path = os.path.join(output_dir, "detection_details.csv")
    detected_objects.to_csv(csv_path, index=False)

    # Indiquer que les résultats ont été sauvegardés
    st.success("Résultats sauvegardés avec succès.")

    # Ajouter les boutons de téléchargement
    with open(image_path, "rb") as img_file:
        st.download_button(
            label="Télécharger l'image avec détections",
            data=img_file,
            file_name="detection_result.jpg",
            mime="image/jpeg"
        )

    with open(csv_path, "rb") as csv_file:
        st.download_button(
            label="Télécharger les détails des détections (CSV)",
            data=csv_file,
            file_name="detection_details.csv",
            mime="text/csv"
        )

def process_video_fast(video_path, model, skip_frames=2, save_images=False, output_dir="output_frames"):
    """
    Process a video with minimal delay to maintain original speed.
    Optionally save frames with detections.
    """
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()  # Placeholder for video frames
    all_detections = []
    frame_count = 0
    saved_frame = None  # To store the last frame with detections

    # Créer le dossier de sortie pour les images si nécessaire
    if save_images:
        os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue  # Skip frames to speed up processing

        # Convert BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform detection
        result_img, detected_objects = detect_objects(rgb_frame, model)
        
        if not detected_objects.empty:
            all_detections.append(detected_objects)

            # Sauvegarder l'image du frame avec détection, si activé
            if save_images:
                image_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                result_image = Image.fromarray(result_img)  # Convertir NumPy array en image PIL
                result_image.save(image_path)

            # Conserver le dernier frame avec détection pour le téléchargement
            saved_frame = result_img

        # Display frame in Streamlit
        stframe.image(result_img, channels="RGB", use_container_width=True)

    cap.release()
    st.success("Traitement vidéo terminé.")

    if all_detections:
        return pd.concat(all_detections, ignore_index=True), saved_frame
    else:
        return pd.DataFrame(), None

# --- Streamlit App ---

def main():
    st.title("Détection d'Animaux Sauvages avec YOLOv8")

    #st.sidebar.header("Options")
    model = load_model()

    # File Upload
    uploaded_file = st.file_uploader("Téléchargez une image ou une vidéo", type=["jpg", "png", "jpeg", "mp4", "avi", "mov"])

    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension in ["jpg", "png", "jpeg"]:
            # Load and display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Image Importée", use_container_width=True)

            # Preprocess and Detect
            st.write("Prétraitement en cours...")
            processed_image = preprocess_image(image)

            st.write("Détection des animaux...")
            result_img, detected_objects = detect_objects(processed_image, model)

            # Display results
            display_results(result_img, detected_objects)

            # Save results option
            if st.button("Sauvegarder les résultats"):
                save_results(result_img, detected_objects)

        elif file_extension in ["mp4", "avi", "mov"]:
            # Process and display the video
            st.write("Traitement de la vidéo en cours...")
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Activer l'enregistrement des images des frames
            all_detections, saved_frame = process_video_fast("temp_video.mp4", model, skip_frames=2, save_images=True)

            if not all_detections.empty:
                st.write("Détails des détections dans la vidéo :")
                st.dataframe(all_detections[["name", "confidence", "Classification"]])

                # Save results option
                if st.button("Sauvegarder les résultats de la vidéo"):
                    save_results(np.zeros((10, 10, 3), dtype=np.uint8), all_detections, output_dir="output_video")

                # Option to download the last frame with detections
                if saved_frame is not None:
                    with open("frame_with_detection.jpg", "wb") as img_file:
                        Image.fromarray(saved_frame).save(img_file, format="JPEG")

                    with open("frame_with_detection.jpg", "rb") as img_file:
                        st.download_button(
                            label="Télécharger une image avec détections",
                            data=img_file,
                            file_name="frame_with_detection.jpg",
                            mime="image/jpeg"
                        )

if __name__ == "__main__":
    main()
