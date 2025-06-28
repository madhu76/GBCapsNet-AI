import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import PReLU
import tensorflow as tf
import matplotlib.cm as cm
from keras import backend as K

# Set the page layout to wide
st.set_page_config(layout="wide")

@st.cache_resource
def load_capsnet_model():
    model_path = "C:/Users/jogee/Desktop/DL-Implementation/Gallbladder/Streamlit-App/GBCapsNetR2.h5"
    model = load_model(model_path, custom_objects={'PReLU': PReLU}, compile=False)
    return model

model = load_capsnet_model()

def preprocess_image(image, target_size=(128, 128)):
    img = image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap, alpha=0.4):
    # Resize heatmap to match image size
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = Image.fromarray((jet_heatmap * 255).astype(np.uint8)).resize(original_img.size)
    jet_heatmap = np.array(jet_heatmap)
    superimposed_img = np.array(original_img) * (1 - alpha) + jet_heatmap * alpha
    superimposed_img = np.uint8(superimposed_img)
    return Image.fromarray(superimposed_img)

st.sidebar.title("Gallbladder Disease classification")
selection = st.sidebar.selectbox("Go to", ["Architecture", "Prediction"])

if selection == "Architecture":
    st.title("Architecture Section")
    st.write(
        "The Gallbladder Image Classification Model is based on a Capsule Network (CapsNet) architecture, which is designed to handle image data more efficiently and robustly compared to traditional convolutional neural networks (CNNs)."
    )

    # Main architecture image
    image_path = "Graphical Abstract GBCapsNet.png"
    architecture_image = Image.open(image_path)
    st.image(architecture_image, use_column_width=True)

    # Class images, captions, and content (no brackets in descriptions)
    class_images = [
        "1.jpg",
        "2.jpg",
        "3.jpg",
        "4.jpg",
        "5.jpg",
        "6.jpg",
        "7.jpg",
        "8.jpg",
        "9.jpg"
    ]
    class_captions = [
        "Gallstones",
        "Abdomen and Retroperitoneum",
        "Cholecystitis",
        "Membranous and Gangrenous Cholecystitis",
        "Perforation",
        "Polyps and Cholesterol Crystals",
        "Adenomyomatosis",
        "Carcinoma",
        "Various Causes of Gallbladder Wall Thickening"
    ]
    class_contents = [
        "Gallstones or cholelithiasis are solid deposits mainly of cholesterol or bilirubin that form in the gallbladder. They are the most common gallbladder disease and can cause pain, infection, or blockage of bile ducts.",
        "The abdomen and retroperitoneum include the anatomical regions containing the gallbladder and related organs. Diseases in these areas may affect the gallbladder and its function.",
        "Cholecystitis is inflammation of the gallbladder, most often due to gallstones blocking the cystic duct. It can be acute or chronic and causes pain, fever, and digestive symptoms.",
        "Membranous and gangrenous cholecystitis are severe forms of gallbladder inflammation. Gangrenous cholecystitis involves tissue death due to loss of blood supply and increases risk for perforation and sepsis.",
        "Perforation of the gallbladder is a life-threatening complication, often resulting from untreated or severe cholecystitis. It can lead to bile leakage and peritonitis.",
        "Gallbladder polyps are growths projecting from the lining, while cholesterol crystals may deposit in the wall or bile and sometimes form stones. Most polyps are benign but require monitoring.",
        "Adenomyomatosis is a benign condition characterized by overgrowth of the gallbladder wall and formation of small cystic spaces. It may be asymptomatic or cause biliary symptoms.",
        "Gallbladder carcinoma is a malignant tumor often diagnosed late due to non-specific symptoms. Risk factors include gallstones and chronic inflammation.",
        "Gallbladder wall thickening can result from various causes including inflammation, infection, heart failure, or malignancy. It is a non-specific sign seen on imaging."
    ]

    st.subheader("Gallbladder Disease Classes")

    for row in range(3):
        cols = st.columns(3)
        for col in range(3):
            idx = row * 3 + col
            with cols[col]:
                st.image(class_images[idx], caption=class_captions[idx], use_column_width=True)
                st.write(class_contents[idx])


elif selection == "Prediction":
    st.title("Gallbladder Image Classification")
    st.write("Upload an image to get the model prediction.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    CATEGORIES = [
        "Gallstones", "Abdomen and retroperitoneum", "cholecystitis", 
        "Membranous and gangrenous cholecystitis",
        "Perforation", "Polyps and cholesterol crystals", 
        "Adenomyomatosis", "Carcinoma",
        "Various causes of gallbladder wall thickening"
    ]

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Predicting..."):
            input_image = preprocess_image(image)
            logits = model.predict(input_image)
            predicted_class_index_before = np.argmax(logits)
            predicted_class_before = CATEGORIES[predicted_class_index_before]
            confidence_before = logits[0][predicted_class_index_before]
            temperature = 0.3473
            scaled_logits = logits / temperature
            prediction = tf.nn.softmax(scaled_logits)
            predicted_class_index_after = np.argmax(prediction)
            predicted_class_after = CATEGORIES[predicted_class_index_after]
            confidence_after = prediction[0][predicted_class_index_after]

            # Grad-CAM
            last_conv_layer_name = "cnn4"  # Replace with actual layer name
            try:
                heatmap = make_gradcam_heatmap(input_image, model, last_conv_layer_name, pred_index=predicted_class_index_before)
                gradcam_img = overlay_gradcam(image, heatmap)
            except Exception as e:
                gradcam_img = Image.new("RGB", image.size, color=(255,255,255))
                st.warning(f"Grad-CAM visualization failed: {e}. Please check your last conv layer name.")

        # Display images side by side, filling their columns
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.image(gradcam_img, caption="Grad-CAM Heatmap", use_column_width=True)

        # Display prediction class and confidence on new lines
        st.markdown(f"**Prediction Class (Before Temperature Scaling):**")
        st.markdown(f"<span style='color:blue'><i>{predicted_class_before}</i></span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence (Before Temperature Scaling):** {confidence_before * 100:.2f}%")
        st.markdown(f"**Prediction Class (After Temperature Scaling):**")
        st.markdown(f"<span style='color:blue'><i>{predicted_class_after}</i></span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence (After Temperature Scaling):** {confidence_after * 100:.2f}%")
