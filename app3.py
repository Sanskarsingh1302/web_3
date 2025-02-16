import streamlit as st
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Load the pre-trained Faster R-CNN model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    return model, device

# Preprocess images for alignment and resizing
def preprocess_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.resize(gray1, (640, 480))
    gray2 = cv2.resize(gray2, (640, 480))
    return gray1, gray2

# Resize images dynamically for performance
def resize_image(image, max_dim=800):
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

# Improved change detection using SSIM
def change_detection_ssim(img1, img2, threshold=30):
    score, diff = ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh, diff

# Perform object detection on the image
def detect_objects(image, model, device):
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    image_tensor = transform(image).to(device)
    with torch.no_grad():
        predictions = model([image_tensor])
    return predictions

# Calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Streamlit application
def main():
    st.title("Change Detection")
    st.write("Upload two images to detect changes and new objects between them.")

    model, device = load_model()

    # Upload images
    uploaded_file1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
    uploaded_file2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        # Read the uploaded images
        image1 = Image.open(uploaded_file1).convert("RGB")
        image2 = Image.open(uploaded_file2).convert("RGB")
        image1_cv = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        image2_cv = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

        # Resize images for processing
        image1_cv = resize_image(image1_cv)
        image2_cv = resize_image(image2_cv)

        # Preprocess the images
        gray1, gray2 = preprocess_images(image1_cv, image2_cv)

        # SSIM threshold slider
        threshold = st.slider("SSIM Threshold", min_value=0, max_value=255, value=30, step=5)

        # Detect changes using SSIM
        change_mask, diff_map = change_detection_ssim(gray1, gray2, threshold)

        # Detect objects in both images
        predictions1 = detect_objects(image1_cv, model, device)
        predictions2 = detect_objects(image2_cv, model, device)

        # Draw bounding boxes on detected objects and identify new objects
        new_objects = []
        all_objects = []
        for bbox1 in predictions1[0]['boxes']:
            x1, y1, x2, y2 = bbox1.int().cpu().numpy()
            cv2.rectangle(image1_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green boxes for image 1

        for bbox2 in predictions2[0]['boxes']:
            x1, y1, x2, y2 = bbox2.int().cpu().numpy()
            new = True
            for bbox1 in predictions1[0]['boxes']:
                if calculate_iou(bbox1.int().cpu().numpy(), bbox2.int().cpu().numpy()) > 0.4:
                    new = False
                    break
            if new:
                new_objects.append([x1, y1, x2, y2])
                cv2.rectangle(image2_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red boxes for new objects
            all_objects.append([x1, y1, x2, y2])

        # Display results
        st.subheader("Change Mask (Thresholded)")
        st.image(change_mask, channels="GRAY")

        st.subheader("SSIM Difference Map")
        st.image(diff_map, channels="GRAY")

        st.subheader("Detected Objects in First Image")
        st.image(cv2.cvtColor(image1_cv, cv2.COLOR_BGR2RGB))

        st.subheader("Detected Objects in Second Image")
        st.image(cv2.cvtColor(image2_cv, cv2.COLOR_BGR2RGB))

        # Composite image with new objects
        composite_image = image2_cv.copy()
        for bbox in new_objects:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(composite_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(composite_image, "New Object", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        st.subheader("Composite Image with New Objects")
        st.image(cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB))

        # List bounding boxes of new objects
        st.subheader("Bounding Boxes of New Objects")
        for i, bbox in enumerate(new_objects):
            st.write(f"New Object {i + 1}: {bbox}")

        # Display all detected objects
        st.subheader("All Detected Objects in Second Image")
        for i, bbox in enumerate(all_objects):
            x1, y1, x2, y2 = bbox
            cropped_image = image2_cv[y1:y2, x1:x2]
            st.image(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB), caption=f"Detected Object {i + 1}")

if __name__ == "__main__":
    main()
