import cv2
import os
import glob
import numpy as np
import joblib
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

dataset_dir = "dataset"
categories = ["hole", "tear", "stain", "no defect"]

model_path = "svm_fabric_defect.pkl"
scaler_path = "scaler.pkl"

#normalizing image
def normalize_image(image):
    "Normalize image to 0-255 range."
    if image.dtype in [np.float32, np.float64]:
        min_val = image.min()
        max_val = image.max()
        
        if min_val == max_val:
            return np.zeros_like(image, dtype=np.uint8)
        
        image = (image - min_val) / (max_val - min_val)
    
    return (image * 255).astype(np.uint8)

#rotated, cropped, flipped image
def aug_img(image, category, image_number):
    flipped = cv2.flip(image, 1)
    angle = np.random.randint(-15, 15)
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    brightness = np.random.uniform(0.8, 1.2)
    brightened = np.clip(image * brightness, 0, 255).astype(np.uint8)
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)

    # Save augmented images
    save_augmented_image(category, image_number, flipped, "_flipped")
    save_augmented_image(category, image_number, rotated, "_rotated")
    save_augmented_image(category, image_number, brightened, "_brightened")
    save_augmented_image(category, image_number, noisy, "_noisy")

    return [image, flipped, rotated, brightened, noisy]

def save_augmented_image(category, image_number, augmented_image, suffix):
    filename = os.path.join(dataset_dir, category, f"{category}_{image_number}{suffix}.jpg")
    cv2.imwrite(filename, augmented_image)

#LBP
def lbp_transform(image):
    "Apply Local Binary Pattern transformation."
    radius = 0.5
    n_points = 2 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='ror')
    return normalize_image(lbp)

#Gabor Filter
def gabor_transform(image):
    "Apply Gabor filter."
    frequency = 0.69
    gabor_response, _ = gabor(image, frequency=frequency)
    return normalize_image(gabor_response)

for category in categories:
    os.makedirs(os.path.join(dataset_dir, category), exist_ok=True)

#camera
cap = cv2.VideoCapture(3)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Press 'h' for 'hole', 't' for 'tear', 's' for 'stain', 'n' for 'no defect', and 'q' to quit.")

def get_next_image_number(category):
    "Finds the highest numbered image in the category folder."
    category_path = os.path.join(dataset_dir, category)
    files = glob.glob(os.path.join(category_path, f"{category}_*.jpg"))

    numbers = []
    for file in files:
        filename = os.path.basename(file)
        match = filename.split('_')[-1].split('.')[0]
        if match.isdigit():
            numbers.append(int(match))

    return max(numbers, default=0) + 1

#capturing images defect vs no-defect
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Fabric Defect Capture", gray_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('h'):
        label = "hole"
    elif key == ord('t'):
        label = "tear"
    elif key == ord('s'):
        label = "stain"
    elif key == ord('n'):
        label = "no defect"
    elif key == ord('q'):
        break
    else:
        continue

    #Save original image
    image_number = get_next_image_number(label)
    image_path = os.path.join(dataset_dir, label, f"{label}_{image_number}.jpg")
    frame_resized = cv2.resize(frame, (640, 480))
    gray_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(image_path, gray_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    aug_images = aug_img(gray_resized, label, image_number)
    #Apply and save feature extracted images
    lbp_img = lbp_transform(gray_resized)
    gabor_img = gabor_transform(gray_resized)

    cv2.imwrite(os.path.join(dataset_dir, label, f"{label}_{image_number}_lbp.jpg"), lbp_img)
    cv2.imwrite(os.path.join(dataset_dir, label, f"{label}_{image_number}_gabor.jpg"), gabor_img)
    
    print(f"Saved: {image_path}, LBP, Gabor")

cap.release()
cv2.destroyAllWindows()
print("Image capturing complete.")

#svm processing
def extract_features_and_labels():
    X, y = [], []
    image_size = (128, 128)

    for category in categories:
        label = categories.index(category)
        image_paths = glob.glob(os.path.join(dataset_dir, category, "*.jpg"))

        if not image_paths:
            print(f"Warning: No images found in category '{category}'.")

        for img_path in image_paths:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not load image {img_path}")
                continue

            image_resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)

            image_number = get_next_image_number(category)

            #Augment image and save
            augmented_images = aug_img(image_resized, category, get_next_image_number(category))  # pass 'category' and 'image_number'
            
            #Extract features
            for aug in augmented_images:
                aug_resized = cv2.resize(aug, image_size, interpolation=cv2.INTER_AREA)  # Ensure same size
                lbp_features = lbp_transform(aug_resized).flatten()
                gabor_features = gabor_transform(aug_resized).flatten()
                raw_pixels = aug_resized.flatten()

                #Ensure that all feature vectors are the same number of features
                features = np.hstack([raw_pixels, lbp_features, gabor_features])

                #Append to X and y
                X.append(features)
                y.append(label)

    if not X:
        print("Error: No valid image data found!")
        return np.array([]), np.array([])

    X, y = np.array(X), np.array(y)
    print(f"Extracted {X.shape[0]} samples.")

    return shuffle(X, y, random_state=42)

svm_model = SGDClassifier(loss="hinge", alpha=0.001, tol=1e-4, penalty="l2", max_iter=1000)

#Extract features
X, y = extract_features_and_labels()

if len(X) == 0:
    print("No images found for training.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 
classes = np.array([0, 1, 2, 3])

svm_model.partial_fit(X_train, y_train, classes=classes)

joblib.dump(svm_model, model_path)
joblib.dump(scaler, scaler_path)
y_pred = svm_model.predict(X_test)

accuracy = np.mean(y_pred == y_test)

if accuracy > 0.70:
    joblib.dump(svm_model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Training completed with updated data.")

y_pred_test = svm_model.predict(X_test)

print(f"Training feature count: {X_train.shape[1]}")
print(f"Model expects: {svm_model.n_features_in_ if hasattr(svm_model, 'n_features_in_') else 'Unknown'}")