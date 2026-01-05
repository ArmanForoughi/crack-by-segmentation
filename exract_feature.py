import os
pth = "C:/Users/amir/Downloads/Compressed/crack seg1.v1i.png-mask-semantic/train/"
lst = os.listdir(pth)
import cv2
import numpy as np
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize

def thickness(degrees, mask):
    mask1 = mask > 0
    skel = skeletonize(mask1).astype(np.uint8)
    
    dx = np.sin(np.deg2rad(degrees))
    dy = np.cos(np.deg2rad(degrees))

    ys, xs = np.where(skel > 0)

    thicknesses = []

    for x0, y0 in zip(xs, ys):
        profile = []
        for t in range(-50, 50):
            x = int(x0 + dx * t)
            y = int(y0 + dy * t)
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                profile.append(mask[y, x])
            else:
                profile.append(0)

        profile = np.array(profile)
        white = np.where(profile > 0)[0]
        if len(white) > 1:
            thickness = white[-1] - white[0]
            thicknesses.append(thickness)

    mean_thickness = np.mean(thicknesses)

    max_thickness = np.max(thicknesses)

    return {"mean":mean_thickness , "max":max_thickness, 
            "mean_n":mean_thickness/40 , "max_n":max_thickness/40}

def length(mask):
    mask1 = mask > 0
    skel = skeletonize(mask1).astype(np.uint8)
    length_pixels = np.sum(skel)
    return {"len":length_pixels, "len_n":length_pixels/452}

def angle(mask):
    # مختصات پیکسل‌های ترک
    ys, xs = np.where(mask > 0)
    points = np.column_stack((xs, ys))

    pca = PCA(n_components=2)
    pca.fit(points)

    vx, vy = pca.components_[0]
    angle = np.degrees(np.arctan2(vy, vx))
    angle = (- angle) if angle < 0 else 180 - angle
    def encode_angle(theta_deg):
        theta_rad = np.deg2rad(theta_deg)
        return np.cos(theta_rad), np.sin(theta_rad)
    cos, sin = encode_angle(angle)
    return {"angle":angle , "cos":cos, "sin":sin}

def density(mask):
    return {"density":cv2.countNonZero(mask), "density_n":(cv2.countNonZero(mask) / (320*320)) / 0.12}

# import os
# os.environ['SM_FRAMEWORK'] = 'tf.keras'
# import segmentation_models as sm

# BACKBONE = 'efficientnetb3'
# BATCH_SIZE = 4
# CLASSES = ['crack']
# LR = 0.0001
# EPOCHS = 40

# preprocess_input = sm.get_preprocessing(BACKBONE)
# # define network parameters
# n_classes = 1
# activation = 'sigmoid'

# #create model
# model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
# model.load_weights('best_model_fpn_efficientnetb3.h5')


all_features = []
all_labels = []
for i in lst:
    if i.endswith(".png"):
        mask = cv2.imread(pth + i, cv2.IMREAD_GRAYSCALE)

        # image = cv2.imread(pth + i, cv2.IMREAD_COLOR)
        # image_input = np.expand_dims(image, axis=0)
        # mask = model.predict(image_input).round()

        patch_w, patch_h = 320, 320
        h, w = mask.shape
        features = []
        for y in range(0, h, patch_h):
            for x in range(0, w, patch_w):
                patch = mask[y:y+patch_h, x:x+patch_w]
                if patch.shape[0] == patch_h and patch.shape[1] == patch_w:

                    msk = patch
                    dens = density(msk)
                    if dens["density_n"] > 0.01:
                        ang = angle(msk)
                        thick = thickness(ang["angle"], msk)
                    else:
                        ang = {"angle":0 , "cos":0.0, "sin":0.0}
                        thick = {"mean":0.0 , "max":0, "mean_n":0.0 , "max_n":0.0}
                    lengths = length(msk)

                    features.append([dens["density"], ang["cos"], ang["sin"],
                                    lengths["len"], thick["mean"], thick["max"]])
        all_features.append(np.array(features))

    if i.endswith(".txt"):
        all_labels.append(np.loadtxt(pth + i))
    print(f"Processed {i}")

np.savez("dataset.npz", 
         X=np.array(all_features),
         y=np.array(all_labels))
