image = "crack_13.jpg"
image_mack = "crack_14.png"

from keras.models import load_model

model = load_model("cnn_model.keras")

# y_pred = model.predict(X_test)
import os
import cv2
from skimage.morphology import skeletonize
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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


img = cv2.imread(image, cv2.IMREAD_COLOR)
mask = cv2.imread(image_mack, cv2.IMREAD_GRAYSCALE)

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
feature = np.array(features)
feature.shape

x = feature.reshape(1, 3, 5, 6).transpose(0, 3, 1, 2)

data = np.load("dataset.npz")
X = data["X"]
y = data["y"]
X = X.reshape(X.shape[0], 3, 5, 6).transpose(0, 3, 1, 2)  

mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-8

x = (x - mean) / std

y_pred = model.predict(x)

def draw_down_arrow(img, center, size, color, thickness=-1):
    """
    رسم فلش رو به پایین
    """
    cx, cy = center
    w = int(size * 0.6)
    h = int(size * 0.6)

    pts = np.array([
        [cx - w, cy - h//2],
        [cx + w, cy - h//2],
        [cx + w, cy],
        [cx + 2*w, cy],
        [cx, cy + h],
        [cx - 2*w, cy],
        [cx - w, cy],
    ], np.int32)

    cv2.fillPoly(img, [pts], color)


def visualize_settlement_opencv(
    image_path,
    settlements,           # لیست 5تایی بین 0 و 1
    output_path,
    footer_height=180,
    low_th=0.02,
    high_th=0.50
):
    if len(settlements) != 5:
        raise ValueError("settlements باید شامل 5 مقدار باشد")

    # خواندن تصویر
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("تصویر ورودی خوانده نشد")

    h, w, _ = img.shape

    # ساخت تصویر جدید با footer
    canvas = np.ones((h + footer_height, w, 3), dtype=np.uint8) * 255
    canvas[0:h, 0:w] = img

    # خط جداکننده
    cv2.line(canvas, (0, h), (w, h), (150, 150, 150), 2)

    segment_w = w // 5
    cy = h + footer_height // 2
    shape_size = int(min(segment_w * 0.6, 130))

    for i, val in enumerate(settlements):
        cx = int(segment_w * (i + 0.5))
        percent_text = f"{int(val * 100)}%"

        # رنگ‌ها (BGR)
        GREEN = (60, 180, 75)
        YELLOW = (0, 215, 255)
        RED = (0, 0, 255)
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)

        if val <= low_th:
            # مربع سبز
            half = shape_size // 2
            cv2.rectangle(
                canvas,
                (cx - half*2, cy - half),
                (cx + half*2, cy + half),
                GREEN,
                -1
            )
            cv2.rectangle(
                canvas,
                (cx - half*2, cy - half),
                (cx + half*2, cy + half),
                BLACK,
                2
            )
            text_color = BLACK
        
        elif val >= high_th:
            # فلش قرمز
            draw_down_arrow(canvas, (cx, cy), shape_size, RED)
            text_color = WHITE

        else:
            # مربع زرد
            half = shape_size // 2
            cv2.rectangle(
                canvas,
                (cx - half*2, cy - half),
                (cx + half*2, cy + half),
                YELLOW,
                -1
            )
            cv2.rectangle(
                canvas,
                (cx - half*2, cy - half),
                (cx + half*2, cy + half),
                BLACK,
                2
            )
            text_color = BLACK
            
        # نوشتن درصد داخل شکل
        (tw, th), _ = cv2.getTextSize(
            percent_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            2
        )
        cv2.putText(
            canvas,
            percent_text,
            (cx - tw // 2, cy + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            text_color,
            2,
            cv2.LINE_AA
        )

        

    # ذخیره خروجی
    cv2.imwrite(output_path, canvas)
    
    print("Saved:", output_path)
    os.system(f'xdg-open "{output_path}"')

visualize_settlement_opencv(
    image_path=image,
    settlements=y_pred[0],
    output_path="wall_settlement_result-05.png"
)
