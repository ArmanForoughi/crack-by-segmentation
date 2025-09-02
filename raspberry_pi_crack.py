import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import smtplib
import time

import tensorflow
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

start_time = 12

import cv2

x_test_dir = os.path.join("", "pic_camera")

# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
    
def send_mail_function(msg):
    recipientEmail = "armanforoughi36@gmail.com"
    recipientEmail = recipientEmail.lower()
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login("armanforoughi65@gmail.com", '************')
        server.sendmail('armanforoughi65@gmail.com', recipientEmail, msg)
        print("sent to {}".format(recipientEmail))
        server.close()
    except Exception as e:
    	print(e)

def get_validation_augmentation(target_height=320, target_width=320):
    test_transform = [
        A.PadIfNeeded(min_height=target_height, min_width=target_width, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Resize(height=target_height, width=target_width),
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform"""
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

class Dataset:
    """General dataset with optional segmentation mask support."""
    
    CLASSES = ['crack']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir=None, 
            classes=['crack'], 
            augmentation=None, 
            preprocessing=None,
            img_size=320,
    ):
        self.img_size = img_size
        
        self.list_img = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, img_id) for img_id in self.list_img]
        
        if masks_dir and os.path.exists(masks_dir):
            self.has_masks = True
            self.list_mask = sorted(os.listdir(masks_dir))
            self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.list_mask]
        else:
            self.has_masks = False
            self.masks_fps = None
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes] if classes else []
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # Read image
        print(self.images_fps[i])
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Read mask if exists
        if self.has_masks:
            mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            
            mask = mask.astype('float32')  # یا np.float32
            mask = np.expand_dims(mask, axis=-1)
            
        else:
            mask = np.zeros((self.img_size, self.img_size, 1), dtype=np.float32)  # Dummy mask
        
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask
        
    def __len__(self):
        return len(self.list_img)

import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
from segmentation_models import PSPNet
from segmentation_models import Unet
from segmentation_models import Linknet
from segmentation_models import FPN
from segmentation_models.losses import DiceLoss
from segmentation_models.losses import BinaryFocalLoss
from tensorflow.keras.optimizers import Adam

BACKBONE = 'efficientnetb3'

preprocess_input = sm.get_preprocessing(BACKBONE)

n_classes = 1
activation = 'sigmoid'

model_FPN = FPN(
    backbone_name=BACKBONE,
    classes=n_classes,
    activation=activation
)

dataset = Dataset(x_test_dir, 
                  # y_test_dir, 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),)

image, mask = dataset[0] # get some sample

print(image.shape)

model_FPN.load_weights('best_model.h5') 
while True:
    if time.localtime().tm_hour == start_time:

        # n = 5
        n = len(dataset)
        ids = np.arange(n)
        
        for i in ids:
            image, annotation_mask = dataset[i]
            image = np.expand_dims(image, axis=0)
            pr_FPN = model_FPN.predict(image).round()
            print(cv2.countNonZero(pr_FPN))
            if cv2.countNonZero(pr_FPN) > 650:
                print("Crack detected on the wall")
                # send_mail_function("Crack detected on the wall")
            else:
                print("Cracks on the wall were not detected.")
    time.sleep(60*60)