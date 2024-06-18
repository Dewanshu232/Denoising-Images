

import os
import random
import numpy as np
from glob import glob
import scipy.io
import shutil
import zipfile
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from shutil import copy2
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
from pathlib import Path
from shutil import copy2

Img_size=256
Batch=16
MAX=485
Train_llimages = sorted(glob("./Train/low/*"))[:MAX]
Val_llimages = sorted(glob("./test/low/*"))


print(len(Train_llimages))
print(len(Val_llimages))





def loaddata(image_path):
    image=tf.io.read_file(image_path)
    image=tf.image.decode_png(image,channels=3)
    image=tf.image.resize(images=image,size=[Img_size,Img_size])
    image=image/255.0

    return image


def data_generator(llimages):
    dataset=tf.data.Dataset.from_tensor_slices((llimages))
    dataset =dataset.map(loaddata,num_parallel_calls=tf.data.AUTOTUNE)
    dataset=dataset.batch(Batch,drop_remainder=True)
    return dataset

train_dataset=data_generator(Train_llimages)
val_dataset=data_generator(Val_llimages)

def build_dce_net():
    input_img = keras.Input(shape=[None, None, 3])
    conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding="same")(input_img)
    conv2 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding="same")(conv1)
    conv3 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding="same")(conv2)
    conv4 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding="same")(conv3)
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding="same")(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding="same")(int_con2)
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    output = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(int_con3)
    return keras.Model(inputs=input_img, outputs=output)

def color_constancy_loss (x):
  mean_rgb = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
  mr, mg, mb =mean_rgb[:,:, :, 0], mean_rgb[:, :,:, 1], mean_rgb[:, :, :, 2]
  d_rg =tf.square(mr - mg)
  d_rb = tf.square(mr - mb)
  d_gb = tf.square(mb - mg)
  return tf.sqrt(tf.square (d_rg) + tf.square (d_rb) + tf.square (d_gb))

def exposure_loss (x, mean_val=0.6):
  x = tf.reduce_mean(x, axis=3, keepdims=True)
  mean = tf.nn.avg_pool2d(x, ksize=16, strides=16, padding="VALID")
  return tf.reduce_mean (tf.square (mean -mean_val))

def illumination_smoothness_loss(x):
    batch_size = tf.shape(x)[0]
    h_x = tf.shape(x)[1]
    w_x = tf.shape(x)[2]
    count_h = (h_x - 1) * w_x
    count_w = h_x * (w_x - 1)
    h_tv = tf.reduce_sum(tf.square(x[:, 1:, :, :] - x[:, :h_x - 1, :, :]))
    w_tv = tf.reduce_sum(tf.square(x[:, :, 1:, :] - x[:, :, :w_x - 1, :]))
    batch_size = tf.cast(batch_size, dtype=tf.float32)
    count_h = tf.cast(count_h, dtype=tf.float32)
    count_w = tf.cast(count_w, dtype=tf.float32)
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

class SpatialConsistencyLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(reduction="none")

        self.left_kernel = tf.constant(
            [[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.right_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.up_kernel = tf.constant(
            [[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.down_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=tf.float32
        )

    def call(self, y_true, y_pred):
        original_mean = tf.reduce_mean(y_true, 3, keepdims=True)
        enhanced_mean = tf.reduce_mean(y_pred, 3, keepdims=True)
        original_pool = tf.nn.avg_pool2d(
            original_mean, ksize=4, strides=4, padding="VALID"
        )
        enhanced_pool = tf.nn.avg_pool2d(
            enhanced_mean, ksize=4, strides=4, padding="VALID"
        )

        d_original_left = tf.nn.conv2d(
            original_pool,
            self.left_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_original_right = tf.nn.conv2d(
            original_pool,
            self.right_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_original_up = tf.nn.conv2d(
            original_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_original_down = tf.nn.conv2d(
            original_pool,
            self.down_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        d_enhanced_left = tf.nn.conv2d(
            enhanced_pool,
            self.left_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_enhanced_right = tf.nn.conv2d(
            enhanced_pool,
            self.right_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_enhanced_up = tf.nn.conv2d(
            enhanced_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_enhanced_down = tf.nn.conv2d(
            enhanced_pool,
            self.down_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        d_left = tf.square(d_original_left - d_enhanced_left)
        d_right = tf.square(d_original_right - d_enhanced_right)
        d_up = tf.square(d_original_up - d_enhanced_up)
        d_down = tf.square(d_original_down - d_enhanced_down)
        return d_left + d_right + d_up + d_down

class ZeroDCE(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dce_model = build_dce_net()


    def compile(self, learning_rate, **kwargs):
        super().compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.spatial_constancy_loss = SpatialConsistencyLoss(reduction="none")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.illumination_smoothness_loss_tracker = keras.metrics.Mean(
            name="illumination_smoothness_loss"
        )
        self.spatial_constancy_loss_tracker = keras.metrics.Mean(
            name="spatial_constancy_loss"
        )
        self.color_constancy_loss_tracker = keras.metrics.Mean(
            name="color_constancy_loss"
        )
        self.exposure_loss_tracker = keras.metrics.Mean(name="exposure_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.illumination_smoothness_loss_tracker,
            self.spatial_constancy_loss_tracker,
            self.color_constancy_loss_tracker,
            self.exposure_loss_tracker,
        ]

    def get_enhanced_image(self, data, output):
        r1 = output[:, :, :, :3]
        r2 = output[:, :, :, 3:6]
        r3 = output[:, :, :, 6:9]
        r4 = output[:, :, :, 9:12]
        r5 = output[:, :, :, 12:15]
        r6 = output[:, :, :, 15:18]
        r7 = output[:, :, :, 18:21]
        r8 = output[:, :, :, 21:24]
        x = data + r1 * (tf.square(data) - data)
        x = x + r2 * (tf.square(x) - x)
        x = x + r3 * (tf.square(x) - x)
        enhanced_image = x + r4 * (tf.square(x) - x)
        x = enhanced_image + r5 * (tf.square(enhanced_image) - enhanced_image)
        x = x + r6 * (tf.square(x) - x)
        x = x + r7 * (tf.square(x) - x)
        enhanced_image = x + r8 * (tf.square(x) - x)
        return enhanced_image

    def call(self, data):
        dce_net_output = self.dce_model(data)
        return self.get_enhanced_image(data, dce_net_output)

    def compute_losses(self, data, output):
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(
            self.spatial_constancy_loss(enhanced_image, data)
        )
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy_loss(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_loss(enhanced_image))
        total_loss = (
            loss_illumination
            + loss_spatial_constancy
            + loss_color_constancy
            + loss_exposure
        )

        return {
            "total_loss": total_loss,
            "illumination_smoothness_loss": loss_illumination,
            "spatial_constancy_loss": loss_spatial_constancy,
            "color_constancy_loss": loss_color_constancy,
            "exposure_loss": loss_exposure,
        }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            losses = self.compute_losses(data, output)

        gradients = tape.gradient(
            losses["total_loss"], self.dce_model.trainable_weights
        )
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))

        self.total_loss_tracker.update_state(losses["total_loss"])
        self.illumination_smoothness_loss_tracker.update_state(
            losses["illumination_smoothness_loss"]
        )
        self.spatial_constancy_loss_tracker.update_state(
            losses["spatial_constancy_loss"]
        )
        self.color_constancy_loss_tracker.update_state(losses["color_constancy_loss"])
        self.exposure_loss_tracker.update_state(losses["exposure_loss"])

        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        output = self.dce_model(data)
        losses = self.compute_losses(data, output)

        self.total_loss_tracker.update_state(losses["total_loss"])
        self.illumination_smoothness_loss_tracker.update_state(
            losses["illumination_smoothness_loss"]
        )
        self.spatial_constancy_loss_tracker.update_state(
            losses["spatial_constancy_loss"]
        )
        self.color_constancy_loss_tracker.update_state(losses["color_constancy_loss"])
        self.exposure_loss_tracker.update_state(losses["exposure_loss"])

        return {metric.name: metric.result() for metric in self.metrics}

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """While saving the weights, we simply save the weights of the DCE-Net"""
        self.dce_model.save_weights(
            filepath,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        """While loading the weights, we simply load the weights of the DCE-Net"""
        self.dce_model.load_weights(
            filepath=filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )



zero_dce_model = ZeroDCE()
zero_dce_model.compile(learning_rate=1e-4)
history = zero_dce_model.fit(train_dataset, validation_data=val_dataset, epochs=100)


def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_result("total_loss")
plot_result("illumination_smoothness_loss")
plot_result("spatial_constancy_loss")
plot_result("color_constancy_loss")
plot_result("exposure_loss")

def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        plt.imshow(images[i])
    plt.axis("off")
    plt.show()

# Function to enhance image using the model
def infer(original_image):

    image = keras.preprocessing.image.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = zero_dce_model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    return output_image

def process_and_convert_images(input_files, output_dir):
    for val_image_file in input_files:
        original_image = Image.open(val_image_file)
        enhanced_image = infer(original_image)

      
        plot_results(
            [original_image, ImageOps.autocontrast(original_image), enhanced_image],
            ["Original", "PIL Autocontrast", "Enhanced"],
            (20, 12),
        )

        # Define output path
        output_path = os.path.join(output_dir)
        os.makedirs(output_path, exist_ok=True)

    
        enhanced_image.save(os.path.join(output_path, os.path.basename(val_image_file)))

# Define the directories for processing
output_dirtest = './test/predicted'


process_and_convert_images(Val_llimages, output_dirtest)

# Define the directories for processing
output_dirtrain = '/Train/predicted'


process_and_convert_images(Train_llimages, output_dirtrain)



def calculate_metrics(dataset, model):
    mse = keras.metrics.MeanSquaredError()
    psnr = keras.metrics.MeanSquaredError() # PSNR is a function of MSE
    mae = keras.metrics.MeanAbsoluteError()

    for data in dataset:
        original_image = data[0]
        enhanced_image = model(data[0])
        mse.update_state(original_image, enhanced_image)
        mae.update_state(original_image, enhanced_image)
        psnr.update_state(tf.image.psnr(original_image, enhanced_image, max_val=1.0))

    print(f"MSE: {mse.result().numpy()}")
    print(f"PSNR: {tf.reduce_mean(psnr.result()).numpy()}") # Convert mean PSNR
    print(f"MAE: {mae.result().numpy()}")

# Report metrics on the training dataset
calculate_metrics(train_dataset, zero_dce_model)





def compute_psnr(img1_path, img2_path):
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        print(f"Error reading images: {img1_path}, {img2_path}")
        return None

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # If MSE is zero, the images are identical
    max_pixel_value = 255.0  # Assuming 8-bit depth images
    psnr_value = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    return psnr_value

def process_and_compare_images(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    psnr_values = []

    # Iterate over all PNG files in the input directory
    for input_image_path in input_dir.glob('*.png'):
        output_image_path = output_dir / input_image_path.name

        if output_image_path.exists():
            psnr_value = compute_psnr(input_image_path, output_image_path)
            if psnr_value is not None:
                psnr_values.append(psnr_value)
                print(f"PSNR between {input_image_path} and {output_image_path}: {psnr_value:.2f} dB")
        else:
            print(f"Output image not found for {input_image_path}")

    if psnr_values:
        avg_psnr = sum(psnr_values) / len(psnr_values)
        print(f"Average PSNR: {avg_psnr:.2f} dB")
    else:
        print("No PSNR values calculated.")

# Define the directories for processing
input_dir = './test/low'  # Replace with your input images directory
output_dir = './test/predicted'  # Replace with your output images directory

# Process and compare the images
process_and_compare_images(input_dir, output_dir)
input_dir = './Train/low'  # Replace with your input images directory
output_dir = './Train/predicted'  # Replace with your output images directory

# Process and compare the images
process_and_compare_images(input_dir, output_dir)


