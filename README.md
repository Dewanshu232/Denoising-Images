![image](https://github.com/Dewanshu232/Denoising-Images/assets/122469929/6ae3e02c-fef6-43a0-88aa-53a95d36c7af)



# Zero-DCE: Low-Light Image Enhancement

## Introduction

### Architecture Used
In this project, we implemented the Zero-DCE (Zero-Reference Deep Curve Estimation) model to enhance low-light images. Zero-DCE is a deep learning model designed to enhance low-light images by estimating pixel-wise light enhancement curves. The model operates without the need for paired low/normal light images during training, making it a highly efficient and versatile solution for low-light image enhancement.

### Specifications
- **Image Size:** 256 x 256
- **Batch Size:** 16
- **Max Training Images:** 400
- **Learning Rate:** 1e-4
- **Epochs:** 100
- 
### Performance Metrics for Traning data

#### Peak Signal to Noise Ratio (PSNR)
The term Peak Signal-to-Noise Ratio (PSNR) is an expression for the ratio between the maximum possible value (power) of a signal and the power of distorting noise that affects the quality of its representation. Because many signals have a very wide dynamic range (ratio between the largest and smallest possible values of a changeable quantity), the PSNR is usually expressed in terms of the logarithmic decibel scale. The higher the PSNR, the better the quality of the compressed image.

![PSNR Equation](https://github.com/Dewanshu232/Denoising-Images/assets/122469929/0fb99d9f-7b3d-42f5-aac7-4e991fcd56f6)
### PSNR Value Achieved
The Peak Signal-to-Noise Ratio (PSNR) value achieved on the training dataset is computed during evaluation and reported below.
28.7
![PSNR Value](https://github.com/Dewanshu232/Denoising-Images/assets/122469929/7bffcf84-d7ce-4af5-acf3-4ef6031a6be2)


#### Mean Squared Error (MSE)
The Mean Squared Error (MSE) is a measure of the average squared difference between the estimated values and the actual value. It provides an idea of how well the model is performing by penalizing larger errors more than smaller ones. Lower values of MSE indicate better performance.

The formula for MSE is:
\[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]

#### Mean Absolute Error (MAE)
The Mean Absolute Error (MAE) measures the average magnitude of the errors in a set of predictions, without considering their direction. It’s the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.

The formula for MAE is:
\[ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \]
![image](https://github.com/Dewanshu232/Denoising-Images/assets/122469929/4ef4f77a-c356-471e-9081-6b759ef76fc2)




### Paper Implemented
The project is based on the paper: "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement" by Chien-Hsiang Huang, Chia-Yu Chen, Chu-Song Chen, and Chen-Chi Chang. The paper can be accessed [here](https://arxiv.org/abs/2001.06826).
![image](https://github.com/Dewanshu232/Denoising-Images/assets/122469929/f3e62080-9811-4422-b11c-e3be88547b86)



## Project Details
## Zero DCE (Zero-Reference Deep Curve Estimation) –
This method trains a lightweight deep network, DCE-Net, to estimate pixel-wise and high-order curves for dynamic range adjustment of a given image. The curve estimation is specially designed, considering pixel value range, monotonicity, and differentiability. It does not require any paired or unpaired data during training. This is achieved through a set of carefully formulated non-reference loss functions, which implicitly measure the enhancement quality and drive the learning of the network. This method is efficient as image enhancement can be achieved by an intuitive and simple nonlinear curve mapping. A Deep Curve Estimation Network (DCE-Net) is devised to estimate a set of best-fitting Light-Enhancement curves (LE-curves) given an input image. The framework then maps all pixels of the input’s RGB channels by applying the curves iteratively for obtaining the final enhanced image. The input to the DCE-Net is a low-light image while the outputs are a set of pixel-wise curve parameter maps for corresponding higher-order curves. It is a plain CNN of seven convolutional layers with symmetrical concatenation. Each layer consists of 32 convolutional kernels of size 3×3 and stride 1 followed by the ReLU activation function.

![image](https://github.com/Dewanshu232/Denoising-Images/assets/122469929/315263ca-26d8-480c-8405-4c920768eda5)
![image](https://github.com/Dewanshu232/Denoising-Images/assets/122469929/233486bc-30e8-4cc3-b233-c0a61c8734c2)


## Building of Model-
![image](https://github.com/Dewanshu232/Denoising-Images/assets/122469929/922a0f66-e104-4cd5-bee4-0146c6b48f59)


## Testing data-


## Training and Testing of data-

![image](https://github.com/Dewanshu232/Denoising-Images/assets/122469929/43318965-cc0c-4a28-8c03-72d9062fd7a3)

##PSNR value-
![image](https://github.com/Dewanshu232/Denoising-Images/assets/122469929/356b420a-1f1d-46f1-9db8-1a55b7d1c285)

### Training Code
The training code for the Zero-DCE model is executed with the specified hyperparameters and data preprocessing steps.

### Evaluation Metrics
Upon training completion, the model is evaluated on the training dataset using the following metrics:
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Mean Absolute Error (MAE)

## Summary and Future Improvements

### Findings
The Zero-DCE model effectively enhances low-light images by estimating pixel-wise light enhancement curves. The model achieves competitive PSNR values, indicating a significant improvement in image quality. The visual results also demonstrate the model's capability to enhance image details and overall brightness without introducing significant artifacts.

### Future Improvements
To further enhance the model, the following approaches can be considered:
- Data Augmentation: Augmenting the training dataset with more diverse low-light images can improve the model's robustness and generalization.
- Advanced Loss Functions: Incorporating additional loss functions such as perceptual loss can improve the visual quality of the enhanced images.
- Model Architecture: Experimenting with more advanced neural network architectures, such as attention mechanisms, can further enhance the model's performance.

By implementing these improvements, we can continue to push the boundaries of low-light image enhancement and achieve even better results in future iterations of this project.

### REFERENCES  

https://medium.com/analytics-vidhya/noise-removal-in-images-using-deep-learning-models-3972544372d2
https://www.youtube.com/watch?v=ggJhpgXK6E0
https://github.com/sunilbelde/Imagedenoising-dncnn-ridnet-keras/tree/main/code
https://medium.com/analytics-vidhya/image-denoising-using-deep-learning-dc2b19a3fd54


