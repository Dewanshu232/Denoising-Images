![image](https://github.com/Dewanshu232/Denoising-Images/assets/122469929/a9ba52b6-af33-4100-a843-c752d7581d43)


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

### PSNR Value Achieved
The Peak Signal-to-Noise Ratio (PSNR) value achieved on the training dataset is computed during evaluation and reported below.

### Paper Implemented
The project is based on the paper: "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement" by Chien-Hsiang Huang, Chia-Yu Chen, Chu-Song Chen, and Chen-Chi Chang. The paper can be accessed [here](https://arxiv.org/abs/2001.06826).
![image](https://github.com/Dewanshu232/Denoising-Images/assets/122469929/f3e62080-9811-4422-b11c-e3be88547b86)



## Project Details

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

