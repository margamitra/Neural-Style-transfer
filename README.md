# Neural Style Transfer Implementation

I have tried to implement the neural style transfer algorithm as described in the paper **"A Neural Algorithm of Artistic Style"** by Gatys et al. https://arxiv.org/abs/1508.06576. The goal of neural style transfer is to combine the *content* of one image with the *style* of another to create a new, stylized image.

The provided implementation uses **PyTorch** and **VGG-19**, a pre-trained convolutional neural network, to extract features from both the content and style images. The generated image is then optimized to minimize the loss between the content, style, and the target output.

## Features
1. Extracts content and style features from selected layers of a pre-trained VGG-19 model.
2. Computes:
   - **Content loss**: Difference between the content image and generated image.
   - **Style loss**: Difference between the style image and the generated image using the Gram matrix.
3. Allows customization of weights for content and style loss.
4. Iteratively generates an output image by optimizing a randomly initialized image.
5. Saves the generated stylized image at intervals during training.

## Code Details

### 1. **VGG Model Implementation**
The implementation uses a modified VGG-19 model to extract features from the content and style images. Key points:
- The pre-trained VGG-19 model is truncated to use layers up to `conv4_1`.
- Feature maps are extracted from selected layers: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, and `conv5_1`.

### 2. **Loss Functions**
Two types of losses are used here:
- **Content Loss**: Measures the difference between the feature representations of the original image and the generated image.
  \[
  \text{Content Loss} = \text{Mean}((F_{\text{generated}} - F_{\text{original}})^2)
  \]
  <img width="536" alt="image" src="https://github.com/user-attachments/assets/80ce275d-edfa-4378-a71e-f8bc24e0a51e">

  
- **Style Loss**: Measures the difference in style using the Gram matrix of feature representations.
  \[
  \text{Style Loss} = \text{Mean}((G_{\text{generated}} - G_{\text{style}})^2)
  \]

  Where \( G = F \cdot F^T \), the Gram matrix.

  <img width="555" alt="image" src="https://github.com/user-attachments/assets/1c5cfdc7-0f91-43e4-bc71-6b3e8172e889">
  


### 3. **Optimization**
- The generated image is initialized as a copy of the content image and optimized directly using the Adam optimizer.
- The total loss is a weighted sum of the content loss and style loss:
  \[
  \text{Total Loss} = \alpha \cdot \text{Content Loss} + \beta \cdot \text{Style Loss}
  \]
<img width="533" alt="image" src="https://github.com/user-attachments/assets/973674f5-e9f5-4048-bee5-bbf00c4a6066">

### 3. **Output**
- Every 200 iterations, the generated image is saved as `generated.png`.
- The final result is a beautiful blend of the content and style images.

