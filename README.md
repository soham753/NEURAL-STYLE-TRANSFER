# NEURAL-STYLE-TRANSFER

COMPANY: CODTECH IT SOLUTIONS PVT.LTD

NAME : PATEL SOHAM VIKRAMBHAI

INTERN ID:CT04DG2984

DOMAIN : Artificial Intelligence

DURATION : 4 WEEKS

MENTOR: NEELA SANTOSH


This project implements a **Neural Style Transfer** model using **TensorFlow** and **VGG19**, which allows you to apply artistic styles from one image (style image) to another (content image). The result is a visually captivating image that blends the content of a photograph with the aesthetics of artwork — effectively turning photos into paintings using deep learning.

---

## 📜 Project Description

Neural Style Transfer (NST) is a deep learning technique that blends two images: a content image and a style image, using convolutional neural networks (CNNs). This project uses the **pre-trained VGG19 model** to extract and manipulate image features and construct a new image that maintains the content of the original photo while adopting the artistic characteristics of a chosen style image.

The core idea involves:
- Extracting high-level content representations from the **content image** using the deeper layers of the CNN.
- Extracting stylistic features from the **style image** using the shallower layers.
- Generating a target image and iteratively updating it by minimizing the content and style loss functions to blend both types of features.

This implementation is optimized using **TensorFlow 2.x** and uses **Adam optimizer** for gradient-based optimization. The training loop applies both content and style weights to refine the image progressively over multiple iterations.

This project can be applied to:
- Transform landscape photos into Van Gogh-like paintings.
- Create artistic renditions of selfies using styles from famous artworks.
- Generate creative content for posters, media, and marketing.

---

## 🚀 Features

- ✅ Uses pretrained **VGG19** model from Keras.
- ✅ Fully functional TensorFlow 2.x pipeline.
- ✅ Real-time visualization of transformation.
- ✅ Adjustable weights for style/content influence.
- ✅ Saves the final stylized output image.

---

## 🖼️ Input


## 🧠 How it Works

1. **Preprocessing**:
   - Load and resize content & style images.
   - Convert them to tensors and normalize using VGG19 preprocessing.

2. **Feature Extraction**:
   - Extract intermediate layers from VGG19: deeper for content, shallower for style.
   - Compute Gram matrices to capture texture patterns in style features.

3. **Loss Calculation**:
   - Content loss: Difference between generated image and content image features.
   - Style loss: Difference between Gram matrices of generated image and style image.
   - Total loss = Content loss × weight + Style loss × weight

4. **Optimization**:
   - Using Adam optimizer and gradient tape.
   - The target image is iteratively updated to minimize total loss.

---
##Input

**Content Image**  
<img src="https://github.com/user-attachments/assets/b5fdc43a-5dae-4968-8a52-38dc4a2f6981" width="400"/>



**Style Image**  
<img src="https://github.com/user-attachments/assets/95296d53-6bed-4ff7-9f5c-c4d8d57ec377" width="500"/>


---

##Output:-


<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9f3c39b0-b03c-48c7-a4ba-992728f307e8" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/18314838-8cca-43dd-b9a1-8131d11d2277" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/16b71f0c-3a30-4305-a9a8-d257cc359603" width="300"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/764f7923-46fd-400d-b6a7-2a69c675be8e" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/5b9eb084-3ef4-4acf-909e-3072fca4583c" width="300"/></td>
    <td><img width="300"  alt="Figure_1" src="https://github.com/user-attachments/assets/02ad5a84-26c0-4a80-b101-1460ea0dd92c" /></td>
  </tr>
</table>



