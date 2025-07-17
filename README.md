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
![Content](https://github.com/user-attachments/assets/b5fdc43a-5dae-4968-8a52-38dc4a2f6981)


**Style Image**  
![Style](https://github.com/user-attachments/assets/95296d53-6bed-4ff7-9f5c-c4d8d57ec377)

---

##Output:-



