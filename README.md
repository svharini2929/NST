# 🎨 Neural Style Transfer using PyTorch & Gradio

Transform ordinary images into artistic masterpieces using **Neural Style Transfer (NST)** powered by **Deep Learning**.

This project applies the style of one image (artwork) onto another image (content) using a pretrained **VGG19 Convolutional Neural Network**. A simple and interactive **Gradio web interface** allows users to experiment with different styles and control parameters in real time.

---

## 🚀 Features

✅ Apply artistic styles to any image  
✅ Uses pretrained **VGG19 (ImageNet)** model  
✅ GPU acceleration (if CUDA available)  
✅ Adjustable style & content weights  
✅ Interactive UI with **Gradio**  
✅ Fully implemented in **PyTorch**

---

## 🧠 How It Works

Neural Style Transfer separates and recombines:

- **Content** → Structure of the image
- **Style** → Texture, colors, patterns

The model:

1. Extracts feature maps from VGG19 layers
2. Computes **Content Loss**
3. Computes **Style Loss** using **Gram Matrices**
4. Optimizes a target image using **Adam Optimizer**
5. Produces a stylized output image

---

## 🏗️ Model Details

- **Architecture:** VGG19 (pretrained on ImageNet)
- **Framework:** PyTorch
- **Optimization:** Adam
- **Loss Functions:**
  - Mean Squared Error (Content Loss)
  - Gram Matrix-based Style Loss

---

## 🎛️ Adjustable Parameters

Users can control:

- **Steps** → More steps = Better quality, slower
- **Content Weight (Alpha)** → Preserve image structure
- **Style Weight (Beta)** → Strength of artistic style

---

## 🖥️ Interface

The project uses **Gradio Blocks UI** for a clean web-based interface:

✔ Upload Content Image  
✔ Upload Style Image  
✔ Adjust parameters via sliders  
✔ Generate stylized image  

---

