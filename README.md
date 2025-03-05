# 🧠 **CSC578 - Final Research Project**  
### **Hebbian Learning-Inspired Object Detection Model**  

## 📌 **Introduction**  
This project explores an alternative approach to object detection by integrating **Hebbian Learning** principles into a deep learning framework. Our goal was to evaluate whether this method could lead to **faster learning** compared to conventional **Gradient-Based Backpropagation**.  

The inspiration for this research comes from the original **[HebbNet](https://ieeexplore.ieee.org/document/9414241)** paper, which demonstrated Hebbian Learning on an **MNIST digit classification** task. We aimed to extend this concept to object detection models.  

## 🛠 **Technology Stack**  
- **Detectron2** (Object Detection Framework)  
- **Python** (Core Development)  
- **Hugging Face** (Pretrained Models & NLP Integration)  
- **HebbNet** (Hebbian Learning Model)  

## 🔬 **Methodology**  
We followed a structured approach inspired by **HebbNet**, incorporating Hebbian Learning in the **feature extraction** phase of our object detection model. Our implementation involved the following key steps:  

* **Modifying Detectron2 Backbone**  
  - We started with **Facebook Research’s Detectron2** framework.  
  - Instead of using a standard **ResNet backbone**, we replaced it with a **custom HebbNet-based backbone** for feature extraction.  

* **Using Hebbian Learning for Feature Extraction**  
  - Due to resource limitations, we applied **Hebbian Learning** **only to dense layers** in the feature extraction phase.  
  - We **did not modify convolutional layers**, as implementing a Hebbian Learning algorithm for them requires additional research and computational resources.  

📄 **For more details, refer to our** `Project_Report.pdf` **in this repository.**  

## 🔮 **Future Work**  
While our current implementation focuses on **dense layers**, future directions include:  
- **Implementing Hebbian Learning for Convolutional Layers** – Once viable implementations are available, we aim to extend Hebbian principles to CNN architectures.   

🚀 **This research opens the door to alternative learning paradigms, potentially enhancing the efficiency of object detection models!**  
