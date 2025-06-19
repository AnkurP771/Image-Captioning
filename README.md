# Image-captioning
A deep learning project for generating captions from images using CNN + LSTM

# 🖼️ Image Captioning using Deep Learning (CNN + LSTM)

This project implements an end-to-end image captioning model using a Convolutional Neural Network (Xception) for feature extraction and an LSTM-based Recurrent Neural Network for generating captions. The model is trained on the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) and predicts meaningful natural language descriptions for new images.

---

## 📂 Project Structure

ML_Project_visual/
│
├── Flickr8k_Dataset/
│ └── Flicker8k_Dataset/ # Folder containing all .jpg images
│
├── Flickr8k_text/
│ ├── Flickr8k.token.txt # Image-caption pairs
│ ├── Flickr_8k.trainImages.txt # Training image names
│ └── Flickr_8k.testImages.txt # Testing image names
│
├── features.p # Pickled image feature vectors
├── descriptions.txt # Cleaned captions
├── tokenizer.p # Saved tokenizer
├── models2/ # Trained models saved after each epoch
└── model.png # Visual model architecture
