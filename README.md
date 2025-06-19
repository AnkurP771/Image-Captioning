# Image-captioning
A deep learning project for generating captions from images using CNN + LSTM

# 🖼️ Image Captioning using Deep Learning (CNN + LSTM)

This project implements an end-to-end image captioning model using a Convolutional Neural Network (Xception) for feature extraction and an LSTM-based Recurrent Neural Network for generating captions. The model is trained on the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) and predicts meaningful natural language descriptions for new images.

---

## 📂 Project Structure

```
ML_Project_visual/
│
├── Flickr8k_Dataset/
│   └── Flicker8k_Dataset/              # Folder containing all .jpg images
│
├── Flickr8k_text/
│   ├── Flickr8k.token.txt              # Image-caption pairs
│   ├── Flickr_8k.trainImages.txt       # Training image names
│   └── Flickr_8k.testImages.txt        # Testing image names
│
├── features.p                          # Pickled image feature vectors
├── descriptions.txt                    # Cleaned captions
├── tokenizer.p                         # Saved tokenizer
├── models2/                            # Trained models saved after each epoch
└── model.png                           # Visual model architecture
```

---

## 📌 Key Concepts Used

- **Transfer Learning**: Pretrained Xception model from ImageNet used to extract features.
- **LSTM**: Sequence learning model used to generate language based on image context.
- **Tokenizer**: Converts text to sequences for training and inference.
- **Padding**: Ensures uniform sequence lengths using `pad_sequences`.
- **Categorical Crossentropy**: Used for multi-class prediction loss.
- **Softmax Activation**: Outputs probabilities for each word in vocabulary.
- **Data Generator**: Custom generator using `tf.data.Dataset` for efficient training.

---

## 🔧 How It Works

1. **Data Preparation**
    - Extract image-caption mappings from `Flickr8k.token.txt`.
    - Clean captions (remove punctuation, lowercase, etc.)
    - Save cleaned captions to `descriptions.txt`.
    - Build vocabulary and tokenizer.

2. **Feature Extraction**
    - Use pretrained Xception model (without top layer) to extract 2048-dim vectors.
    - Save vectors in `features.p` for reuse.

3. **Sequence Modeling**
    - Define a model with:
        - CNN path: Dense(256) on image features.
        - RNN path: Embedding + LSTM on caption input.
        - Fusion: Add both paths and use Dense layers to predict the next word.

4. **Training**
    - Use a custom generator to prepare input pairs `[image, partial_caption] → next_word`.
    - Train model for multiple epochs, saving after each one.

5. **Inference**
    - Extract features from a new image.
    - Start with `<start>` token, predict next word repeatedly until `<end>` or max length is reached.

---

## 🏗️ Model Architecture

```
Image Feature (2048) → Dropout → Dense(256)
                                ↘
Text Sequence → Embedding → LSTM(256) → 
                                ↘
                       Add [Image + Text]
                             ↓
                      Dense(256) → Dense(vocab_size=7577, softmax)
```

> Visual:
![Model Architecture](model.png)

---

## 🧪 Example Inference

```python
img_path = "/path/to/test/image.jpg"
feature = extract_features(img_path, xception_model)
caption = generate_desc(trained_model, tokenizer, feature, max_length)
print("Caption:", caption)
```

---

## 🚀 Requirements

- Python 3.7+
- TensorFlow / Keras
- NumPy, Pillow
- tqdm, matplotlib
- Google Drive (if using Colab)

Install dependencies:
```bash
pip install tensorflow keras numpy pillow matplotlib tqdm
```

---

## 💾 Training & Checkpointing

- Models saved in `models2/model_{epoch}.h5`.
- Tokenizer saved as `tokenizer.p`
- Feature vectors saved as `features.p`

---

## 📊 Dataset

[Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

- 8000 images
- Each image has 5 human-annotated captions

---

## 🙌 Credits

- 📸 Dataset: Flickr8k
- 📚 Pretrained Model: [Xception](https://keras.io/api/applications/xception/)
- 💻 Implemented and trained using: Python, TensorFlow/Keras

---

## 🧠 Future Work

- Beam Search for better caption generation
- BLEU Score evaluation
- Extend to COCO dataset
- Deploy with Streamlit or Flask

---

## 📌 Sample Output (Optional)
```text
Input Image: 🏇
Predicted Caption: "a man riding a horse on a grassy field"
```

---


