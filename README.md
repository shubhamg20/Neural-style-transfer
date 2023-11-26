# Neural Style Transfer Models

This repository contains two PyTorch implementations of neural style transfer models: one fine-tuned VGG model and another model trained from scratch. 
## Prerequisites: 

Add the "best-artworks-of-all-time" dataset in your kaggle workspace if you want to train your above models. But, if you want to use only style transfer algorithm then 
use the pretrained weights for both the kaggle notebooks

## Pretrained Models

Download the pretrained weights for both models:
- [Fine-Tuned VGG Model](#) (Place in 'vgg_weights' directory)
- [Scratch-Trained Model](#) (Place in 'scratch_weights' directory)

### 1. Fine-Tuned VGG Model

1. **Usage:**
   Open the provided notebook (`vgg_style_transfer_demo.ipynb`) for a detailed example of using the fine-tuned VGG model for style transfer.

2. **Improvements and other approaches:**
   Please foolow the last section in this notebook

### 2. Scratch-Trained Model

1. **Training:**
   Train the model from scratch using the provided training script (`train_scratch_model.py`). Customize training parameters as needed. If you want to use the
   trained weights then fetch it using this link or you can see the custom examples for the resuls. Although the custom model results are not as good as the fine tuned VGG,
   it needs further hyperparameter tuning or if needed you can increase the diversity of your dataset for modelling better feature extractor 

3. **Usage:**
   Incorporate the scratch-trained model into your Kaggle notebook by using the provided functions in the 'scratch_model.py' file.



