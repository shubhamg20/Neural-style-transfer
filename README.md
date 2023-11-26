# Neural Style Transfer Models

This repository contains two PyTorch implementations of neural style transfer models: one fine-tuned VGG model and another model trained from scratch. 
## Prerequisites: 

Add the "best-artworks-of-all-time" dataset in your kaggle workspace if you want to train your above models. But, if you want to use only style transfer algorithm then use the pretrained weights for both the kaggle notebooks as given below or enjoy analyzing the sample generated images within the notebook

## Pretrained Models

Download the pretrained weights for both models:
- [Fine-Tuned VGG Model](https://drive.google.com/file/d/1l4wKy_5rd905fAaFkhrk0wW9G_dEnvHe/view?usp=sharing) (Place in 'fine_tuned_vgg_weights' directory)
- [Scratch-Trained Model](https://drive.google.com/file/d/1QIsr4WK1nBIdVShpt0KzTrEYQIsN4zSa/view?usp=sharing) (Place in 'scratch_weights' directory)

### 1. Fine-Tuned VGG Model

1. **Usage:**
   Open the provided notebook (`vgg_style_transfer_demo.ipynb`) for a detailed example of using the fine-tuned VGG model for style transfer.

2. **Improvements and other approaches:**
   Please foolow the last section in this notebook

### 2. Scratch-Trained Model

1. **Training:**
   Train the model from scratch using the provided training script (`train_scratch_model.ipynb`). Customize training parameters as needed. If you want to use the
   trained weights then fetch it using this link or you can see the custom examples for the resuls. Although the custom model results are not as good as the fine tuned VGG,
   it needs further hyperparameter tuning or if needed you can increase the diversity of your dataset for modelling better feature extractor 

2. **Usage:**
   Incorporate the scratch-trained model into your Kaggle notebook by using the provided functions in the 'scratch_model.py' file.

## Other Approaches

### 1. Encoder-Decoder Model
Instaed of initializing random image pixels that leads to the increase in number of steps while training, Encoder decoder model can be taken into consideration where your genearted image will be sampled from the latent space of style and content image rather than initializing it randonly.

### 2. Generative Adversarial Networks (GANs)
Integrating Generative Adversarial Networks (GANs) enhances the style transfer algorithm by introducing adversarial training. This involves a generator network creating stylized images and a discriminator discerning real from generated ones.

### 3. Diffusion Models
Leveraging diffusion models in style transfer can integrate techniques from generative models that learn complex data distributions. Incorporating diffusion models introduces a probabilistic approach to style transfer, allowing for fine-grained control over stylized output and the potential to handle a broader range of artistic variations. By training the model on a diverse dataset of artistic styles, the diffusion process learns to generate images that match the desired style.

Feel free to explore each approach in the respective branches and provide feedback or improvements.


