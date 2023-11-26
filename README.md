# Neural Style Transfer Models

This repository contains two PyTorch implementations of neural style transfer models: one fine-tuned VGG model and another model trained from scratch. 
## Prerequisites: 

Add the "best-artworks-of-all-time" dataset in your kaggle workspace if you want to train your above models. But, if you want to use only style transfer algorithm then 
use the pretrained weights for both the kaggle notebooks

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
   Train the model from scratch using the provided training script (`train_scratch_model.py`). Customize training parameters as needed. If you want to use the
   trained weights then fetch it using this link or you can see the custom examples for the resuls. Although the custom model results are not as good as the fine tuned VGG,
   it needs further hyperparameter tuning or if needed you can increase the diversity of your dataset for modelling better feature extractor 

2. **Usage:**
   Incorporate the scratch-trained model into your Kaggle notebook by using the provided functions in the 'scratch_model.py' file.

## Other Approaches

### 1. Encoder-Decoder Model
Instead of initializing random image pixels, consider using an Encoder-Decoder model for style transfer. In this approach, the generated image is sampled from the latent space of the style and content images, providing a more controlled and efficient training process. To implement this, explore the 'encoder_decoder' branch of the repository.

### 2. Generative Adversarial Networks (GANs)
Enhance the style transfer algorithm by integrating Generative Adversarial Networks (GANs). GANs introduce adversarial training, where a generator network creates stylized images, and a discriminator distinguishes real from generated ones. This addition can lead to more realistic and diverse stylized outputs. To explore GAN-based style transfer, check out the 'gan_integration' branch.

### 3. Diffusion Models
Consider leveraging diffusion models in style transfer, incorporating techniques from generative models that learn complex data distributions. Diffusion models introduce a probabilistic approach to style transfer, providing fine-grained control over stylized output and the ability to handle a broader range of artistic variations. To explore this, refer to the 'diffusion_models' branch, which includes training the model on a diverse dataset of artistic styles.

Feel free to explore each approach in the respective branches and provide feedback or improvements.


