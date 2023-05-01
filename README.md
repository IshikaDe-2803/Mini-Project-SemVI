# Mini-Project-SemVI

# Problem Statement
The aim of this project is to take an input image from one domain and translate it into an output image that belongs to another domain while preserving its visual content.
Image-to-image translation refers to a category of problems in the field of computer vision and graphics, where the objective is to learn the relationship between an input image and an output image using a dataset that contains pairs of aligned images.

# Neural Style Transfer

Neural Style Transfer is a technique in deep learning that combines the content of one image with the style of another image to create a new, stylized image. 

This technique uses a convolutional neural network (CNN) to extract the features of the content image and style image separately and then optimizes a third image to minimize both the content loss and the style loss.

By minimizing both losses, the algorithm generates a new image that has the same content as the content image but with the style of the style image.

Neural Style Transfer has various applications in computer graphics, image and video processing, and even in the creation of art.

# Technology Stack

    Frontend:
    User Interface: Streamlit

    Backend - Python Libraries:
    Machine Learning: PyTorch
    Image processing: PIL, numpy
    Visualization: Matplotlib

    IDE and version control:
    Google Colaboratory

# Cycle GAN

Method for unpaired image to image translation using conditional GAN's
Can capture the characteristics of one image domain and figure out how these characteristics could be translated into another image domain, all in the absence of any paired training examples.
CycleGAN uses a cycle consistency loss to enable training without the need for paired data. In other words, it can translate from one domain to another without a one-to-one mapping between the source and target domain.
This opens up the possibility to do a lot of interesting tasks like photo-enhancement, image colorization, style transfer, etc. All you need is the source and the target dataset (which is simply a directory of images).

# Results

![image](https://user-images.githubusercontent.com/81436870/235454204-371c955b-516a-4dd8-8674-e2559fa09462.png)

![image](https://user-images.githubusercontent.com/81436870/235454225-63d30992-339e-40ad-b55c-5fef918ea899.png)

![image](https://user-images.githubusercontent.com/81436870/235454244-25cd7ccd-a93f-4c4b-9696-663582b2262d.png)

![image](https://user-images.githubusercontent.com/81436870/235454262-7f9d5ca1-ff04-45d8-bff7-a4a3af5cdbe3.png)

![image](https://user-images.githubusercontent.com/81436870/235454274-fd1573f6-5093-4d15-b097-c7924123978a.png)

![image](https://user-images.githubusercontent.com/81436870/235454298-b6a5792e-3390-4856-8074-c82659c8e822.png)

# References

1. https://arxiv.org/pdf/1508.06576.pdf
2. https://arxiv.org/pdf/1703.10593.pdf 
3. https://arxiv.org/pdf/1611.07004.pdf 
4. https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-neural-style-transfer-ef88e46697ee 
5. https://en.wikipedia.org/wiki/Neural_style_transfer
6. https://www.tensorflow.org/tutorials/generative/style_transfer
7. https://pytorch.org/tutorials/advanced/neural_style_tutorial.html8. 
8. https://keras.io/examples/generative/neural_style_transfer

The Neural Style Transfer code has been run on Google Colab and can be found [here](https://colab.research.google.com/drive/1onTmze3i0jthd22I_nkpUSzGVkNHxMQu?usp=sharing).

The Cycle GAN code has been run on Kaggle and can be found [here](https://www.kaggle.com/code/ishikade/cyclegan-pytorch-implementation).




