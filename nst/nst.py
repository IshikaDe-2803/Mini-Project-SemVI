import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from gan import G_BA
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

Tensor = torch.Tensor


def load_checkpoint(ckpt_path, map_location=torch.device('cpu')):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt



st.title('NeuralArt')
st.write("Upload an image and get generated stylized images!")
st.sidebar.write("## Upload and download :gear:")

def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def stylize(image_url):
    if style_option == 'Ukiyoe':
        g = load_checkpoint('./ukiyoe.ckpt')
    if style_option == 'Van Gogh':
        g = load_checkpoint('./vangogh.ckpt')
    if style_option == 'Monet':
        g = load_checkpoint('./monet.ckpt')

    G_BA.load_state_dict(g['G_BA'])
    generate_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    to_image = transforms.ToPILImage()

    G_BA.eval()

    imgs = []
    img = Image.open(image_url)
    img = generate_transforms(img)
    imgs.append(img)
    imgs = torch.stack(imgs, 0).type(Tensor)
    fake_imgs = G_BA(imgs).detach().cpu()
    for j in range(fake_imgs.size(0)):
        img = fake_imgs[j].squeeze().permute(1, 2, 0)
        img_arr = img.numpy()
        img_arr = (img_arr - np.min(img_arr)) * 255 / (np.max(img_arr) - np.min(img_arr))
        img_arr = img_arr.astype(np.uint8)        
        img = Image.fromarray(img_arr)
        img.save(os.path.join('./', 'stylized.png'))

    

def fix_image(content, isContent = False):
    content_image = Image.open(content)
    col1.write("Input Image :camera:")
    col1.image(content_image)
    if isContent:
        stylize(content)
        output_image = Image.open('./stylized.png')
        col2.write("Stylized Image :art:")
        col2.image(output_image)

        st.sidebar.markdown("\n")

        output_image = convert_image(output_image)
        st.sidebar.download_button( label="Download image",
                                    data=output_image,
                                    file_name="stylized.png",
                                    mime="image/png")
     


col1, col2 = st.columns(2)

style_option = st.sidebar.selectbox(
    'Select the style that you want:',
    ('Van Gogh', 'Ukiyoe', 'Monet'))

content_image = st.sidebar.file_uploader("Upload a input image :camera:", type=["png", "jpg", "jpeg"], key=1)

if content_image is not None:
    fix_image(content=content_image, isContent=True)
else:
    file_ = open("./nstvid.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="gif">',
        unsafe_allow_html=True,
    )
    # fix_image(content="./content/1.png")




