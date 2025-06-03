import cv2
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from datetime import datetime
import subprocess
import os
import secrets

############################# start - variables ################################
sr = cv2.dnn_superres.DnnSuperResImpl_create()

models_2x = ['EDSR_x2.pb', 'ESPCN_x2.pb',
             'FSRCNN-small_x2.pb', 'FSRCNN_x2.pb', 'LapSRN_x2.pb']
models_3x = ['EDSR_x3.pb', 'ESPCN_x3.pb', 'FSRCNN-small_x3.pb', 'FSRCNN_x3.pb']
models_4x = ['EDSR_x4.pb', 'ESPCN_x4.pb',
             'FSRCNN-small_x4.pb', 'FSRCNN_x4.pb', 'LapSRN_x4.pb',
             'realesrgan-x4plus']
models_8x = ['LapSRN_x8.pb']

BASE_PATH = 'models'
STREAMLIT = False # change it to False if you are running in your local machine.

############################# start - functions ################################

def upscale_realesrgan(scale: str, img, img_type: str):
    input_filename = f'input_{secrets.token_urlsafe(10)}.{img_type}'
    output_filename = f'output_{secrets.token_urlsafe(10)}.{img_type}'
    scale = int(scale.split('x')[0])
    realesrgan_path = os.path.join(BASE_PATH, 'realesrgan-win-portable')
    exe_file = os.path.join(realesrgan_path, 'realesrgan-ncnn-vulkan.exe')
    
    if scale == 4:    
        model_file = 'realesrgan-x4plus'
    
    cv2.imwrite(input_filename, img)
    #./realesrgan-ncnn-vulkan.exe -i input.jpg -o output.jpg -n realesrgan-x4plus
    cmd_args = [exe_file, '-i', input_filename, '-o', output_filename, '-n', model_file]    
    subprocess.run(cmd_args)
    
    os.remove(input_filename)
    save_path = f'result.{img_type}'
    result = cv2.imread(output_filename, cv2.IMREAD_COLOR)  
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    os.remove(output_filename)
    return result, save_path

def upscale(model_path: str, model_name: str, scale: str, img, img_type: str):
    img_type = img_type.split('/')[1]

    if model_name == 'realesrgan':
        return upscale_realesrgan(scale, img, img_type)
    
    scale = int(scale.split('x')[0])
    sr.readModel(model_path)
    sr.setModel(model_name, scale)
    result = sr.upsample(img)
    img_type = img_type.split('/')[1]
    save_path = f'result.{img_type}'
    plt.imsave(save_path, result[:, :, ::-1])
    return result[:, :, ::-1], save_path


def get_modelname(selected_model: str) -> str:
    if 'EDSR' in selected_model:
        return 'edsr'
    elif 'LapSRN' in selected_model:
        return 'lapsrn'
    elif 'ESPCN' in selected_model:
        return 'espcn'
    elif 'FSRCNN' in selected_model:
        return 'fsrcnn'
    elif 'LapSRN' in selected_model:
        return 'lapsrn'
    elif 'realesrgan' in selected_model:
        return 'realesrgan'


def model_selector(scale: str) -> tuple:
    model = ''
    options = ['Not selected']
    if scale == '2x':
        options.extend(models_2x)
    elif scale == '3x':
        options.extend(models_3x)
    elif scale == '4x':
        options.extend(models_4x)
    elif scale == '8x':
        options.extend(models_8x)
    else:
        return False, False
    
    model = st.selectbox(
        'Which model do you want to use?',
        tuple(options)
    )

    model_name = get_modelname(model)
    return model, model_name


# def model_selector(scale: str) -> tuple:
#     model = ''
#     if scale == '2x':
#         model = st.selectbox(
#             'Which model do you want to use?',
#             ('Not selected', models_2x[0], models_2x[1], models_2x[2], models_2x[3],
#              models_2x[4]))
#     elif scale == '3x':
#         model = st.selectbox(
#             'Which model do you want to use?',
#             ('Not selected', models_3x[0], models_3x[1], models_3x[2], models_3x[3]))
#     elif scale == '4x':
#         model = st.selectbox(
#             'Which model do you want to use?',
#             ('Not selected', models_4x[0], models_4x[1], models_4x[2], models_4x[3], models_4x[4]))
#     elif scale == '8x':
#         model = st.selectbox(
#             'Which model do you want to use?',
#             ('Not selected', models_8x[0]))
#     else:
#         return False, False

#     model_name = get_modelname(model)
#     return model, model_name


############################# start - Streamlit ################################

st.title('Image Upscaler Using Deep Learning')
st.markdown(
    'Adapted from [Mehrdad Mohammadian](https://mehrdad-dev.github.io)', unsafe_allow_html=True)
scale = st.selectbox(
    'Which scale do you want to apply to your image?',
    ('Not selected', '2x', '3x', '4x', '8x'))


uploaded_file = None
model, model_name = model_selector(scale)
if model and model != 'Not selected':
    model_path = os.path.join(BASE_PATH, scale, model)
    uploaded_file = st.file_uploader("Upload a jpg image", type=["jpg", "png"])


image = None
if uploaded_file is not None:
    # file_details = {"Filename":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, channels="BGR", caption='Your uploaded image')

    if scale == '8x' and image.shape[0] <= 128 and STREAMLIT==True:
        st.error("Your image for the 8x scale is too big, because there is a shortage \
             in terms of CPU, to solve this issue use GitHub codes on your own device or \
            **please select another image or use another scale twice.**")
    elif scale == '4x' and image.shape[0] <= 200 and STREAMLIT==True:
        st.error("Your image for the 4x scale is too big, because there is a shortage \
             in terms of CPU, to solve this issue use GitHub codes on your own device or \
            **please select another image or use another scale twice.**")  
    elif scale == '3x' and image.shape[0] <= 540 and STREAMLIT==True:
        st.error("Your image for the 3x scale is too big, because there is a shortage \
             in terms of CPU, to solve this issue use GitHub codes on your own device or \
            **please select another image or use another scale twice.**")    
    elif scale == '2x' and image.shape[0] <= 550 and STREAMLIT==True:
        st.error("Your image for the 3x scale is too big, because there is a shortage \
             in terms of CPU, to solve this issue use GitHub codes on your own device or \
            **please select another image or use another scale twice.**")                                       
    else:
        left_column, right_column = st.columns(2)
        pressed = left_column.button('Upscale!')

        if pressed:
            pressed = False
            st.info('Processing ...')
            result, save_path = upscale(
                model_path, model_name, scale, image, uploaded_file.type)
            st.success('Image is ready, you can download it now!')
            st.balloons()
            st.image(result, channels="RGB", caption='Your upscaled image')
            with open(save_path, 'rb') as f:
                st.download_button('Download the image', f, file_name=scale +
                                   '_' + str(datetime.now()) + '_' + save_path)
