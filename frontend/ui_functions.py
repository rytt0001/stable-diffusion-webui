import re
import gradio as gr
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from io import BytesIO
import base64
import re
import os


def change_image_editor_mode(choice, cropped_image, resize_mode, width, height):
    if choice == "Mask":
        return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)]
    return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]

def update_image_mask(cropped_image, resize_mode, width, height):
    resized_cropped_image = resize_image(resize_mode, cropped_image, width, height) if cropped_image else None
    return gr.update(value=resized_cropped_image)

def copy_img_to_input(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        tab_update = gr.update(selected='img2img_tab')
        img_update = gr.update(value=processed_image)
        return processed_image, processed_image , tab_update
    except IndexError:
        return [None, None]

def copy_img_to_edit(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        tab_update = gr.update(selected='img2img_tab')
        img_update = gr.update(value=processed_image)
        mode_update = gr.update(value='Crop')
        return processed_image, tab_update, mode_update
    except IndexError:
        return [None, None]

def copy_img_to_mask(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        tab_update = gr.update(selected='img2img_tab')
        img_update = gr.update(value=processed_image)
        mode_update = gr.update(value='Mask')
        return processed_image, tab_update, mode_update
    except IndexError:
        return [None, None]



def copy_img_to_upscale_esrgan(img):
    tabs_update = gr.update(selected='realesrgan_tab')
    image_data = re.sub('^data:image/.+;base64,', '', img)
    processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return processed_image, tabs_update

def copy_img_to_upscale_gobig(img):
    tabs_update = gr.update(selected='gobig_tab')
    image_data = re.sub('^data:image/.+;base64,', '', img)
    processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return processed_image, tabs_update

help_text = """
    ## Mask/Crop
    * Masking is not inpainting. You will probably get better results manually masking your images in photoshop instead.
    * Built-in masking/cropping is very temperamental.
    * It may take some time for the image to show when switching from Crop to Mask.
    * If the image doesn't appear after switching to Mask, switch back to Crop and then back again to Mask
    * If the mask appears distorted (the brush is weirdly shaped instead of round), switch back to Crop and then back again to Mask.

    ## Advanced Editor
    * Click 💾 Save to send your editor changes to the img2img workflow
    * Click ❌ Clear to discard your editor changes

    If anything breaks, try switching modes again, switch tabs, clear the image, or reload.
"""

def show_help():
    return [gr.update(visible=False), gr.update(visible=True), gr.update(value=help_text)]

def hide_help():
    return [gr.update(visible=True), gr.update(visible=False), gr.update(value="")]

def resize_image(resize_mode, im, width, height):
    LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
    if resize_mode == 0:
        res = im.resize((width, height), resample=LANCZOS)
    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGBA", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGBA", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res

def switch_history_vis(value:bool):
    return gr.update(visible=value)


def show_gobig(toggles):
    show = 7 in toggles
    return [gr.update(visible=show), gr.update(visible=show)]


upscalers_type = ['Internal']
upscalers_args = ['']
upscaler_scale = [[4]]
if os.path.exists('./realesrgan-ncnn-vulkan') or os.path.exists('./realesrgan-ncnn-vulkan.exe'):
    upscalers_type.append('realesrgan-ncnn-vulkan')
    upscalers_args.append('-n')
    upscaler_scale.append([4])
if os.path.exists('./realesrgan-ncnn-vulkan') or os.path.exists('./realesrgan-ncnn-vulkan.exe'):
    upscalers_type.append('waifu2x-ncnn-vulkan')
    upscalers_args.append('-m')
    upscaler_scale.append([1,2,4])
if os.path.exists('./realesrgan-ncnn-vulkan') or os.path.exists('./realesrgan-ncnn-vulkan.exe'):
    upscalers_type.append('realsr-ncnn-vulkan')
    upscalers_args.append('-m')
    upscaler_scale.append([4])
ext_esrgan_models = ['realesrgan-x4plus', 'realesrgan-x4plus-anime']
ext_srgan_models = ['models-DF2K','models-DF2K_JPEG']
ext_waifu_models = ['models-cunet','models-upconv_7_photo','models-upconv_7_anime_style_art_rgb']

internal_esrgan_model =['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B']

def change_model(model_type: str):
    if model_type == 'realesrgan-ncnn-vulkan':
        return gr.update(choices=ext_esrgan_models, value=ext_esrgan_models[0])
    if model_type == 'waifu2x-ncnn-vulkan':
        return gr.update(choices=ext_waifu_models, value=ext_waifu_models[0])
    if model_type == 'realsr-ncnn-vulkan':
        return gr.update(choices=ext_srgan_models, value=ext_srgan_models[0])
    if model_type == 'Internal':
        return gr.update(choices=internal_esrgan_model, value=internal_esrgan_model[0])
