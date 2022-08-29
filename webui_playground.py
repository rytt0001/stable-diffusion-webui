import gradio as gr 
import time
import os 
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from io import BytesIO
import base64
import re

from frontend.frontend import draw_gradio_ui
from frontend.frontend import draw_ui_custom
from frontend.ui_functions import resize_image

"""
This file is here to play around with the interface without loading the whole model 

TBD - extract all the UI into this file and import from the main webui. 
"""

GFPGAN = True
RealESRGAN = True 

old_images = []
new_images =[]
old_info = []
new_info = []
old_params =[]
new_params =[]

def run_goBIG():
    pass
def txt2img(*args, **kwargs):
  images = [
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80",
    "https://images.unsplash.com/photo-1554151228-14d9def656e4?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=386&q=80",
    "https://images.unsplash.com/photo-1542909168-82c3e7fdca5c?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8aHVtYW4lMjBmYWNlfGVufDB8fDB8fA%3D%3D&w=1000&q=80",
    "https://images.unsplash.com/photo-1546456073-92b9f0a8d413?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80",
    "https://images.unsplash.com/photo-1601412436009-d964bd02edbc?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=464&q=80",
]
  global new_images
  global new_info
  global new_params
  new_images= images
  new_info = 'random'
  new_params = ('prompt', 'ddim_steps', 'sampler_name', [8], 'realesrgan_model_name', 'ddim_eta', 'n_iter', 'batch_size', 'cfg_choice', 0.75, 0.75, 'height', 'width', 'fp', -1)
  return images, 1234, new_info, 'random output'
def img2img(*args, **kwargs):
    images = [
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80",
    "https://images.unsplash.com/photo-1554151228-14d9def656e4?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=386&q=80",
    "https://images.unsplash.com/photo-1542909168-82c3e7fdca5c?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8aHVtYW4lMjBmYWNlfGVufDB8fDB8fA%3D%3D&w=1000&q=80",
    "https://images.unsplash.com/photo-1546456073-92b9f0a8d413?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80",
    "https://images.unsplash.com/photo-1601412436009-d964bd02edbc?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=464&q=80",
    ]
    return images, 1234, 'random', 'random'

def run_GFPGAN(*args, **kwargs):
  time.sleep(.1)
  return "yo"
def run_RealESRGAN(*args, **kwargs):
  time.sleep(.2)
  return "yo"


class model():
  def __init__():
    pass

class opt():
    def __init__(self, name):
        self.name = name

    no_progressbar_hiding = True 

css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
"""

css =  css_hide_progressbar
css = css + """
[data-testid="image"] {min-height: 512px !important};
#main_body {display:none !important};
#main_body>.col:nth-child(2){width:200%;}
"""

user_defaults = {}

# make sure these indicies line up at the top of txt2img()
txt2img_toggles = [
    'Create prompt matrix (separate multiple prompts using |, and get all combinations of them)',
    'Normalize Prompt Weights (ensure sum of weights add up to 1.0)',
    'Save individual images',
    'Save grid',
    'Sort samples by prompt',
    'Write sample info files',
    'jpg samples',
]
if GFPGAN is not None:
    txt2img_toggles.append('Fix faces using GFPGAN')
if RealESRGAN is not None:
    txt2img_toggles.append('Upscale images using RealESRGAN')

txt2img_defaults = {
    'prompt': '',
    'ddim_steps': 50,
    'toggles': [1, 2, 3],
    'sampler_name': 'k_lms',
    'ddim_eta': 0.0,
    'n_iter': 1,
    'batch_size': 1,
    'cfg_scale': 7.5,
    'seed': '',
    'height': 512,
    'width': 512,
    'fp': None,
    'submit_on_enter': 'Yes'
}

if 'txt2img' in user_defaults:
    txt2img_defaults.update(user_defaults['txt2img'])

txt2img_toggle_defaults = [txt2img_toggles[i] for i in txt2img_defaults['toggles']]

sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

# make sure these indicies line up at the top of img2img()
img2img_toggles = [
    'Create prompt matrix (separate multiple prompts using |, and get all combinations of them)',
    'Normalize Prompt Weights (ensure sum of weights add up to 1.0)',
    'Loopback (use images from previous batch when creating next batch)',
    'Random loopback seed',
    'Save individual images',
    'Save grid',
    'Sort samples by prompt',
    'Write sample info files',
    'jpg samples',
]
if GFPGAN is not None:
    img2img_toggles.append('Fix faces using GFPGAN')
if RealESRGAN is not None:
    img2img_toggles.append('Upscale images using RealESRGAN')

img2img_mask_modes = [
    "Keep masked area",
    "Regenerate only masked area",
]

img2img_resize_modes = [
    "Just resize",
    "Crop and resize",
    "Resize and fill",
]

img2img_defaults = {
    'prompt': '',
    'ddim_steps': 50,
    'toggles': [1, 4, 5],
    'sampler_name': 'k_lms',
    'ddim_eta': 0.0,
    'n_iter': 1,
    'batch_size': 1,
    'cfg_scale': 5.0,
    'denoising_strength': 0.75,
    'mask_mode': 0,
    'resize_mode': 0,
    'seed': '',
    'height': 512,
    'width': 512,
    'fp': None,
}

if 'img2img' in user_defaults:
    img2img_defaults.update(user_defaults['img2img'])

img2img_toggle_defaults = [img2img_toggles[i] for i in img2img_defaults['toggles']]
img2img_image_mode = 'sketch'


css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
"""

styling = """
[data-testid="image"] {min-height: 512px !important}
* #body>.col:nth-child(2){width:250%;max-width:89vw}
#generate{width: 100%; }
#prompt_row input{
 font-size:20px
 }
input[type=number]:disabled { -moz-appearance: textfield;+ }
"""

def SaveToHistory():
    global old_images
    global new_images
    global old_info
    global new_info
    global old_params
    global new_params
    old_images = new_images
    old_info = new_info
    old_params = new_params
    print("push")
    return old_images, old_info


def SaveToCsv(image_id_array, use_history=False):
    import csv
    import time
    import base64
    img = image_id_array[0]
    index = int(image_id_array[1])
    print(f"Saving parameter for Image at index: {index}")
    image_data = re.sub('^data:image/.+;base64,', '', img)
    processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
    if processed_image is None:
        return
    global old_params
    global new_params
    os.makedirs("log/images", exist_ok=True)
        # those must match the "txt2img" function !! + images, seed, comment, stats !! NOTE: changes to UI output must be reflected here too

    filenames = []




    savedImgs = processed_image
    params = new_params if not use_history else old_params
    prompt, ddim_steps, sampler_name, toggles, realesrgan_model_name, ddim_eta, n_iter, batch_size, cfg_choice, cfg_scale, pcfg_scale, height, width, fp, seed = params
    seed = seed+(index)
    cfg = cfg_scale if cfg_choice == 0 else pcfg_scale
    with open("log/Resultlog.csv", "a", encoding="utf8", newline='') as file:

        at_start = file.tell() == 0
        writer = csv.writer(file, delimiter =",")
        if at_start:
            writer.writerow(["prompt", "seed", "width", "height", "sampler", "toggles", "n_iter", "n_samples", "cfg_scale", "steps", "filename", "RealESRGAN name"])

        filename_base = str(int(time.time() * 1000))

        filename = "log/images/"+filename_base + "-"+str(int(index)) + ".png"
        savedImgs.save(filename)
        #if savedImgs.startswith("data:image/png;base64,"):
            #savedImgs = savedImgs[len("data:image/png;base64,"):]

        #with open(filename, "wb") as imgfile:
            #imgfile.write(base64.decodebytes(savedImgs.encode('utf-8')))

        use_RealESRGAN = 8 in toggles

        writer.writerow([prompt if prompt else " ", seed, width, height, sampler_name, toggles, n_iter, batch_size, cfg, ddim_steps, filename, realesrgan_model_name if use_RealESRGAN else 'Disabled'])
    print("Parameter Saved")
    return gr.update(value ="log/Resultlog.csv")


def SaveToCsvHistory(image_id_array):
    return SaveToCsv(image_id_array, True)

#demo = draw_gradio_ui(opt,
                      #user_defaults=user_defaults,
                      #txt2img=txt2img,
                      #img2img=img2img,
                      #txt2img_defaults=txt2img_defaults,
                      #txt2img_toggles=txt2img_toggles,
                      #txt2img_toggle_defaults=txt2img_toggle_defaults,
                      #show_embeddings=hasattr(model, "embedding_manager"),
                      #img2img_defaults=img2img_defaults,
                      #img2img_toggles=img2img_toggles,
                      #img2img_toggle_defaults=img2img_toggle_defaults,
                      #img2img_mask_modes=img2img_mask_modes,
                      #img2img_resize_modes=img2img_resize_modes,
                      #sample_img2img=sample_img2img,
                      #RealESRGAN=RealESRGAN,
                      #GFPGAN=GFPGAN,
                      #run_GFPGAN=run_GFPGAN,
                      #run_RealESRGAN=run_RealESRGAN
                        #)
demo = draw_ui_custom(opt,
                      user_defaults=user_defaults,
                      txt2img=txt2img,
                      img2img=img2img,
                      txt2img_defaults=txt2img_defaults,
                      txt2img_toggles=txt2img_toggles,
                      txt2img_toggle_defaults=txt2img_toggle_defaults,
                      show_embeddings=hasattr(model, "embedding_manager"),
                      img2img_defaults=img2img_defaults,
                      img2img_toggles=img2img_toggles,
                      img2img_toggle_defaults=img2img_toggle_defaults,
                      img2img_mask_modes=img2img_mask_modes,
                      img2img_resize_modes=img2img_resize_modes,
                      sample_img2img=sample_img2img,
                      RealESRGAN=RealESRGAN,
                      GFPGAN=GFPGAN,
                      run_GFPGAN=run_GFPGAN,
                      run_RealESRGAN=run_RealESRGAN,
                      run_goBIG=run_goBIG,
                      SaveToHistory=SaveToHistory,
                      SaveToCsv=SaveToCsv,
                      SaveToCsvHistory=SaveToCsvHistory
    )
# demo.queue()
demo.launch(share=False, debug=True)
