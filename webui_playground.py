import gradio as gr 
import time
import os 
"""
This file is here to play around with the interface without loading the whole model 

TBD - extract all the UI into this file and import from the main webui. 
"""

GFPGAN = True
RealESRGAN = True 
def txt2img(*args, **kwargs):
  time.sleep(.5)
  return "yay"
def img2img(*args, **kwargs):
  time.sleep(.2)
  return "yay"
def run_GFPGAN(*args, **kwargs):
  time.sleep(.1)
  return "yo"
def run_RealESRGAN(*args, **kwargs):
  time.sleep(.2)
  return "yo"


class model():
  def __init__():
    pass

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

def change_image_editor_mode(choice, cropped_image, resize_mode, width, height):
    if choice == "Mask":
        return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)]
    return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]

def update_image_mask(cropped_image, resize_mode, width, height):
    resized_cropped_image = resize_image(resize_mode, cropped_image, width, height) if cropped_image else None
    return gr.update(value=resized_cropped_image)

def copy_img_to_input(selected=1, imgs = []):
    try:
        idx = int(0 if selected - 1 < 0 else selected - 1)
        image_data = re.sub('^data:image/.+;base64,', '', imgs[idx])
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        update = gr.update(selected='img2img_tab')
        return [processed_image, processed_image, update]
    except IndexError:
        return [None, None]

help_text = """
    ## Mask/Crop
    * The masking/cropping is very temperamental.
    * It may take some time for the image to show when switching from Crop to Mask.
    * If the image doesn't appear after switching to Mask, switch back to Crop and then back again to Mask
    * If the mask appears distorted (the brush is weirdly shaped instead of round), switch back to Crop and then back again to Mask.
    ## Advanced Editor
    * For now the button needs to be clicked twice the first time.
    * Once you have edited your image, you _need_ to click the save button for the next step to work.
    * Clear the image from the crop editor (click the x)
    * Click "Get Image from Advanced Editor" to get the image you saved. If it doesn't work, try opening the editor and saving again.
    If it keeps not working, try switching modes again, switch tabs, clear the image or reload.
"""

def run():
  print

def show_help():
    return [gr.update(visible=False), gr.update(visible=True), gr.update(value=help_text)]

def hide_help():
    return [gr.update(visible=True), gr.update(visible=False), gr.update(value="")]

styling = """
[data-testid="image"] {min-height: 512px !important};
*{
    border: 1px solid red
}
* #body>.col:nth-child(2){width:250%;max-width:89vw}
#generate{width: 100%; }
#prompt_row input{
 font-size:20px
 }

"""
with gr.Blocks(css=styling, analytics_enabled=False, title="Stable Diffusion WebUI") as demo:
    gr.Markdown('# Stable diffusion webui')
    with gr.Tabs(elem_id='tabss') as tabs:
        with gr.TabItem("txt2img", id="txt2img_tab"):
            with gr.Row(elem_id="prompt_row"):
                txt2img_prompt = gr.Textbox(label="Prompt", 
                elem_id='prompt_input',
                placeholder="A corgi wearing a top hat as an oil painting.", 
                lines=1,
                max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25, 
                value=txt2img_defaults['prompt'], 
                show_label=False).style()
                
            with gr.Row(elem_id='body').style(equal_height=False):
                with gr.Column():
                    txt2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=txt2img_defaults["height"])
                    txt2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=txt2img_defaults["width"])
                    txt2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=txt2img_defaults['cfg_scale'])
                    txt2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, value=txt2img_defaults["seed"])                    
                    txt2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1, label='Batch count (how many batches of images to generate)', value=txt2img_defaults['n_iter'])
                    txt2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=txt2img_defaults['batch_size'])
                with gr.Column():
                    output_txt2img_gallery = gr.Gallery(label="Images", elem_id="gallery_output").style(grid=[4,4])
                    with gr.Row():
                        with gr.Group():
                            output_txt2img_seed = gr.Number(label='Seed', interactive=False)
                            output_txt2img_copy_seed = gr.Button("Copy", full_width=True).click(inputs=output_txt2img_seed, outputs=[], _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                        with gr.Group():
                            output_txt2img_select_image = gr.Number(label='Image # and click Copy to copy to img2img', value=1, precision=None)
                            output_txt2img_copy_to_input_btn = gr.Button("Push to img2img", full_width=True)
                    with gr.Group():
                        output_txt2img_params = gr.Textbox(label="Copy-paste generation parameters", interactive=False)
                        output_txt2img_copy_params = gr.Button("Copy", full_width=True).click(inputs=output_txt2img_params, outputs=[], _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                    output_txt2img_stats = gr.HTML(label='Stats')
                with gr.Column():
                    txt2img_btn = gr.Button("Generate", full_width=True, elem_id="generate", variant="primary")
                    txt2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=txt2img_defaults['ddim_steps'])
                    txt2img_sampling = gr.Radio(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=txt2img_defaults['sampler_name'])
                    with gr.Tabs():
                        with gr.TabItem('Simple'):
                            txt2img_submit_on_enter = gr.Radio(['Yes', 'No'], label="Submit on enter? (no means multiline)", value=txt2img_defaults['submit_on_enter'], interactive=True)
                            txt2img_submit_on_enter.change(lambda x: gr.update(max_lines=1 if x == 'Single' else 25) , txt2img_submit_on_enter, txt2img_prompt)
                        with gr.TabItem('Advanced'):
                            txt2img_toggles = gr.CheckboxGroup(label='', choices=txt2img_toggles, value=txt2img_toggle_defaults, type="index")
                            txt2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B'], value='RealESRGAN_x4plus', visible=RealESRGAN is not None) # TODO: Feels like I shouldnt slot it in here.
                            txt2img_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=txt2img_defaults['ddim_eta'], visible=False)
                            txt2img_embeddings = gr.File(label = "Embeddings file for textual inversion", visible=hasattr(model, "embedding_manager"))
                    
            txt2img_btn.click(
                txt2img,
                [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_toggles, txt2img_realesrgan_model_name, txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfg, txt2img_seed, txt2img_height, txt2img_width, txt2img_embeddings],
                [output_txt2img_gallery, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]
            )
            txt2img_prompt.submit(
                txt2img,
                [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_toggles, txt2img_realesrgan_model_name, txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfg, txt2img_seed, txt2img_height, txt2img_width, txt2img_embeddings],
                [output_txt2img_gallery, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]
            )

        with gr.TabItem("img2img with inpainting", id="img2img_tab"):
            with gr.Row(elem_id="prompt_row"):
                img2img_prompt = gr.Textbox(label="Prompt", 
                elem_id='img2img_prompt_input',
                placeholder="A corgi wearing a top hat as an oil painting.", 
                lines=1,
                max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25, 
                value=img2img_defaults['prompt'], 
                show_label=False).style()
                img2img_btn_mask = gr.Button("Generate",variant="primary", visible=False, elem_id="img2img_mask_btn").style(full_width=True)
                img2img_btn_editor = gr.Button("Generate",variant="primary", elem_id="img2img_editot_btn").style(full_width=True)
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    
                    img2img_image_editor_mode = gr.Radio(choices=["Mask", "Crop"], label="Image Editor Mode", value="Crop")
                    img2img_show_help_btn = gr.Button("Show Hints")
                    img2img_hide_help_btn = gr.Button("Hide Hints", visible=False)
                    img2img_help = gr.Markdown(visible=False, value="")
                    with gr.Row():
                        img2img_painterro_btn = gr.Button("Advanced Editor")
                        img2img_copy_from_painterro_btn = gr.Button(value="Get Image from Advanced Editor")
                    img2img_image_editor = gr.Image(value=sample_img2img, source="upload", interactive=True, type="pil", tool="select")
                    img2img_image_mask = gr.Image(value=sample_img2img, source="upload", interactive=True, type="pil", tool="sketch", visible=False)
                    img2img_mask = gr.Radio(choices=["Keep masked area", "Regenerate only masked area"], label="Mask Mode", type="index", value=img2img_mask_modes[img2img_defaults['mask_mode']], visible=False)
                    img2img_mask_blur_strength = gr.Slider(minimum=1, maximum=10, step=1, label="How much blurry should the mask be? (to avoid hard edges)", value=3, visible=False)
                    img2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=img2img_defaults['ddim_steps'])
                    img2img_sampling = gr.Radio(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=img2img_defaults['sampler_name'])
                    img2img_toggles = gr.CheckboxGroup(label='', choices=img2img_toggles, value=img2img_toggle_defaults, type="index")
                    img2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B'], value='RealESRGAN_x4plus', visible=RealESRGAN is not None) # TODO: Feels like I shouldnt slot it in here.
                    img2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1, label='Batch count (how many batches of images to generate)', value=img2img_defaults['n_iter'])
                    img2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=img2img_defaults['batch_size'])
                    img2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=img2img_defaults['cfg_scale'])
                    img2img_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength', value=img2img_defaults['denoising_strength'])
                    img2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, value=img2img_defaults["seed"])
                    img2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=img2img_defaults["height"])
                    img2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=img2img_defaults["width"])
                    img2img_resize = gr.Radio(label="Resize mode", choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value=img2img_resize_modes[img2img_defaults['resize_mode']])
                    img2img_embeddings = gr.File(label = "Embeddings file for textual inversion", visible=hasattr(model, "embedding_manager"))
                    
                with gr.Column():
                    output_img2img_gallery = gr.Gallery(label="Images")
                    output_img2img_select_image = gr.Number(label='Select image number from results for copying', value=1, precision=None)
                    gr.Markdown("Clear the input image before copying your output to your input. It may take some time to load the image.")
                    output_img2img_copy_to_input_btn = gr.Button("Copy selected image to input")
                    output_img2img_seed = gr.Number(label='Seed')
                    output_img2img_params = gr.Textbox(label="Copy-paste generation parameters")
                    output_img2img_stats = gr.HTML(label='Stats')

            img2img_image_editor_mode.change(
                change_image_editor_mode,
                [img2img_image_editor_mode, img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                [img2img_image_editor, img2img_image_mask, img2img_btn_editor, img2img_btn_mask, img2img_painterro_btn, img2img_copy_from_painterro_btn, img2img_mask, img2img_mask_blur_strength]
            )

            img2img_image_editor.edit(
                update_image_mask,
                [img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                img2img_image_mask
            )

            img2img_show_help_btn.click(
                show_help,
                None,
                [img2img_show_help_btn, img2img_hide_help_btn, img2img_help]
            )

            img2img_hide_help_btn.click(
                hide_help,
                None,
                [img2img_show_help_btn, img2img_hide_help_btn, img2img_help]
            )

            output_img2img_copy_to_input_btn.click(
                copy_img_to_input,
                [output_img2img_select_image, output_img2img_gallery],
                [img2img_image_editor, img2img_image_mask]
            )

            output_txt2img_copy_to_input_btn.click(
                copy_img_to_input,
                [output_txt2img_select_image, output_txt2img_gallery],
                [img2img_image_editor, img2img_image_mask, tabs]
            )

            img2img_btn_mask.click(
                img2img,
                [img2img_prompt, img2img_image_editor_mode, img2img_image_mask, img2img_mask, img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_toggles, img2img_realesrgan_model_name, img2img_batch_count, img2img_batch_size, img2img_cfg, img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize, img2img_embeddings],
                [output_img2img_gallery, output_img2img_seed, output_img2img_params, output_img2img_stats]
            )

            img2img_btn_editor.click(
                img2img,
                [img2img_prompt, img2img_image_editor_mode, img2img_image_editor, img2img_mask, img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_toggles, img2img_realesrgan_model_name, img2img_batch_count, img2img_batch_size, img2img_cfg, img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize, img2img_embeddings],
                [output_img2img_gallery, output_img2img_seed, output_img2img_params, output_img2img_stats]
            )

            img2img_painterro_btn.click(None, [img2img_image_editor], None, _js="""(img) => {
                try {
                    Painterro({
                        hiddenTools: ['arrow'],
                        saveHandler: function (image, done) {
                            localStorage.setItem('painterro-image', image.asDataURL());
                            done(true);
                        },
                    }).show(Array.isArray(img) ? img[0] : img);
                } catch(e) {
                    const script = document.createElement('script');
                    script.src = 'https://unpkg.com/painterro@1.2.78/build/painterro.min.js';
                    document.head.appendChild(script);
                    const style = document.createElement('style');
                    style.appendChild(document.createTextNode('.ptro-holder-wrapper { z-index: 9999 !important; }'));
                    document.head.appendChild(style);
                }
                return [];
            }""")

            img2img_copy_from_painterro_btn.click(None, None, [img2img_image_editor, img2img_image_mask], _js="""() => {
                const image = localStorage.getItem('painterro-image')
                return [image, image];
            }""")

        if GFPGAN is not None:
            gfpgan_defaults = {
                'strength': 100,
            }

            if 'gfpgan' in user_defaults:
                gfpgan_defaults.update(user_defaults['gfpgan'])

            with gr.TabItem("GFPGAN"):
                gr.Markdown("Fix faces on images")
                with gr.Row():
                    with gr.Column():
                        gfpgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                        gfpgan_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Effect strength", value=gfpgan_defaults['strength'])
                        gfpgan_btn = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        gfpgan_output = gr.Image(label="Output")
                gfpgan_btn.click(
                    run_GFPGAN,
                    [gfpgan_source, gfpgan_strength],
                    [gfpgan_output]
                )
        if RealESRGAN is not None:
            with gr.TabItem("RealESRGAN"):
                gr.Markdown("Upscale images")
                with gr.Row():
                    with gr.Column():
                        realesrgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                        realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B'], value='RealESRGAN_x4plus')
                        realesrgan_btn = gr.Button("Generate")
                    with gr.Column():
                        realesrgan_output = gr.Image(label="Output")
                realesrgan_btn.click(
                    run_RealESRGAN,
                    [realesrgan_source, realesrgan_model_name],
                    [realesrgan_output]
                )

demo.queue()
demo.launch(share=False, debug=True, encrypt=False)
