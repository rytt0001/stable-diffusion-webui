import gradio as gr
from frontend.css_and_js import *
from frontend.css_and_js import css
from frontend.css_and_js import js
import frontend.ui_functions as uifn

def draw_gradio_ui(opt, img2img=lambda x: x, txt2img=lambda x: x, txt2img_defaults={}, RealESRGAN=True, GFPGAN=True,
                   txt2img_toggles={}, txt2img_toggle_defaults='k_euler', show_embeddings=False, img2img_defaults={},
                   img2img_toggles={}, img2img_toggle_defaults={}, sample_img2img=None, img2img_mask_modes=None,
                   img2img_resize_modes=None, user_defaults={}, run_GFPGAN=lambda x: x, run_RealESRGAN=lambda x: x):

    with gr.Blocks(css=css(opt), analytics_enabled=False, title="Stable Diffusion WebUI") as demo:
        with gr.Tabs(elem_id='tabss') as tabs:
            with gr.TabItem("Stable Diffusion Text-to-Image Unified", id='txt2img_tab'):
                with gr.Row(elem_id="prompt_row"):
                    txt2img_prompt = gr.Textbox(label="Prompt",
                                                elem_id='prompt_input',
                                                placeholder="A corgi wearing a top hat as an oil painting.",
                                                lines=1,
                                                max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25,
                                                value=txt2img_defaults['prompt'],
                                                show_label=False)
                    txt2img_btn = gr.Button("Generate", elem_id="generate", variant="primary")

                with gr.Row(elem_id='body').style(equal_height=False):
                    with gr.Column():
                        txt2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height",
                                                   value=txt2img_defaults["height"])
                        txt2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width",
                                                  value=txt2img_defaults["width"])
                        txt2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5,
                                                label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)',
                                                value=txt2img_defaults['cfg_scale'])
                        txt2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, max_lines=1,
                                                  value=txt2img_defaults["seed"])
                        txt2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1,
                                                        label='Batch count (how many batches of images to generate)',
                                                        value=txt2img_defaults['n_iter'])
                        txt2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1,
                                                       label='Batch size (how many images are in a batch; memory-hungry)',
                                                       value=txt2img_defaults['batch_size'])
                    with gr.Column():
                        output_txt2img_gallery = gr.Gallery(label="Images", elem_id="txt2img_gallery_output").style(grid=[4, 4])

                        with gr.Tabs():
                            with gr.TabItem("Generated image actions", id="text2img_actions_tab"):
                                gr.Markdown(
                                    'Select an image from the gallery, then click one of the buttons below to perform an action.')
                                with gr.Row():
                                    output_txt2img_copy_clipboard = gr.Button("Copy to clipboard").click(fn=None,
                                                                                                         inputs=output_txt2img_gallery,
                                                                                                         outputs=[],
                                                                                                         _js=js_copy_to_clipboard('txt2img_gallery_output'))
                                    output_txt2img_copy_to_input_btn = gr.Button("Push to img2img")
                                    if RealESRGAN is not None:
                                        output_txt2img_to_upscale_esrgan = gr.Button("Upscale w/ ESRGAN")

                            with gr.TabItem("Output Info", id="text2img_output_info_tab"):
                                output_txt2img_params = gr.Textbox(label="Generation parameters", interactive=False)
                                with gr.Row():
                                    output_txt2img_copy_params = gr.Button("Copy full parameters").click(
                                        inputs=output_txt2img_params, outputs=[],
                                        _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                    output_txt2img_seed = gr.Number(label='Seed', interactive=False, visible=False)
                                    output_txt2img_copy_seed = gr.Button("Copy only seed").click(
                                        inputs=output_txt2img_seed, outputs=[],
                                        _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                output_txt2img_stats = gr.HTML(label='Stats')
                    with gr.Column():

                        txt2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps",
                                                  value=txt2img_defaults['ddim_steps'])
                        txt2img_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)',
                                                       choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a',
                                                                'k_euler', 'k_heun', 'k_lms'],
                                                       value=txt2img_defaults['sampler_name'])
                        with gr.Tabs():
                            with gr.TabItem('Simple'):
                                txt2img_submit_on_enter = gr.Radio(['Yes', 'No'],
                                                                   label="Submit on enter? (no means multiline)",
                                                                   value=txt2img_defaults['submit_on_enter'],
                                                                   interactive=True)
                                txt2img_submit_on_enter.change(
                                    lambda x: gr.update(max_lines=1 if x == 'Yes' else 25), txt2img_submit_on_enter,
                                    txt2img_prompt)
                            with gr.TabItem('Advanced'):
                                txt2img_toggles = gr.CheckboxGroup(label='', choices=txt2img_toggles,
                                                                   value=txt2img_toggle_defaults, type="index")
                                txt2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model',
                                                                            choices=['RealESRGAN_x4plus',
                                                                                     'RealESRGAN_x4plus_anime_6B'],
                                                                            value='RealESRGAN_x4plus',
                                                                            visible=RealESRGAN is not None)  # TODO: Feels like I shouldnt slot it in here.
                                txt2img_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA",
                                                             value=txt2img_defaults['ddim_eta'], visible=False)
                                txt2img_variant_amount = gr.Slider(minimum=0.0, maximum=1.0, label='Variation Amount',
                                                                   value=txt2img_defaults['variant_amount'])
                                txt2img_variant_seed = gr.Textbox(label="Variant Seed (blank to randomize)", lines=1,
                                                                  max_lines=1, value=txt2img_defaults["variant_seed"])

                        txt2img_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                     visible=show_embeddings)

                txt2img_btn.click(
                    txt2img,
                    [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_toggles, txt2img_realesrgan_model_name,
                     txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfg, txt2img_seed,
                     txt2img_height, txt2img_width, txt2img_embeddings, txt2img_variant_amount, txt2img_variant_seed],
                    [output_txt2img_gallery, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]
                )
                txt2img_prompt.submit(
                    txt2img,
                    [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_toggles, txt2img_realesrgan_model_name,
                     txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfg, txt2img_seed,
                     txt2img_height, txt2img_width, txt2img_embeddings, txt2img_variant_amount, txt2img_variant_seed],
                    [output_txt2img_gallery, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]
                )

            with gr.TabItem("Stable Diffusion Image-to-Image Unified", id="img2img_tab"):
                with gr.Row(elem_id="prompt_row"):
                    img2img_prompt = gr.Textbox(label="Prompt",
                                                elem_id='img2img_prompt_input',
                                                placeholder="A fantasy landscape, trending on artstation.",
                                                lines=1,
                                                max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25,
                                                value=img2img_defaults['prompt'],
                                                show_label=False).style()
                    img2img_btn_mask = gr.Button("Generate", variant="primary", visible=False,
                                                 elem_id="img2img_mask_btn")
                    img2img_btn_editor = gr.Button("Generate", variant="primary", elem_id="img2img_edit_btn")
                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        gr.Markdown('#### Img2Img input')
                        img2img_image_editor = gr.Image(value=sample_img2img, source="upload", interactive=True,
                                                        type="pil", tool="select", elem_id="img2img_editor",
                                                        image_mode="RGBA")
                        img2img_image_mask = gr.Image(value=sample_img2img, source="upload", interactive=True,
                                                      type="pil", tool="sketch", visible=False,
                                                      elem_id="img2img_mask")

                        with gr.Row():
                            img2img_image_editor_mode = gr.Radio(choices=["Mask", "Crop", "Uncrop"], label="Image Editor Mode",
                                                             value="Crop", elem_id='edit_mode_select')

                            img2img_painterro_btn = gr.Button("Advanced Editor")
                            img2img_show_help_btn = gr.Button("Show Hints")
                            img2img_hide_help_btn = gr.Button("Hide Hints", visible=False)
                        img2img_help = gr.Markdown(visible=False, value="")



                    with gr.Column():
                        gr.Markdown('#### Img2Img Results')
                        output_img2img_gallery = gr.Gallery(label="Images", elem_id="img2img_gallery_output").style(grid=[4,4,4])
                        with gr.Tabs():
                            with gr.TabItem("Generated image actions", id="img2img_actions_tab"):
                                with gr.Group():
                                    gr.Markdown("Select an image, then press one of the buttons below")
                                    output_img2img_copy_to_clipboard_btn = gr.Button("Copy to clipboard")
                                    output_img2img_copy_to_input_btn = gr.Button("Push to img2img input")
                                    output_img2img_copy_to_mask_btn = gr.Button("Push to img2img input mask")
                                    gr.Markdown("Warning: This will clear your current image and mask settings!")
                            with gr.TabItem("Output info", id="img2img_output_info_tab"):
                                output_img2img_params = gr.Textbox(label="Generation parameters")
                                with gr.Row():
                                    output_img2img_copy_params = gr.Button("Copy full parameters").click(
                                        inputs=output_img2img_params, outputs=[],
                                        _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                    output_img2img_seed = gr.Number(label='Seed', interactive=False, visible=False)
                                    output_img2img_copy_seed = gr.Button("Copy only seed").click(
                                        inputs=output_img2img_seed, outputs=[],
                                        _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                output_img2img_stats = gr.HTML(label='Stats')
                gr.Markdown('# img2img settings')
                with gr.Row():
                    with gr.Column():
                        img2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1,
                                                        label='Batch count (how many batches of images to generate)',
                                                        value=img2img_defaults['n_iter'])
                        img2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width",
                                                  value=img2img_defaults["width"])
                        img2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height",
                                                   value=img2img_defaults["height"])
                        img2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1,
                                                  value=img2img_defaults["seed"])
                        img2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps",
                                                  value=img2img_defaults['ddim_steps'])
                        img2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1,
                                                       label='Batch size (how many images are in a batch; memory-hungry)',
                                                       value=img2img_defaults['batch_size'])
                    with gr.Column():
                        img2img_mask = gr.Radio(choices=["Keep masked area", "Regenerate only masked area"],
                                                label="Mask Mode", type="index",
                                                value=img2img_mask_modes[img2img_defaults['mask_mode']], visible=False)
                        img2img_mask_blur_strength = gr.Slider(minimum=1, maximum=10, step=1,
                                                               label="How much blurry should the mask be? (to avoid hard edges)",
                                                               value=3, visible=False)

                        img2img_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)',
                                                       choices=["DDIM", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler',
                                                                'k_heun', 'k_lms'],
                                                       value=img2img_defaults['sampler_name'])
                        img2img_toggles = gr.CheckboxGroup(label='', choices=img2img_toggles,
                                                           value=img2img_toggle_defaults, type="index")
                        img2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model',
                                                                    choices=['RealESRGAN_x4plus',
                                                                             'RealESRGAN_x4plus_anime_6B'],
                                                                    value='RealESRGAN_x4plus',
                                                                    visible=RealESRGAN is not None)  # TODO: Feels like I shouldnt slot it in here.


                        img2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5,
                                                label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)',
                                                value=img2img_defaults['cfg_scale'])
                        img2img_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength',
                                                      value=img2img_defaults['denoising_strength'])

                        img2img_resize = gr.Radio(label="Resize mode",
                                                  choices=["Just resize", "Crop and resize", "Resize and fill"],
                                                  type="index",
                                                  value=img2img_resize_modes[img2img_defaults['resize_mode']])
                        img2img_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                     visible=show_embeddings)

                img2img_image_editor_mode.change(
                    uifn.change_image_editor_mode,
                    [img2img_image_editor_mode, img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                    [img2img_image_editor, img2img_image_mask, img2img_btn_editor, img2img_btn_mask,
                     img2img_painterro_btn, img2img_mask, img2img_mask_blur_strength]
                )

                img2img_image_editor.edit(
                    uifn.update_image_mask,
                    [img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                    img2img_image_mask
                )

                img2img_show_help_btn.click(
                    uifn.show_help,
                    None,
                    [img2img_show_help_btn, img2img_hide_help_btn, img2img_help]
                )

                img2img_hide_help_btn.click(
                    uifn.hide_help,
                    None,
                    [img2img_show_help_btn, img2img_hide_help_btn, img2img_help]
                )

                output_txt2img_copy_to_input_btn.click(
                    uifn.copy_img_to_input,
                    [output_txt2img_gallery],
                    [img2img_image_editor, img2img_image_mask, tabs],
                    _js=js_move_image('txt2img_gallery_output', 'img2img_editor')
                )

                output_img2img_copy_to_input_btn.click(
                    uifn.copy_img_to_edit,
                    [output_img2img_gallery],
                    [img2img_image_editor, tabs, img2img_image_editor_mode],
                    _js=js_move_image('img2img_gallery_output', 'img2img_editor')
                )
                output_img2img_copy_to_mask_btn.click(
                    uifn.copy_img_to_mask,
                    [output_img2img_gallery],
                    [img2img_image_mask, tabs, img2img_image_editor_mode],
                    _js=js_move_image('img2img_gallery_output', 'img2img_editor')
                )

                output_img2img_copy_to_clipboard_btn.click(fn=None, inputs=output_img2img_gallery, outputs=[],
                                                           _js=js_copy_to_clipboard('img2img_gallery_output'))

                img2img_btn_mask.click(
                    img2img,
                    [img2img_prompt, img2img_image_editor_mode, img2img_image_mask, img2img_mask,
                     img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_toggles,
                     img2img_realesrgan_model_name, img2img_batch_count, img2img_batch_size, img2img_cfg,
                     img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize,
                     img2img_embeddings],
                    [output_img2img_gallery, output_img2img_seed, output_img2img_params, output_img2img_stats]
                )
                def img2img_submit_params():
                    return (img2img,
                    [img2img_prompt, img2img_image_editor_mode, img2img_image_editor, img2img_mask,
                     img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_toggles,
                     img2img_realesrgan_model_name, img2img_batch_count, img2img_batch_size, img2img_cfg,
                     img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize,
                     img2img_embeddings],
                    [output_img2img_gallery, output_img2img_seed, output_img2img_params, output_img2img_stats])
                img2img_btn_editor.click(*img2img_submit_params())
                img2img_prompt.submit(*img2img_submit_params())

                img2img_painterro_btn.click(None, [img2img_image_editor], [img2img_image_editor, img2img_image_mask], _js=js_painterro_launch('img2img_editor'))

            if GFPGAN is not None:
                gfpgan_defaults = {
                    'strength': 100,
                }

                if 'gfpgan' in user_defaults:
                    gfpgan_defaults.update(user_defaults['gfpgan'])

                with gr.TabItem("GFPGAN", id='cfpgan_tab'):
                    gr.Markdown("Fix faces on images")
                    with gr.Row():
                        with gr.Column():
                            gfpgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                            gfpgan_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Effect strength",
                                                        value=gfpgan_defaults['strength'])
                            gfpgan_btn = gr.Button("Generate", variant="primary")
                        with gr.Column():
                            gfpgan_output = gr.Image(label="Output")
                    gfpgan_btn.click(
                        run_GFPGAN,
                        [gfpgan_source, gfpgan_strength],
                        [gfpgan_output]
                    )
            if RealESRGAN is not None:
                with gr.TabItem("RealESRGAN", id='realesrgan_tab'):
                    gr.Markdown("Upscale images")
                    with gr.Row():
                        with gr.Column():
                            realesrgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                            realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus',
                                                                                                   'RealESRGAN_x4plus_anime_6B'],
                                                                value='RealESRGAN_x4plus')
                            realesrgan_btn = gr.Button("Generate")
                        with gr.Column():
                            realesrgan_output = gr.Image(label="Output")
                    realesrgan_btn.click(
                        run_RealESRGAN,
                        [realesrgan_source, realesrgan_model_name],
                        [realesrgan_output]
                    )
                output_txt2img_to_upscale_esrgan.click(
                    uifn.copy_img_to_upscale_esrgan,
                    output_txt2img_gallery,
                    [realesrgan_source, tabs],
                    _js=js_move_image('txt2img_gallery_output', 'img2img_editor'))

        gr.HTML("""
    <div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
        <p>For help and advanced usage guides, visit the <a href="https://github.com/hlky/stable-diffusion-webui/wiki" target="_blank">Project Wiki</a></p>
        <p>Stable Diffusion WebUI is an open-source project. You can find the latest stable builds on the <a href="https://github.com/hlky/stable-diffusion" target="_blank">main repository</a>.
        If you would like to contribute to development or test bleeding edge builds, you can visit the <a href="https://github.com/hlky/stable-diffusion-webui" target="_blank">developement repository</a>.</p>
    </div>
    """)
        # Hack: Detect the load event on the frontend
        # Won't be needed in the next version of gradio
        # See the relevant PR: https://github.com/gradio-app/gradio/pull/2108
        load_detector = gr.Number(value=0, label="Load Detector", visible=False)
        load_detector.change(None, None, None, _js=js(opt))
        demo.load(lambda x: 42, inputs=load_detector, outputs=load_detector)
    return demo



def draw_ui_custom(opt, img2img=lambda x: x, txt2img=lambda x: x, txt2img_defaults={}, RealESRGAN=True, GFPGAN=True,
                   txt2img_toggles={}, txt2img_toggle_defaults='k_euler', show_embeddings=False, img2img_defaults={},
                   img2img_toggles={}, img2img_toggle_defaults={}, sample_img2img=None, img2img_mask_modes=None,
                   img2img_resize_modes=None, user_defaults={}, run_GFPGAN=lambda x: x, run_RealESRGAN=lambda x: x, run_goBIG=lambda x: x, SaveToHistory=lambda x: x, SaveToCsv=lambda x: x, SaveToCsvHistory=lambda x: x, reloadParams = lambda x: x):
    with gr.Blocks(css=css(opt), analytics_enabled=False, title="Stable Diffusion WebUI") as demo:
        with gr.Tabs(elem_id='tabss') as tabs:
            with gr.TabItem("Stable Diffusion Text-to-Image Unified", id='txt2img_tab'):
                with gr.Row(elem_id="prompt_row"):
                    txt2img_prompt = gr.Textbox(label="Prompt",
                    elem_id='prompt_input',
                    placeholder="A corgi wearing a top hat as an oil painting.",
                    lines=1,
                    max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25,
                    value=txt2img_defaults['prompt'],
                    show_label=False)
                    txt2img_btn = gr.Button("Generate", elem_id="generate", variant="primary")
                with gr.Row(elem_id='body').style(equal_height=False):
                    with gr.Column():
                        #Sliders
                        txt2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=txt2img_defaults["height"])
                        txt2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=txt2img_defaults["width"])
                        txt2img_cfgPrecision =gr.Radio(label='CFG Precision', choices=["Normal", "Precise"],type="index", value="Normal")
                        txt2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=txt2img_defaults['cfg_scale'])
                        txt2img_pcfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.1, label='Precise Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=txt2img_defaults['cfg_scale'])
                        txt2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, max_lines=1, value=txt2img_defaults["seed"])
                        txt2img_batch_count = gr.Slider(minimum=1, maximum=20, step=1, label='Batch count (how many batches of images to generate)', value=txt2img_defaults['n_iter'])
                        txt2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=txt2img_defaults['batch_size'])
                        txt2img_goBig_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='GoBIG Detail Enhancment (Lower will look more like the original)', value=0.3,interactive=True,visible=False)
                        txt2img_goBig_steps = gr.Slider(minimum=1, maximum=300, step=1, label='GoBIG Sampling Steps', value=150,interactive=True, visible=False)
                        txt2img_reloadParam_btn = gr.Button("Reload Parameter saved in server RAM")
                    with gr.Column():
                        output_txt2img_history = gr.Box()

                        with output_txt2img_history:
                            #History
                            gr.Markdown("Image History")
                            output_txt2img_showHostory = gr.Checkbox(label="Open", value=True)
                            output_txt2img_history_intern = gr.Box()
                            with output_txt2img_history_intern:
                                output_txt2img_gallery_history = gr.Gallery(label="Old Images", elem_id="txt2img_gallery_output_history").style(grid=[4,4])
                                with gr.Tabs():
                                    with gr.TabItem("Generated image actions", id="text2img_actions_tab"):
                                        with gr.Row():
                                            output_txt2img_copy_clipboard_history = gr.Button("Copy to clipboard").click(fn=None,
                                                                                                            inputs=output_txt2img_gallery_history,
                                                                                                            outputs=[],
                                                                                                            _js=js_copy_to_clipboard('txt2img_gallery_output_history'))
                                            output_txt2img_copy_to_input_btn_history = gr.Button("Copy selected image to img2img input")
                                            output_txt2img_to_upscale_esrgan_history = gr.Button("Upscale w/ Upscaler")
                                            output_txt2img_copy_to_gobig_input_btn_history = gr.Button("Copy selected image to goBig input")
                                            output_txt2img_save_to_csv_btn_history = gr.Button("Save the Selected Image parameters to CSV")
                                    with gr.TabItem("Output Info", id="text2img_output_info_tab"):
                                        output_txt2img_params_history = gr.Textbox(label="generation parameters History")
                                        with gr.Row():
                                            output_txt2img_copy_params_history = gr.Button("Copy full parameters").click(
                                                inputs=output_txt2img_params_history, outputs=[],
                                                _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                            output_txt2img_seed_history = gr.Number(label='Seed', interactive=False, visible=False)
                                            output_txt2img_copy_seed_history = gr.Button("Copy only seed").click(
                                                inputs=output_txt2img_seed_history, outputs=[],
                                                _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)



                        with gr.Box():
                            #Output
                            output_txt2img_gallery = gr.Gallery(label="Images", elem_id="txt2img_gallery_output").style(grid=[4,4])
                            with gr.Tabs():
                                with gr.TabItem("Generated image actions", id="text2img_actions_tab"):
                                    gr.Markdown(
                                        'Select an image from the gallery, then click one of the buttons below to perform an action.')
                                    with gr.Row():
                                        output_txt2img_copy_clipboard = gr.Button("Copy to clipboard").click(fn=None,
                                                                                                            inputs=output_txt2img_gallery,
                                                                                                            outputs=[],
                                                                                                            _js=js_copy_to_clipboard('txt2img_gallery_output'))
                                        output_txt2img_copy_to_input_btn = gr.Button("Push to img2img")
                                        output_txt2img_to_upscale_esrgan = gr.Button("Upscale with Upscaler")
                                        output_txt2img_copy_to_gobig_input_btn = gr.Button("Copy selected image to goBig input")
                                        output_txt2img_save_to_csv_btn = gr.Button("Save the Selected Image parameters to CSV")
                                        output_txt2img_save_to_history_btn = gr.Button("Push images to history")

                                with gr.TabItem("Output Info", id="text2img_output_info_tab"):
                                    output_txt2img_params = gr.Textbox(label="Generation parameters", interactive=False)
                                    with gr.Row():
                                        output_txt2img_copy_params = gr.Button("Copy full parameters").click(
                                            inputs=output_txt2img_params, outputs=[],
                                            _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                        output_txt2img_seed = gr.Number(label='Seed', interactive=False, visible=False)
                                        output_txt2img_copy_seed = gr.Button("Copy only seed").click(
                                            inputs=output_txt2img_seed, outputs=[],
                                            _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                    output_txt2img_stats = gr.HTML(label='Stats')

                    with gr.Column():
                        txt2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=txt2img_defaults['ddim_steps'])
                        txt2img_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=txt2img_defaults['sampler_name'])
                        with gr.Tabs():
                            with gr.TabItem('Simple'):
                                txt2img_submit_on_enter = gr.Radio(['Yes', 'No'], label="Submit on enter? (no means multiline)", value=txt2img_defaults['submit_on_enter'], interactive=True)
                                txt2img_submit_on_enter.change(lambda x: gr.update(max_lines=1 if x == 'Yes' else 25) , txt2img_submit_on_enter, txt2img_prompt)
                            with gr.TabItem('Advanced'):
                                txt2img_togglesBox = gr.CheckboxGroup(label='', choices=txt2img_toggles, value=txt2img_toggle_defaults, type="index")
                                txt2img_realesrgan_model_type = gr.Dropdown(label='Upscaler type (internal is RealESRGAN)', choices=uifn.upscalers_type, value=uifn.upscalers_type[0])
                                txt2img_realesrgan_model_name = gr.Dropdown(label='Upscaler model', choices=uifn.internal_esrgan_model, value=uifn.internal_esrgan_model[0]) # TODO: Feels like I shouldnt slot it in here.
                                txt2img_realesrgan_scale = gr.Slider(minimum=2.0, maximum=4.0, step=1, label="Upscale Ratio", value=4)
                                txt2img_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=txt2img_defaults['ddim_eta'], visible=False)
                                txt2img_variant_amount = gr.Slider(minimum=0.0, maximum=1.0, label='Variation Amount',
                                                                   value=txt2img_defaults['variant_amount'])
                                txt2img_variant_step = gr.Slider(minimum=0.0, maximum=1.0,label="Variation steps (will be added to variation amount for each iteration only if Variant seed is provided)",
                                                                  value=0.0)
                                txt2img_variant_seed = gr.Textbox(label="Variant Seed (blank to randomize)", lines=1,
                                                                  max_lines=1, value=txt2img_defaults["variant_seed"])
                        txt2img_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                     visible=show_embeddings)


                txt2img_realesrgan_model_type.change(
                    uifn.change_model,
                    [txt2img_realesrgan_model_type],
                    [txt2img_realesrgan_model_name]
                )
                output_txt2img_showHostory.change(
                    uifn.switch_history_vis,
                    [output_txt2img_showHostory],
                    [output_txt2img_history_intern]
                )
                txt2img_togglesBox.change(
                    uifn.show_gobig,
                    [txt2img_togglesBox],
                    [txt2img_goBig_strength, txt2img_goBig_steps]
                )
                txt2img_btn.click(
                    txt2img,
                    [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_togglesBox, txt2img_realesrgan_model_type,txt2img_realesrgan_model_name,txt2img_realesrgan_scale, txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfgPrecision, txt2img_cfg, txt2img_pcfg,txt2img_goBig_strength,txt2img_goBig_steps, txt2img_seed, txt2img_height, txt2img_width, txt2img_embeddings, txt2img_variant_amount, txt2img_variant_seed,txt2img_variant_step],
                    [output_txt2img_gallery, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]
                )
                txt2img_prompt.submit(
                    txt2img,
                    [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_togglesBox, txt2img_realesrgan_model_type,txt2img_realesrgan_model_name,txt2img_realesrgan_scale, txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfgPrecision, txt2img_cfg, txt2img_pcfg,txt2img_goBig_strength,txt2img_goBig_steps, txt2img_seed, txt2img_height, txt2img_width, txt2img_embeddings, txt2img_variant_amount, txt2img_variant_seed,txt2img_variant_step],
                    [output_txt2img_gallery, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]
                )


                output_txt2img_save_to_history_btn.click(
                    SaveToHistory,
                    None,
                    [output_txt2img_gallery_history,output_txt2img_params_history]
                )
                txt2img_reloadParam_btn.click(
                    reloadParams,
                    None,
                    [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_togglesBox, txt2img_realesrgan_model_type,txt2img_realesrgan_model_name,txt2img_realesrgan_scale, txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfgPrecision, txt2img_cfg, txt2img_pcfg,txt2img_goBig_strength,txt2img_goBig_steps, txt2img_seed, txt2img_height, txt2img_width, txt2img_embeddings, txt2img_variant_amount, txt2img_variant_seed,txt2img_variant_step,output_txt2img_gallery, output_txt2img_params, output_txt2img_gallery_history,output_txt2img_params_history]
                    )


            with gr.TabItem("Stable Diffusion Image-to-Image Unified", id="img2img_tab"):
                with gr.Row(elem_id="prompt_row"):
                    img2img_prompt = gr.Textbox(label="Prompt",
                                                elem_id='img2img_prompt_input',
                                                placeholder="A fantasy landscape, trending on artstation.",
                                                lines=1,
                                                max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25,
                                                value=img2img_defaults['prompt'],
                                                show_label=False).style()
                    img2img_btn_mask = gr.Button("Generate", variant="primary", visible=False,
                                                 elem_id="img2img_mask_btn")
                    img2img_btn_editor = gr.Button("Generate", variant="primary", elem_id="img2img_edit_btn")
                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        gr.Markdown('#### Img2Img input')
                        img2img_image_editor = gr.Image(value=sample_img2img, source="upload", interactive=True,
                                                        type="pil", tool="select", elem_id="img2img_editor",image_mode="RGBA")
                        img2img_image_mask = gr.Image(value=sample_img2img, source="upload", interactive=True,
                                                      type="pil", tool="sketch", visible=False,
                                                      elem_id="img2img_mask")

                        with gr.Row():
                            img2img_image_editor_mode = gr.Radio(choices=["Mask", "Crop", "Uncrop"], label="Image Editor Mode",
                                                             value="Crop", elem_id='edit_mode_select')

                            img2img_painterro_btn = gr.Button("Advanced Editor")
                            img2img_copy_from_painterro_btn = gr.Button(value="Get Image from Advanced Editor")
                            img2img_show_help_btn = gr.Button("Show Hints")
                            img2img_hide_help_btn = gr.Button("Hide Hints", visible=False)
                        img2img_help = gr.Markdown(visible=False, value="")

                    with gr.Column():
                        gr.Markdown('#### Img2Img Results')
                        output_img2img_gallery = gr.Gallery(label="Images", elem_id="img2img_gallery_output").style(grid=[4,4,4])
                        with gr.Tabs():
                            with gr.TabItem("Generated image actions", id="img2img_actions_tab"):
                                with gr.Group():
                                    gr.Markdown("Select an image, then press one of the buttons below")
                                    output_img2img_copy_to_clipboard_btn = gr.Button("Copy to clipboard")
                                    output_img2img_copy_to_input_btn = gr.Button("Push to img2img input")
                                    output_img2img_copy_to_mask_btn = gr.Button("Push to img2img input mask")
                                    gr.Markdown("Warning: This will clear your current image and mask settings!")
                            with gr.TabItem("Output info", id="img2img_output_info_tab"):
                                output_img2img_params = gr.Textbox(label="Generation parameters")
                                with gr.Row():
                                    output_img2img_copy_params = gr.Button("Copy full parameters").click(
                                        inputs=output_img2img_params, outputs=[],
                                        _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                    output_img2img_seed = gr.Number(label='Seed', interactive=False, visible=False)
                                    output_img2img_copy_seed = gr.Button("Copy only seed").click(
                                        inputs=output_img2img_seed, outputs=[],
                                        _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                output_img2img_stats = gr.HTML(label='Stats')
                gr.Markdown('# img2img settings')
                with gr.Row():
                    with gr.Column():
                        img2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1,
                                                       label='Batch size (how many images are in a batch; memory-hungry)',
                                                       value=img2img_defaults['batch_size'])
                        img2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width",
                                                  value=img2img_defaults["width"])
                        img2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height",
                                                   value=img2img_defaults["height"])
                        img2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1,
                                                  value=img2img_defaults["seed"])
                        img2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps",
                                                  value=img2img_defaults['ddim_steps'])
                        img2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1,
                                                        label='Batch count (how many batches of images to generate)',
                                                        value=img2img_defaults['n_iter'])
                    with gr.Column():
                        img2img_mask = gr.Radio(choices=["Keep masked area", "Regenerate only masked area"],
                                                label="Mask Mode", type="index",
                                                value=img2img_mask_modes[img2img_defaults['mask_mode']], visible=False)
                        img2img_mask_blur_strength = gr.Slider(minimum=1, maximum=10, step=1,
                                                               label="How much blurry should the mask be? (to avoid hard edges)",
                                                               value=3, visible=False)

                        img2img_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)',
                                                       choices=["DDIM", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler',
                                                                'k_heun', 'k_lms'],
                                                       value=img2img_defaults['sampler_name'])
                        img2img_togglesBox = gr.CheckboxGroup(label='', choices=img2img_toggles,
                                                           value=img2img_toggle_defaults, type="index")
                        img2img_realesrgan_model_type = gr.Dropdown(label='Upscaler type (internal is RealESRGAN)', choices=uifn.upscalers_type, value=uifn.upscalers_type[0])
                        img2img_realesrgan_model_name = gr.Dropdown(label='Upscaler model', choices=uifn.internal_esrgan_model, value=uifn.internal_esrgan_model[0])
                        img2img_realesrgan_scale = gr.Slider(minimum=2.0, maximum=4.0, step=1, label="Upscale Ratio", value=4)
                        img2img_goBig_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='GoBIG Detail Enhancment (Lower will look more like the original)', value=0.3,interactive=True)
                        img2img_goBig_steps = gr.Slider(minimum=1, maximum=300, step=1, label='GoBIG Sampling Steps', value=150,interactive=True)
                        img2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5,
                                                label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)',
                                                value=img2img_defaults['cfg_scale'])
                        img2img_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength',
                                                      value=img2img_defaults['denoising_strength'])

                        img2img_resize = gr.Radio(label="Resize mode",
                                                  choices=["Just resize", "Crop and resize", "Resize and fill"],
                                                  type="index",
                                                  value=img2img_resize_modes[img2img_defaults['resize_mode']])
                        img2img_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                     visible=show_embeddings)






                img2img_realesrgan_model_type.change(
                    uifn.change_model,
                    [img2img_realesrgan_model_type],
                    [img2img_realesrgan_model_name]
                )
                img2img_togglesBox.change(
                        uifn.show_gobig,
                        [img2img_togglesBox],
                        [img2img_goBig_strength, img2img_goBig_steps]
                )

                img2img_image_editor_mode.change(
                    uifn.change_image_editor_mode,
                    [img2img_image_editor_mode, img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                    [img2img_image_editor, img2img_image_mask, img2img_btn_editor, img2img_btn_mask,
                     img2img_painterro_btn, img2img_mask, img2img_mask_blur_strength]
                )

                img2img_image_editor.edit(
                    uifn.copy_img_to_input,
                    output_txt2img_gallery,
                    [img2img_image_editor, img2img_image_mask, tabs],
                    _js=js_move_image('txt2img_gallery_output', 'img2img_editor')
                )

                img2img_show_help_btn.click(
                    uifn.show_help,
                    None,
                    [img2img_show_help_btn, img2img_hide_help_btn, img2img_help]
                )

                img2img_hide_help_btn.click(
                    uifn.hide_help,
                    None,
                    [img2img_show_help_btn, img2img_hide_help_btn, img2img_help]
                )

                output_txt2img_copy_to_input_btn.click(
                    uifn.copy_img_to_input,
                    output_txt2img_gallery,
                    [img2img_image_editor, img2img_image_mask, tabs],
                    _js=js_move_image('txt2img_gallery_output', 'img2img_editor')
                )
                output_txt2img_copy_to_input_btn_history.click(
                    uifn.copy_img_to_input,
                    output_txt2img_gallery_history,
                    [img2img_image_editor, img2img_image_mask, tabs],
                    _js=js_move_image('txt2img_gallery_output_history', 'img2img_editor')
                )
                output_img2img_copy_to_input_btn.click(
                    uifn.copy_img_to_edit,
                    output_img2img_gallery,
                    [img2img_image_editor, tabs, img2img_image_editor_mode],
                    _js=js_move_image('img2img_gallery_output', 'img2img_editor')
                )
                output_img2img_copy_to_mask_btn.click(
                    uifn.copy_img_to_mask,
                    output_img2img_gallery,
                    [img2img_image_mask, tabs, img2img_image_editor_mode],
                    _js=js_move_image('img2img_gallery_output', 'img2img_editor')
                )

                output_img2img_copy_to_clipboard_btn.click(fn=None, inputs=output_img2img_gallery, outputs=[],
                                                           _js=js_copy_to_clipboard('img2img_gallery_output'))

                img2img_btn_mask.click(
                    img2img,
                    [img2img_prompt, img2img_image_editor_mode, img2img_image_mask, img2img_mask,
                     img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_togglesBox,img2img_realesrgan_model_type,
                     img2img_realesrgan_model_name,img2img_realesrgan_scale, img2img_batch_count, img2img_batch_size, img2img_cfg,txt2img_goBig_strength,txt2img_goBig_steps,
                     img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize,
                     img2img_embeddings],
                    [output_img2img_gallery, output_img2img_seed, output_img2img_params, output_img2img_stats]
                )
                def img2img_submit_params():
                    return (img2img,
                    [img2img_prompt, img2img_image_editor_mode, img2img_image_editor, img2img_mask,
                     img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_togglesBox,img2img_realesrgan_model_type,
                     img2img_realesrgan_model_name,img2img_realesrgan_scale, img2img_batch_count, img2img_batch_size, img2img_cfg,txt2img_goBig_strength,txt2img_goBig_steps,
                     img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize,
                     img2img_embeddings],
                    [output_img2img_gallery, output_img2img_seed, output_img2img_params, output_img2img_stats])
                img2img_btn_editor.click(*img2img_submit_params())
                img2img_prompt.submit(*img2img_submit_params())

                img2img_painterro_btn.click(None, [img2img_image_editor], [img2img_image_editor, img2img_image_mask], _js=js_painterro_launch('img2img_editor'))


                img2img_copy_from_painterro_btn.click(None, None, [img2img_image_editor, img2img_image_mask], _js="""() => {
                const image = localStorage.getItem('painterro-image')
                return [image, image];
            }""")



            gfpgan_defaults = {
                'strength': 100,
            }

            if 'gfpgan' in user_defaults:
                gfpgan_defaults.update(user_defaults['gfpgan'])

            with gr.TabItem("GFPGAN",id="gfpgan_tab"):
                gr.Markdown("Fix faces on images")
                with gr.Row():
                    with gr.Column():
                        gfpgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                        gfpgan_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Effect strength", value=gfpgan_defaults['strength'])
                        gfpgan_btn = gr.Button("Generate")
                    with gr.Column():
                        gfpgan_output = gr.Image(label="Output")
                gfpgan_btn.click(
                    run_GFPGAN,
                    [gfpgan_source, gfpgan_strength],
                    [gfpgan_output]
                )

            with gr.TabItem("RealESRGAN",id="realesrgan_tab"):
                gr.Markdown("Upscale images")
                with gr.Row():
                    with gr.Column():
                        realesrgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="esrgan_img")
                        realesrgan_model_type = gr.Dropdown(label='Upscaler type (internal is RealESRGAN)', choices=uifn.upscalers_type, value=uifn.upscalers_type[0])
                        realesrgan_model_name = gr.Dropdown(label='Upscaler model', choices=uifn.internal_esrgan_model, value=uifn.internal_esrgan_model[0])
                        realesrgan_scale = gr.Slider(minimum=2.0, maximum=4.0, step=1, label="Upscale Ratio", value=4)
                        realesrgan_btn = gr.Button("Generate")
                    with gr.Column():
                        realesrgan_output = gr.Image(label="Output")
                realesrgan_model_type.change(
                    uifn.change_model,
                    [realesrgan_model_type],
                    [realesrgan_model_name]
                )
                realesrgan_btn.click(
                    run_RealESRGAN,
                    [realesrgan_source,realesrgan_model_type, realesrgan_model_name, realesrgan_scale],
                    [realesrgan_output]
                )
                output_txt2img_to_upscale_esrgan.click(
                    uifn.copy_img_to_upscale_esrgan,
                    output_txt2img_gallery,
                    [realesrgan_source, tabs],
                    _js=js_move_image('txt2img_gallery_output', 'esrgan_img')
                )
                output_txt2img_to_upscale_esrgan_history.click(
                    uifn.copy_img_to_upscale_esrgan,
                    output_txt2img_gallery_history,
                    [realesrgan_source, tabs],
                    _js=js_move_image('txt2img_gallery_output_history', 'esrgan_img')
                )
            with gr.TabItem("goBIG",id="gobig_tab"):
                gr.Markdown("Upscale and detail images")
                with gr.Row():
                    with gr.Column():
                        realesrganGoBig_source = gr.Image(source="upload", interactive=True, type="pil", tool="select", elem_id="gobig_img")
                        realesrganGoBig_model_type = gr.Dropdown(label='Upscaler type (internal is RealESRGAN)', choices=uifn.upscalers_type, value=uifn.upscalers_type[0])
                        realesrganGoBig_model_name = gr.Dropdown(label='Upscaler model', choices=uifn.internal_esrgan_model, value=uifn.internal_esrgan_model[0])
                        realesrganGoBig_prompt = gr.Textbox(label="Prompt",
                        elem_id='img2img_prompt_input',
                        placeholder="A fantasy landscape, trending on artstation.",
                        lines=1,
                        value="",
                        show_label=False).style()
                        GoBig_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='GoBIG Detail Enhancment (Lower will look more like the original)', value=0.3,interactive=True)
                        GoBig_steps = gr.Slider(minimum=1, maximum=300, step=1, label='GoBIG Sampling Steps', value=150,interactive=True)
                        realesrganGoBig_btn = gr.Button("Generate")
                    with gr.Column():
                        realesrganGoBig_output = gr.Image(label="Output")
                realesrganGoBig_model_type.change(
                    uifn.change_model,
                    [realesrganGoBig_model_type],
                    [realesrganGoBig_model_name]
                )
                realesrganGoBig_btn.click(
                    run_goBIG,
                    [realesrganGoBig_source, realesrganGoBig_model_type, realesrganGoBig_model_name, GoBig_strength, GoBig_steps, realesrganGoBig_prompt],
                    [realesrganGoBig_output]
                )
                output_txt2img_copy_to_gobig_input_btn.click(
                    uifn.copy_img_to_upscale_gobig,
                    output_txt2img_gallery,
                    [realesrganGoBig_source,tabs],
                    _js=js_move_image('txt2img_gallery_output', 'gobig_img')
                )
                output_txt2img_copy_to_gobig_input_btn_history.click(
                    uifn.copy_img_to_upscale_gobig,
                    output_txt2img_gallery_history,
                    [realesrganGoBig_source,tabs],
                    _js=js_move_image('txt2img_gallery_output', 'gobig_img')
                )
            with gr.TabItem("Parameter History"):
                def refresh():
                    return gr.update(value ="log/Resultlog.csv")
                with gr.Column():
                    csv_file = gr.Dataframe(value="log/Resultlog.csv", elem_id="CSV_DRAW").style(rounded=[True, True,False,False])
                    csv_btn = gr.Button("Refresh", elem_id="refresh_csv").style(rounded=[False, False,True,True],full_width=True).click(refresh, None, [csv_file])

            output_txt2img_save_to_csv_btn_history.click(
                SaveToCsvHistory,
                output_txt2img_gallery_history,
                [csv_file],
                _js=js_save_to_csv_image('txt2img_gallery_output_history')
            )
            output_txt2img_save_to_csv_btn.click(
                SaveToCsv,
                output_txt2img_gallery,
                [csv_file],
                _js=js_save_to_csv_image('txt2img_gallery_output')
            )



        # Hack: Detect the load event on the frontend
        # Won't be needed in the next version of gradio
        # See the relevant PR: https://github.com/gradio-app/gradio/pull/2108
        load_detector = gr.Number(value=0, label="Load Detector", visible=False)
        load_detector.change(None, None, None, _js=js(opt))
        demo.load(lambda x: 42, inputs=load_detector, outputs=load_detector)
        demo.queue(concurrency_count=1)
    return demo
