import argparse, os, sys, glob
import gradio as gr
import k_diffusion as K
import math
import mimetypes
import numpy as np
import pynvml
import random
import threading, asyncio
import time
import torch
import torch.nn as nn
import yaml
from typing import List, Union

from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from itertools import islice
from omegaconf import OmegaConf
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from io import BytesIO
import base64
import re
from torch import autocast
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging
    logging.set_verbosity_error()
except:
    pass

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the bowser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')


# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
invalid_filename_chars = '<>:"/\|?*\n'

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default=None)
parser.add_argument("--outdir_txt2img", type=str, nargs="?", help="dir to write txt2img results to (overrides --outdir)", default=None)
parser.add_argument("--outdir_img2img", type=str, nargs="?", help="dir to write img2img results to (overrides --outdir)", default=None)
parser.add_argument("--skip_grid", action='store_true', help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",)
parser.add_argument("--skip_save", action='store_true', help="do not save indiviual samples. For speed measurements.",)
parser.add_argument("--n_rows", type=int, default=-1, help="rows in the grid; use -1 for autodetect and 0 for n_rows to be same as batch_size (default: -1)",)
parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model",)
parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="path to checkpoint of model",)
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN')) # i disagree with where you're putting it but since all guidefags are doing it this way, there you go
parser.add_argument("--realesrgan-dir", type=str, help="RealESRGAN directory", default=('./src/realesrgan' if os.path.exists('./src/realesrgan') else './RealESRGAN'))
parser.add_argument("--realesrgan-model", type=str, help="Upscaling model for RealESRGAN", default=('RealESRGAN_x4plus'))
parser.add_argument("--no-verify-input", action='store_true', help="do not verify input to check if it's too long")
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats")
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware accleration in browser)")
parser.add_argument("--defaults", type=str, help="path to configuration file providing UI defaults, uses same format as cli parameter", default='configs/webui/webui.yaml')
parser.add_argument("--cli", type=str, help="don't launch web server, take Python function kwargs from this file.", default=None)
opt = parser.parse_args()

GFPGAN_dir = opt.gfpgan_dir
RealESRGAN_dir = opt.realesrgan_dir

css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
"""

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def crash(e, s):
    global model
    global device

    print(s, '\n', e)

    del model
    del device

    print('exiting...calling os._exit(0)')
    t = threading.Timer(0.25, os._exit, args=[0])
    t.start()

class MemUsageMonitor(threading.Thread):
    stop_flag = False
    max_usage = 0
    total = 0

    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        print(f"[{self.name}] Recording max memory usage...\n")
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        while not self.stop_flag:
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.max_usage = max(self.max_usage, m.used)
            # print(self.max_usage)
            time.sleep(0.1)
        print(f"[{self.name}] Stopped recording.\n")
        pynvml.nvmlShutdown()

    def read(self):
        return self.max_usage, self.total

    def stop(self):
        self.stop_flag = True

    def read_and_stop(self):
        self.stop_flag = True
        return self.max_usage, self.total

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class KDiffusionSampler:
    def __init__(self, m, sampler):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)
        self.schedule = sampler

    def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T):
        sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)

        samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale}, disable=False)

        return samples_ddim, None

class MemUsageMonitor(threading.Thread):
    stop_flag = False
    max_usage = 0
    total = 0

    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        print(f"[{self.name}] Recording max memory usage...\n")
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        while not self.stop_flag:
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.max_usage = max(self.max_usage, m.used)
            # print(self.max_usage)
            time.sleep(0.1)
        print(f"[{self.name}] Stopped recording.\n")
        pynvml.nvmlShutdown()

    def read(self):
        return self.max_usage, self.total

    def stop(self):
        self.stop_flag = True

    def read_and_stop(self):
        self.stop_flag = True
        return self.max_usage, self.total

def create_random_tensors(shape, seeds):
    xs = []
    for seed in seeds:
        torch.manual_seed(seed)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this so i do not dare change it for now because
        # it will break everyone's seeds.
        xs.append(torch.randn(shape, device=device))
    x = torch.stack(xs)
    return x

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def load_GFPGAN():
    model_name = 'GFPGANv1.3'
    model_path = os.path.join(GFPGAN_dir, 'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception("GFPGAN model not found at path "+model_path)

    sys.path.append(os.path.abspath(GFPGAN_dir))
    from gfpgan import GFPGANer

    return GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

def load_RealESRGAN(model_name: str):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    RealESRGAN_models = {
        'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        'RealESRGAN_x4plus_anime_6B': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    }

    model_path = os.path.join(RealESRGAN_dir, 'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception(model_name+".pth not found at path "+model_path)

    sys.path.append(os.path.abspath(RealESRGAN_dir))
    from realesrgan import RealESRGANer

    instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=True)
    instance.model.name = model_name
    return instance

GFPGAN = None
if os.path.exists(GFPGAN_dir):
    try:
        GFPGAN = load_GFPGAN()
        print("Loaded GFPGAN")
    except Exception:
        import traceback
        print("Error loading GFPGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

RealESRGAN = None
def try_loading_RealESRGAN(model_name: str):
    global RealESRGAN
    if os.path.exists(RealESRGAN_dir):
        try:
            RealESRGAN = load_RealESRGAN(model_name) # TODO: Should try to load both models before giving up
            print("Loaded RealESRGAN with model "+RealESRGAN.model.name)
        except Exception:
            import traceback
            print("Error loading RealESRGAN:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
try_loading_RealESRGAN('RealESRGAN_x4plus')

config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = (model if opt.no_half else model.half()).to(device)

def load_embeddings(fp):
    if fp is not None and hasattr(model, "embedding_manager"):
        model.embedding_manager.load(fp.name)

def image_grid(imgs, batch_size, round_down=False, force_n_rows=None):
    if force_n_rows is not None:
        rows = force_n_rows
    elif opt.n_rows > 0:
        rows = opt.n_rows
    elif opt.n_rows == 0:
        rows = batch_size
    else:
        rows = math.sqrt(len(imgs))
        rows = int(rows) if round_down else round(rows)

    cols = math.ceil(len(imgs) / rows)

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid

def seed_to_int(s):
    if type(s) is int:
        return s
    if s is None or s == '':
        return random.randint(0,2**32)
    n = abs(int(s) if s.isdigit() else hash(s))
    while n > 2**32:
        n = n >> 32
    return n

def draw_prompt_matrix(im, width, height, all_prompts):
    def wrap(text, d, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if d.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return '\n'.join(lines)

    def draw_texts(pos, x, y, texts, sizes):
        for i, (text, size) in enumerate(zip(texts, sizes)):
            active = pos & (1 << i) != 0

            if not active:
                text = '\u0336'.join(text) + '\u0336'

            d.multiline_text((x, y + size[1] / 2), text, font=fnt, fill=color_active if active else color_inactive, anchor="mm", align="center")

            y += size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2
    fonts = ["arial.ttf", "DejaVuSans.ttf"]
    for font_name in fonts:
        try:
            fnt = ImageFont.truetype(font_name, fontsize)
            break
        except OSError:
           pass
    else:
        # ImageFont.load_default() is practically unusable as it only supports
        # latin1, so raise an exception instead
        raise Exception(f"No usable font found (tried {', '.join(fonts)})")
    color_active = (0, 0, 0)
    color_inactive = (153, 153, 153)

    pad_top = height // 4
    pad_left = width * 3 // 4 if len(all_prompts) > 2 else 0

    cols = im.width // width
    rows = im.height // height

    prompts = all_prompts[1:]

    result = Image.new("RGB", (im.width + pad_left, im.height + pad_top), "white")
    result.paste(im, (pad_left, pad_top))

    d = ImageDraw.Draw(result)

    boundary = math.ceil(len(prompts) / 2)
    prompts_horiz = [wrap(x, d, fnt, width) for x in prompts[:boundary]]
    prompts_vert = [wrap(x, d, fnt, pad_left) for x in prompts[boundary:]]

    sizes_hor = [(x[2] - x[0], x[3] - x[1]) for x in [d.multiline_textbbox((0, 0), x, font=fnt) for x in prompts_horiz]]
    sizes_ver = [(x[2] - x[0], x[3] - x[1]) for x in [d.multiline_textbbox((0, 0), x, font=fnt) for x in prompts_vert]]
    hor_text_height = sum([x[1] + line_spacing for x in sizes_hor]) - line_spacing
    ver_text_height = sum([x[1] + line_spacing for x in sizes_ver]) - line_spacing

    for col in range(cols):
        x = pad_left + width * col + width / 2
        y = pad_top / 2 - hor_text_height / 2

        draw_texts(col, x, y, prompts_horiz, sizes_hor)

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + height * row + height / 2 - ver_text_height / 2

        draw_texts(row, x, y, prompts_vert, sizes_ver)

    return result


def resize_image(resize_mode, im, width, height):
    if resize_mode == 0:
        res = im.resize((width, height), resample=LANCZOS)
    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
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


def check_prompt_length(prompt, comments):
    """this function tests if prompt is too long, and if so, adds a message to comments"""

    tokenizer = model.cond_stage_model.tokenizer
    max_length = model.cond_stage_model.max_length

    info = model.cond_stage_model.tokenizer([prompt], truncation=True, max_length=max_length, return_overflowing_tokens=True, padding="max_length", return_tensors="pt")
    ovf = info['overflowing_tokens'][0]
    overflowing_count = ovf.shape[0]
    if overflowing_count == 0:
        return

    vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    overflowing_words = [vocab.get(int(x), "") for x in ovf]
    overflowing_text = tokenizer.convert_tokens_to_string(''.join(overflowing_words))

    comments.append(f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")


def process_images(outpath, func_init, func_sample, prompt, seed, sampler_name, skip_grid, skip_save, batch_size, n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, use_RealESRGAN, realesrgan_model_name, fp, do_not_save_grid=False, normalize_prompt_weights=True, init_img=None, init_mask=None, keep_mask=False):
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
    assert prompt is not None
    torch_gc()
    # start time after garbage collection (or before?)
    start_time = time.time()

    mem_mon = MemUsageMonitor('MemMon')
    mem_mon.start()

    if hasattr(model, "embedding_manager"):
        load_embeddings(fp)

    os.makedirs(outpath, exist_ok=True)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    comments = []

    prompt_matrix_parts = []
    if prompt_matrix:
        all_prompts = []
        prompt_matrix_parts = prompt.split("|")
        combination_count = 2 ** (len(prompt_matrix_parts) - 1)
        for combination_num in range(combination_count):
            current = prompt_matrix_parts[0]

            for n, text in enumerate(prompt_matrix_parts[1:]):
                if combination_num & (2 ** n) > 0:
                    current += ("" if text.strip().startswith(",") else ", ") + text

            all_prompts.append(current)

        n_iter = math.ceil(len(all_prompts) / batch_size)
        all_seeds = len(all_prompts) * [seed]

        print(f"Prompt matrix will create {len(all_prompts)} images using a total of {n_iter} batches.")
    else:

        if not opt.no_verify_input:
            try:
                check_prompt_length(prompt, comments)
            except:
                import traceback
                print("Error verifying input:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

        all_prompts = batch_size * n_iter * [prompt]
        all_seeds = [seed + x for x in range(len(all_prompts))]

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    output_images = []
    stats = []
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        init_data = func_init()
        tic = time.time()

        for n in range(n_iter):
            prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
            seeds = all_seeds[n * batch_size:(n + 1) * batch_size]

            uc = model.get_learned_conditioning(len(prompts) * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)

            # split the prompt if it has : for weighting
            # TODO for speed it might help to have this occur when all_prompts filled??
            subprompts,weights = split_weighted_subprompts(prompts[0])
            # get total weight for normalizing, this gets weird if large negative values used
            totalPromptWeight = sum(weights)

            # sub-prompt weighting used if more than 1
            if len(subprompts) > 1:
                c = torch.zeros_like(uc) # i dont know if this is correct.. but it works
                for i in range(0,len(subprompts)): # normalize each prompt and add it
                    weight = weights[i]
                    if normalize_prompt_weights:
                        weight = weight / totalPromptWeight
                    #print(f"{subprompts[i]} {weight*100.0}%")
                    # note if alpha negative, it functions same as torch.sub
                    c = torch.add(c,model.get_learned_conditioning(subprompts[i]), alpha=weight)
            else: # just behave like usual
                c = model.get_learned_conditioning(prompts)

            shape = [opt_C, height // opt_f, width // opt_f]

            # we manually generate all input noises because each one should have a specific seed
            x = create_random_tensors([opt_C, height // opt_f, width // opt_f], seeds=seeds)
            samples_ddim = func_sample(init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc, sampler_name=sampler_name)


            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                x_sample = x_sample.astype(np.uint8)

                if use_GFPGAN and GFPGAN is not None:
                    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(x_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
                    x_sample = restored_img[:,:,::-1]

                if use_RealESRGAN and RealESRGAN is not None:
                    if RealESRGAN.model.name != realesrgan_model_name:
                        try_loading_RealESRGAN(realesrgan_model_name)

                    output, img_mode = RealESRGAN.enhance(x_sample[:,:,::-1])
                    x_sample = output[:,:,::-1]

                image = Image.fromarray(x_sample)
                if init_mask:
                    #init_mask = init_mask if keep_mask else ImageOps.invert(init_mask)
                    init_mask = init_mask.filter(ImageFilter.GaussianBlur(3))
                    init_mask = init_mask.convert('L')
                    init_img = init_img.convert('RGB')
                    image = image.convert('RGB')

                    if use_RealESRGAN and RealESRGAN is not None:
                        if RealESRGAN.model.name != realesrgan_model_name:
                            try_loading_RealESRGAN(realesrgan_model_name)
                        output, img_mode = RealESRGAN.enhance(np.array(init_img, dtype=np.uint8))
                        init_img = Image.fromarray(output)
                        init_img = init_img.convert('RGB')

                        output, img_mode = RealESRGAN.enhance(np.array(init_mask, dtype=np.uint8))
                        init_mask = Image.fromarray(output)
                        init_mask = init_mask.convert('L')

                    image = Image.composite(init_img, image, init_mask)

                filename = f"{base_count:05}-{seeds[i]}_{prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.png"
                if not skip_save:
                    image.save(os.path.join(sample_path, filename))

                output_images.append(image)
                base_count += 1

        if (prompt_matrix or not skip_grid) and not do_not_save_grid:
            grid = image_grid(output_images, batch_size, round_down=prompt_matrix)

            if prompt_matrix:
                try:
                    grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
                except Exception:
                    import traceback
                    print("Error creating prompt_matrix text:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

                output_images.insert(0, grid)


            grid_file = f"grid-{grid_count:05}-{seed}_{prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.jpg"
            grid.save(os.path.join(outpath, grid_file), 'jpeg', quality=100, optimize=True)
            grid_count += 1
        toc = time.time()

    mem_max_used, mem_total = mem_mon.read_and_stop()
    time_diff = time.time()-start_time

    info = f"""
{prompt}
Steps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}{', GFPGAN' if use_GFPGAN and GFPGAN is not None else ''}{', '+realesrgan_model_name if use_RealESRGAN and RealESRGAN is not None else ''}{', Prompt Matrix Mode.' if prompt_matrix else ''}""".strip()
    stats = f'''
Took { round(time_diff, 2) }s total ({ round(time_diff/(len(all_prompts)),2) }s per image)
Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%'''

    for comment in comments:
        info += "\n\n" + comment

    #mem_mon.stop()
    #del mem_mon
    torch_gc()

    return output_images, seed, info, stats


def txt2img(prompt: str, ddim_steps: int, sampler_name: str, toggles: List[int], realesrgan_model_name: str, ddim_eta: float, n_iter: int,
            batch_size: int, cfg_scale: float, seed: Union[int, str, None], height: int, width: int, fp):
    outpath = opt.outdir_txt2img or opt.outdir or "outputs/txt2img-samples"
    err = False
    seed = seed_to_int(seed)

    prompt_matrix = 0 in toggles
    normalize_prompt_weights = 1 in toggles
    skip_save = 2 not in toggles
    skip_grid = 3 not in toggles
    use_GFPGAN = 4 in toggles
    use_RealESRGAN = 5 in toggles

    if sampler_name == 'PLMS':
        sampler = PLMSSampler(model)
    elif sampler_name == 'DDIM':
        sampler = DDIMSampler(model)
    elif sampler_name == 'k_dpm_2_a':
        sampler = KDiffusionSampler(model,'dpm_2_ancestral')
    elif sampler_name == 'k_dpm_2':
        sampler = KDiffusionSampler(model,'dpm_2')
    elif sampler_name == 'k_euler_a':
        sampler = KDiffusionSampler(model,'euler_ancestral')
    elif sampler_name == 'k_euler':
        sampler = KDiffusionSampler(model,'euler')
    elif sampler_name == 'k_heun':
        sampler = KDiffusionSampler(model,'heun')
    elif sampler_name == 'k_lms':
        sampler = KDiffusionSampler(model,'lms')
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    def init():
        pass

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=cfg_scale, unconditional_conditioning=unconditional_conditioning, eta=ddim_eta, x_T=x)
        return samples_ddim

    try:
        output_images, seed, info, stats = process_images(
            outpath=outpath,
            func_init=init,
            func_sample=sample,
            prompt=prompt,
            seed=seed,
            sampler_name=sampler_name,
            skip_save=skip_save,
            skip_grid=skip_grid,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=ddim_steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            prompt_matrix=prompt_matrix,
            use_GFPGAN=use_GFPGAN,
            use_RealESRGAN=use_RealESRGAN,
            realesrgan_model_name=realesrgan_model_name,
            fp=fp,
            normalize_prompt_weights=normalize_prompt_weights
        )

        del sampler

        return output_images, seed, info, stats
    except RuntimeError as e:
        err = e
        err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
        stats = err_msg
        return [], seed, 'err', stats
    finally:
        if err:
            crash(err, '!!Runtime error (txt2img)!!')


class Flagging(gr.FlaggingCallback):

    def setup(self, components, flagging_dir: str):
        pass

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None):
        import csv

        os.makedirs("log/images", exist_ok=True)

        # those must match the "txt2img" function !! + images, seed, comment, stats !! NOTE: changes to UI output must be reflected here too
        prompt, ddim_steps, sampler_name, toggles, ddim_eta, n_iter, batch_size, cfg_scale, seed, height, width, fp, images, seed, comment, stats = flag_data

        filenames = []

        with open("log/log.csv", "a", encoding="utf8", newline='') as file:
            import time
            import base64

            at_start = file.tell() == 0
            writer = csv.writer(file)
            if at_start:
                writer.writerow(["sep=,"])
                writer.writerow(["prompt", "seed", "width", "height", "sampler", "toggles", "n_iter", "n_samples", "cfg_scale", "steps", "filename"])

            filename_base = str(int(time.time() * 1000))
            for i, filedata in enumerate(images):
                filename = "log/images/"+filename_base + ("" if len(images) == 1 else "-"+str(i+1)) + ".png"

                if filedata.startswith("data:image/png;base64,"):
                    filedata = filedata[len("data:image/png;base64,"):]

                with open(filename, "wb") as imgfile:
                    imgfile.write(base64.decodebytes(filedata.encode('utf-8')))

                filenames.append(filename)

            writer.writerow([prompt, seed, width, height, sampler_name, toggles, n_iter, batch_size, cfg_scale, ddim_steps, filenames[0]])

        print("Logged:", filenames[0])


def img2img(prompt: str, image_editor_mode: str, cropped_image, image_with_mask, mask_mode: str, ddim_steps: int, sampler_name: str,
            toggles: List[int], realesrgan_model_name: str, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float,
            seed: int, height: int, width: int, resize_mode: int, fp):
    outpath = opt.outdir_img2img or opt.outdir or "outputs/img2img-samples"
    err = False
    seed = seed_to_int(seed)

    prompt_matrix = 0 in toggles
    normalize_prompt_weights = 1 in toggles
    loopback = 2 in toggles
    random_seed_loopback = 3 in toggles
    skip_save = 4 not in toggles
    skip_grid = 5 not in toggles
    use_GFPGAN = 6 in toggles
    use_RealESRGAN = 7 in toggles

    if sampler_name == 'DDIM':
        sampler = DDIMSampler(model)
    elif sampler_name == 'k_dpm_2_a':
        sampler = KDiffusionSampler(model,'dpm_2_ancestral')
    elif sampler_name == 'k_dpm_2':
        sampler = KDiffusionSampler(model,'dpm_2')
    elif sampler_name == 'k_euler_a':
        sampler = KDiffusionSampler(model,'euler_ancestral')
    elif sampler_name == 'k_euler':
        sampler = KDiffusionSampler(model,'euler')
    elif sampler_name == 'k_heun':
        sampler = KDiffusionSampler(model,'heun')
    elif sampler_name == 'k_lms':
        sampler = KDiffusionSampler(model,'lms')
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    if image_editor_mode == 'Mask':
        init_img = image_with_mask["image"]
        init_img = init_img.convert("RGB")
        init_img = resize_image(resize_mode, init_img, width, height)
        init_mask = image_with_mask["mask"]
        init_mask = init_mask.convert("RGB")
        init_mask = resize_image(resize_mode, init_mask, width, height)
        keep_mask = mask_mode == 0
        init_mask = init_mask if keep_mask else ImageOps.invert(init_mask)
    else:
        init_img = cropped_image
        init_mask = None
        keep_mask = False

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(denoising_strength * ddim_steps)

    def init():
        image = init_img.convert("RGB")
        image = resize_image(resize_mode, image, width, height)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        init_image = 2. * image - 1.
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        return init_latent,

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        if sampler_name != 'DDIM':
            x0, = init_data

            sigmas = sampler.model_wrap.get_sigmas(ddim_steps)
            noise = x * sigmas[ddim_steps - t_enc - 1]

            xi = x0 + noise
            sigma_sched = sigmas[ddim_steps - t_enc - 1:]
            model_wrap_cfg = CFGDenoiser(sampler.model_wrap)
            samples_ddim = K.sampling.sample_lms(model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': cfg_scale}, disable=False)
        else:
            x0, = init_data
            sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
            z_enc = sampler.stochastic_encode(x0, torch.tensor([t_enc]*batch_size).to(device))
                                # decode it
            samples_ddim = sampler.decode(z_enc, conditioning, t_enc,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=unconditional_conditioning,)
        return samples_ddim


    try:
        if loopback:
            output_images, info = None, None
            history = []
            initial_seed = None

            for i in range(n_iter):
                output_images, seed, info, stats = process_images(
                    outpath=outpath,
                    func_init=init,
                    func_sample=sample,
                    prompt=prompt,
                    seed=seed,
                    sampler_name=sampler_name,
                    skip_save=skip_save,
                    skip_grid=skip_grid,
                    batch_size=1,
                    n_iter=1,
                    steps=ddim_steps,
                    cfg_scale=cfg_scale,
                    width=width,
                    height=height,
                    prompt_matrix=prompt_matrix,
                    use_GFPGAN=use_GFPGAN,
                    use_RealESRGAN=False, # Forcefully disable upscaling when using loopback
                    realesrgan_model_name=realesrgan_model_name,
                    fp=fp,
                    do_not_save_grid=True,
                    normalize_prompt_weights=normalize_prompt_weights,
                    init_img=init_img,
                    init_mask=init_mask,
                    keep_mask=keep_mask
                )

                if initial_seed is None:
                    initial_seed = seed

                init_img = output_images[0]
                if not random_seed_loopback:
                    seed = seed + 1
                else:
                    seed = seed_to_int(None)
                denoising_strength = max(denoising_strength * 0.95, 0.1)
                history.append(init_img)

            if not skip_grid:
                grid_count = len(os.listdir(outpath)) - 1
                grid = image_grid(history, batch_size, force_n_rows=1)
                grid_file = f"grid-{grid_count:05}-{seed}_{prompt.replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.jpg"
                grid.save(os.path.join(outpath, grid_file), 'jpeg', quality=100, optimize=True)


            output_images = history
            seed = initial_seed

        else:
            output_images, seed, info, stats = process_images(
                outpath=outpath,
                func_init=init,
                func_sample=sample,
                prompt=prompt,
                seed=seed,
                sampler_name=sampler_name,
                skip_save=skip_save,
                skip_grid=skip_grid,
                batch_size=batch_size,
                n_iter=n_iter,
                steps=ddim_steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                prompt_matrix=prompt_matrix,
                use_GFPGAN=use_GFPGAN,
                use_RealESRGAN=use_RealESRGAN,
                realesrgan_model_name=realesrgan_model_name,
                fp=fp,
                normalize_prompt_weights=normalize_prompt_weights,
                init_img=init_img,
                init_mask=init_mask,
                keep_mask=keep_mask
            )

        del sampler

        return output_images, seed, info, stats
    except RuntimeError as e:
        err = e
        err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
        stats = err_msg
        return [], seed, 'err', stats
    finally:
        if err:
            crash(err, '!!Runtime error (img2img)!!')

# grabs all text up to the first occurrence of ':' as sub-prompt
# takes the value following ':' as weight
# if ':' has no value defined, defaults to 1.0
# repeats until no text remaining
# TODO this could probably be done with less code
def split_weighted_subprompts(text):
    print(text)
    remaining = len(text)
    prompts = []
    weights = []
    while remaining > 0:
        if ":" in text:
            idx = text.index(":") # first occurrence from start
            # grab up to index as sub-prompt
            prompt = text[:idx]
            remaining -= idx
            # remove from main text
            text = text[idx+1:]
            # find value for weight, assume it is followed by a space or comma
            idx = len(text) # default is read to end of text
            if " " in text:
                idx = min(idx,text.index(" ")) # want the closer idx
            if "," in text:
                idx = min(idx,text.index(",")) # want the closer idx
            if idx != 0:
                try:
                    weight = float(text[:idx])
                except: # couldn't treat as float
                    print(f"Warning: '{text[:idx]}' is not a value, are you missing a space or comma after a value?")
                    weight = 1.0
            else: # no value found
                weight = 1.0
            # remove from main text
            remaining -= idx
            text = text[idx+1:]
            # append the sub-prompt and its weight
            prompts.append(prompt)
            weights.append(weight)
        else: # no : found
            if len(text) > 0: # there is still text though
                # take remainder as weight 1
                prompts.append(text)
                weights.append(1.0)
            remaining = 0
    return prompts, weights

def run_GFPGAN(image, strength):
    image = image.convert("RGB")

    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(np.array(image, dtype=np.uint8), has_aligned=False, only_center_face=False, paste_back=True)
    res = Image.fromarray(restored_img)

    if strength < 1.0:
        res = Image.blend(image, res, strength)

    return res

def run_RealESRGAN(image, model_name: str):
    if RealESRGAN.model.name != model_name:
            try_loading_RealESRGAN(model_name)

    image = image.convert("RGB")

    output, img_mode = RealESRGAN.enhance(np.array(image, dtype=np.uint8))
    res = Image.fromarray(output)

    return res

css = "" if opt.no_progressbar_hiding else css_hide_progressbar
css = css + '[data-testid="image"] {min-height: 512px !important}'

if opt.defaults is not None and os.path.isfile(opt.defaults):
    try:
        with open(opt.defaults, "r", encoding="utf8") as f:
            user_defaults = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        print(f"Error loading defaults file {opt.defaults}:", e)
        print("Falling back to program defaults.")
        user_defaults = {}
else:
    user_defaults = {}

# make sure these indicies line up at the top of txt2img()
txt2img_toggles = [
    'Create prompt matrix (separate multiple prompts using |, and get all combinations of them)',
    'Normalize Prompt Weights (ensure sum of weights add up to 1.0)',
    'Save individual images',
    'Save grid',
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
        return [gr.update(visible=False), gr.update(visible=True)]
    return [gr.update(visible=True), gr.update(visible=False)]

def update_image_mask(cropped_image, resize_mode, width, height):
    resized_cropped_image = resize_image(resize_mode, cropped_image, width, height) if cropped_image else None
    return gr.update(value=resized_cropped_image)

def copy_img_to_input(selected=1, imgs = []):
    try:
        idx = int(0 if selected - 1 < 0 else selected - 1)
        image_data = re.sub('^data:image/.+;base64,', '', imgs[idx])
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        return [processed_image, processed_image]
    except IndexError:
        return [None, None]

with gr.Blocks(css=css) as demo:
    with gr.Tabs():
        with gr.TabItem("Stable Diffusion Text-to-Image Unified"):
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    gr.Markdown("Generate images from text with Stable Diffusion")
                    txt2img_prompt = gr.Textbox(label="Prompt", placeholder="A corgi wearing a top hat as an oil painting.", lines=1, value=txt2img_defaults['prompt'])
                    txt2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=txt2img_defaults['ddim_steps'])
                    txt2img_sampling = gr.Radio(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=txt2img_defaults['sampler_name'])
                    txt2img_toggles = gr.CheckboxGroup(label='', choices=txt2img_toggles, value=txt2img_toggle_defaults, type="index")
                    txt2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B'], value='RealESRGAN_x4plus', visible=RealESRGAN is not None) # TODO: Feels like I shouldnt slot it in here.
                    txt2img_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=False)
                    txt2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1, label='Batch count (how many batches of images to generate)', value=1)
                    txt2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=1)
                    txt2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=7.5)
                    txt2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, value="")
                    txt2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                    txt2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                    txt2img_embeddings = gr.File(label = "Embeddings file for textual inversion", visible=hasattr(model, "embedding_manager"))
                    txt2img_btn = gr.Button("Generate")
                with gr.Column():
                    output_txt2img_gallery = gr.Gallery(label="Images")
                    output_txt2img_select_image = gr.Number(label='Select image number from results for copying', value=1, precision=None)
                    output_txt2img_copy_to_input_btn = gr.Button("Copy selected image to img2img input")
                    output_txt2img_seed = gr.Number(label='Seed')
                    output_txt2img_params = gr.Textbox(label="Copy-paste generation parameters")
                    output_txt2img_stats = gr.HTML(label='Stats')

            txt2img_btn.click(
                txt2img,
                [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_toggles, txt2img_realesrgan_model_name, txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfg, txt2img_seed, txt2img_height, txt2img_width, txt2img_embeddings],
                [output_txt2img_gallery, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]
            )

        with gr.TabItem("Stable Diffusion Image-to-Image Unified"):
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    gr.Markdown("Generate images from images with Stable Diffusion")
                    img2img_prompt = gr.Textbox(label="Prompt", placeholder="A fantasy landscape, trending on artstation.", lines=1, value=img2img_defaults['prompt'])
                    img2img_image_editor_mode = gr.Radio(choices=["Mask", "Crop"], label="Image Editor Mode", value="Crop")
                    gr.Markdown("The masking/cropping is very temperamental. It may take some time for the image to show when switching from Crop to Mask. If it doesn't work try switching modes again, switch tabs, clear the image or reload.")
                    img2img_image_editor = gr.Image(value=sample_img2img, source="upload", interactive=True, type="pil", tool="select")
                    img2img_image_mask = gr.Image(value=sample_img2img, source="upload", interactive=True, type="pil", tool="sketch", visible=False)
                    img2img_mask = gr.Radio(choices=["Keep masked area", "Regenerate only masked area"], label="Mask Mode", type="index", value=img2img_mask_modes[img2img_defaults['mask_mode']])
                    img2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=img2img_defaults['ddim_steps'])
                    img2img_sampling = gr.Radio(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=img2img_defaults['sampler_name'])
                    img2img_toggles = gr.CheckboxGroup(label='', choices=img2img_toggles, value=img2img_toggle_defaults, type="index")
                    img2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B'], value='RealESRGAN_x4plus', visible=RealESRGAN is not None) # TODO: Feels like I shouldnt slot it in here.
                    img2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1, label='Batch count (how many batches of images to generate)', value=1)
                    img2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=1)
                    img2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=5.0)
                    img2img_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength', value=0.75)
                    img2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, value="")
                    img2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                    img2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                    img2img_resize = gr.Radio(label="Resize mode", choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value="Just resize")
                    img2img_embeddings = gr.File(label = "Embeddings file for textual inversion", visible=hasattr(model, "embedding_manager"))
                    img2img_btn = gr.Button("Generate")
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
                [img2img_image_editor, img2img_image_mask]
            )

            img2img_image_editor.edit(
                update_image_mask,
                [img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                img2img_image_mask
            )

            output_img2img_copy_to_input_btn.click(
                copy_img_to_input,
                [output_img2img_select_image, output_img2img_gallery],
                [img2img_image_editor, img2img_image_mask]
            )

            img2img_btn.click(
                img2img,
                [img2img_prompt, img2img_image_editor_mode, img2img_image_editor, img2img_image_mask, img2img_mask, img2img_steps, img2img_sampling, img2img_toggles, img2img_realesrgan_model_name, img2img_batch_count, img2img_batch_size, img2img_cfg, img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize, img2img_embeddings],
                [output_img2img_gallery, output_img2img_seed, output_img2img_params, output_img2img_stats]
            )
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
                        gfpgan_btn = gr.Button("Generate")
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

    output_txt2img_copy_to_input_btn.click(
        copy_img_to_input,
        [output_txt2img_select_image, output_txt2img_gallery],
        [img2img_image_editor, img2img_image_mask]
    )

demo.queue(concurrency_count=1)

class ServerLauncher(threading.Thread):
    def __init__(self, demo):
        threading.Thread.__init__(self)
        self.name = 'Gradio Server Thread'
        self.demo = demo

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.demo.launch(show_error=True, server_name='0.0.0.0')

    def stop(self):
        self.demo.close() # this tends to hang

if opt.cli is None:
    server_thread = ServerLauncher(demo)
    server_thread.start()

    try:
        while server_thread.is_alive():
            time.sleep(60)
    except (KeyboardInterrupt, OSError) as e:
        crash(e, 'Shutting down...')
else:
    with open(opt.cli, "r", encoding="utf8") as f:
        kwargs = yaml.safe_load(f)
    target = kwargs.pop("target")
    if target == "txt2img":
        target_func = txt2img
    elif target == "img2img":
        target_func = img2img
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown target: {target}")
    kwargs["fp"] = None
    output_images, seed, info, stats = target_func(**kwargs)
    print(f"Seed: {seed}")
    print(info)
    print(stats)
