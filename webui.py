import argparse, os, sys, glob, re

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default=None)
parser.add_argument("--outdir_txt2img", type=str, nargs="?", help="dir to write txt2img results to (overrides --outdir)", default=None)
parser.add_argument("--outdir_img2img", type=str, nargs="?", help="dir to write img2img results to (overrides --outdir)", default=None)
parser.add_argument("--save-metadata", action='store_true', help="Whether to embed the generation parameters in the sample images", default=False)
parser.add_argument("--skip-grid", action='store_true', help="do not save a grid, only individual samples. Helpful when evaluating lots of samples", default=False)
parser.add_argument("--skip-save", action='store_true', help="do not save indiviual samples. For speed measurements.", default=False)
parser.add_argument("--grid-format", type=str, help="png for lossless png files; jpg:quality for lossy jpeg; webp:quality for lossy webp, or webp:-compression for lossless webp", default="jpg:95")
parser.add_argument("--n_rows", type=int, default=-1, help="rows in the grid; use -1 for autodetect and 0 for n_rows to be same as batch_size (default: -1)",)
parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model",)
parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="path to checkpoint of model",)
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--optimized", action='store_true', help="load the model onto the device piecemeal instead of all at once to reduce VRAM usage at the cost of performance")
parser.add_argument("--optimized-turbo", action='store_true', help="alternative optimization mode that does not save as much VRAM but runs siginificantly faster")
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN')) # i disagree with where you're putting it but since all guidefags are doing it this way, there you go
parser.add_argument("--realesrgan-dir", type=str, help="RealESRGAN directory", default=('./src/realesrgan' if os.path.exists('./src/realesrgan') else './RealESRGAN'))
parser.add_argument("--realesrgan-model", type=str, help="Upscaling model for RealESRGAN", default=('RealESRGAN_x4plus'))
parser.add_argument("--no-verify-input", action='store_true', help="do not verify input to check if it's too long", default=False)
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats", default=False)
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware accleration in browser)", default=False)
parser.add_argument("--share", action='store_true', help="Should share your server on gradio.app, this allows you to use the UI from your mobile app", default=False)
parser.add_argument("--share-password", type=str, help="Sharing is open by default, use this to set a password. Username: webui", default=None)
parser.add_argument("--defaults", type=str, help="path to configuration file providing UI defaults, uses same format as cli parameter", default='configs/webui/webui.yaml')
parser.add_argument("--gpu", type=int, help="choose which GPU to use if you have multiple", default=int(os.environ.get('CUDA_VISIBLE_DEVICES', 0)))
parser.add_argument("--extra-models-cpu", action='store_true', help="run extra models (GFGPAN/ESRGAN) on cpu", default=False)
parser.add_argument("--esrgan-cpu", action='store_true', help="run ESRGAN on cpu", default=False)
parser.add_argument("--extern-upscaler", action='store_true', help="run vulkan Upscalers", default=False)
parser.add_argument("--gfpgan-cpu", action='store_true', help="run GFPGAN on cpu", default=False)
parser.add_argument("--cli", type=str, help="don't launch web server, take Python function kwargs from this file.", default=None)
opt = parser.parse_args()

# this should force GFPGAN and RealESRGAN onto the selected gpu as well
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

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
import glob
from typing import List, Union
from pathlib import Path

from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from itertools import islice
from omegaconf import OmegaConf
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from PIL.PngImagePlugin import PngInfo
from io import BytesIO
import base64
import re
from torch import autocast
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import subprocess
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

GFPGAN_dir = opt.gfpgan_dir
RealESRGAN_dir = opt.realesrgan_dir

if opt.optimized_turbo:
    opt.optimized = True

# should probably be moved to a settings menu in the UI at some point
grid_format = [s.lower() for s in opt.grid_format.split(':')]
grid_lossless = False
grid_quality = 100
if grid_format[0] == 'png':
    grid_ext = 'png'
    grid_format = 'png'
elif grid_format[0] in ['jpg', 'jpeg']:
    grid_quality = int(grid_format[1]) if len(grid_format) > 1 else 100
    grid_ext = 'jpg'
    grid_format = 'jpeg'
elif grid_format[0] == 'webp':
    grid_quality = int(grid_format[1]) if len(grid_format) > 1 else 100
    grid_ext = 'webp'
    grid_format = 'webp'
    if grid_quality < 0: # e.g. webp:-100 for lossless mode
        grid_lossless = True
        grid_quality = abs(grid_quality)


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

def load_sd_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

def crash(e, s):
    pass

class MemUsageMonitor(threading.Thread):
    stop_flag = False
    max_usage = 0
    total = -1

    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        try:
            pynvml.nvmlInit()
        except:
            print(f"[{self.name}] Unable to initialize NVIDIA management. No memory stats. \n")
            return
        print(f"[{self.name}] Recording max memory usage...\n")
        handle = pynvml.nvmlDeviceGetHandleByIndex(opt.gpu)
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
    def get_sampler_name(self):
        return self.schedule
    def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T):
        sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)

        samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale}, disable=False)

        return samples_ddim, None


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
    instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
    if opt.gfpgan_cpu or opt.extra_models_cpu:
        instance.device = torch.device('cpu')
    else:
        instance.device = torch.device('cuda') # another way to set gpu device
    return instance

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

    if opt.esrgan_cpu or opt.extra_models_cpu:
        instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=False)
        instance.model.name = model_name
        instance.device = torch.device('cpu')
        instance.device = torch.device('cpu')
        instance.model.to('cpu')
    else:
        instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=not opt.no_half)
        instance.model.name = model_name
        instance.device = torch.device('cuda') # another way to set gpu device

    return instance

GFPGAN = None
def DynamicLoad_GFPGAN():
    if os.path.exists(GFPGAN_dir):
        try:
            global GFPGAN
            GFPGAN = load_GFPGAN()
            print("Loaded GFPGAN")
        except Exception:
            import traceback
            print("Error loading GFPGAN:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

def DynamicUnload_GFPGAN():
    global GFPGAN
    del GFPGAN
    GFPGAN = None

RealESRGAN = None
def DynamicLoad_RealESRGAN(model_name: str):
    global RealESRGAN
    if os.path.exists(RealESRGAN_dir):
        try:
            RealESRGAN = load_RealESRGAN(model_name) # TODO: Should try to load both models before giving up
            print("Loaded RealESRGAN with model "+RealESRGAN.model.name)
        except Exception:
            import traceback
            print("Error loading RealESRGAN:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

def DynamicUnload_RealESRGAN():
    global RealESRGAN
    del RealESRGAN
    RealESRGAN = None
#try_loading_RealESRGAN('RealESRGAN_x4plus')

if opt.optimized:
    sd = load_sd_from_config(opt.ckpt)
    li, lo = [], []
    for key, v_ in sd.items():
        sp = key.split('.')
        if(sp[0]) == 'model':
            if('input_blocks' in sp):
                li.append(key)
            elif('middle_block' in sp):
                li.append(key)
            elif('time_embed' in sp):
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd['model1.' + key[6:]] = sd.pop(key)
    for key in lo:
        sd['model2.' + key[6:]] = sd.pop(key)

    config = OmegaConf.load("optimizedSD/v1-inference.yaml")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    model.turbo = opt.optimized_turbo

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.cond_stage_model.device = device
    modelCS.eval()

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()

    del sd

    if not opt.no_half:
        model = model.half()
        modelCS = modelCS.half()
        modelFS = modelFS.half()
else:
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = (model if opt.no_half else model.half()).to(device)

def load_embeddings(fp):
    if fp is not None and hasattr(model, "embedding_manager"):
        model.embedding_manager.load(fp.name)


def get_font(fontsize):
    fonts = ["arial.ttf", "DejaVuSans.ttf"]
    for font_name in fonts:
        try:
            return ImageFont.truetype(font_name, fontsize)
        except OSError:
           pass

    # ImageFont.load_default() is practically unusable as it only supports
    # latin1, so raise an exception instead if no usable font was found
    raise Exception(f"No usable font found (tried {', '.join(fonts)})")

def image_grid(imgs, batch_size, force_n_rows=None, captions=None):
    if force_n_rows is not None:
        rows = force_n_rows
    elif opt.n_rows > 0:
        rows = opt.n_rows
    elif opt.n_rows == 0:
        rows = batch_size
    else:
        rows = math.sqrt(len(imgs))
        rows = round(rows)

    cols = math.ceil(len(imgs) / rows)

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

    fnt = get_font(30)

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
        if captions:
            d = ImageDraw.Draw( grid )
            size = d.textbbox( (0,0), captions[i], font=fnt, stroke_width=2, align="center" )
            d.multiline_text((i % cols * w + w/2, i // cols * h + h - size[3]), captions[i], font=fnt, fill=(255,255,255), stroke_width=2, stroke_fill=(0,0,0), anchor="mm", align="center")

    return grid

def seed_to_int(s):
    if type(s) is int:
        return s
    if s is None or s == '':
        return random.randint(0, 2**32 - 1)
    n = abs(int(s) if s.isdigit() else random.Random(s).randint(0, 2**32 - 1))
    while n >= 2**32:
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
    fnt = get_font(fontsize)
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

    tokenizer = (model if not opt.optimized else modelCS).cond_stage_model.tokenizer
    max_length = (model if not opt.optimized else modelCS).cond_stage_model.max_length

    info = (model if not opt.optimized else modelCS).cond_stage_model.tokenizer([prompt], truncation=True, max_length=max_length, return_overflowing_tokens=True, padding="max_length", return_tensors="pt")
    ovf = info['overflowing_tokens'][0]
    overflowing_count = ovf.shape[0]
    if overflowing_count == 0:
        return

    vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    overflowing_words = [vocab.get(int(x), "") for x in ovf]
    overflowing_text = tokenizer.convert_tokens_to_string(''.join(overflowing_words))

    comments.append(f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")

def save_sample(image, sample_path_i, filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode):
    filename_i = os.path.join(sample_path_i, filename)
    if not jpg_sample:
        if opt.save_metadata:
            metadata = PngInfo()
            metadata.add_text("SD:prompt", prompts[i])
            metadata.add_text("SD:seed", str(seeds[i]))
            metadata.add_text("SD:width", str(width))
            metadata.add_text("SD:height", str(height))
            metadata.add_text("SD:steps", str(steps))
            metadata.add_text("SD:cfg_scale", str(cfg_scale))
            metadata.add_text("SD:normalize_prompt_weights", str(normalize_prompt_weights))
            metadata.add_text("SD:GFPGAN", str(use_GFPGAN and GFPGAN is not None))
            image.save(f"{filename_i}.png", pnginfo=metadata)
        else:
            image.save(f"{filename_i}.png")
    else:
        image.save(f"{filename_i}.jpg", 'jpeg', quality=100, optimize=True)
    if write_info_files:
        # toggles differ for txt2img vs. img2img:
        offset = 0 if init_img is None else 2
        toggles = []
        if prompt_matrix:
            toggles.append(0)
        if normalize_prompt_weights:
            toggles.append(1)
        if init_img is not None:
            if uses_loopback:
                toggles.append(2)
            if uses_random_seed_loopback:
                toggles.append(3)
        if not skip_save:
            toggles.append(2 + offset)
        if not skip_grid:
            toggles.append(3 + offset)
        if sort_samples:
            toggles.append(4 + offset)
        if write_info_files:
            toggles.append(5 + offset)
        if use_GFPGAN:
            toggles.append(6 + offset)
        info_dict = dict(
            target="txt2img" if init_img is None else "img2img",
            prompt=prompts[i], ddim_steps=steps, toggles=toggles, sampler_name=sampler_name,
            ddim_eta=ddim_eta, n_iter=n_iter, batch_size=batch_size, cfg_scale=cfg_scale,
            seed=seeds[i], width=width, height=height
        )
        if init_img is not None:
            # Not yet any use for these, but they bloat up the files:
            #info_dict["init_img"] = init_img
            #info_dict["init_mask"] = init_mask
            info_dict["denoising_strength"] = denoising_strength
            info_dict["resize_mode"] = resize_mode
        with open(f"{filename_i}.yaml", "w", encoding="utf8") as f:
            yaml.dump(info_dict, f, allow_unicode=True)


def get_next_sequence_number(path, prefix=''):
    """
    Determines and returns the next sequence number to use when saving an
    image in the specified directory.

    If a prefix is given, only consider files whose names start with that
    prefix, and strip the prefix from filenames before extracting their
    sequence number.

    The sequence starts at 0.
    """
    result = -1
    for p in Path(path).iterdir():
        if p.name.endswith(('.png', '.jpg')) and p.name.startswith(prefix):
            tmp = p.name[len(prefix):]
            try:
                result = max(int(tmp.split('-')[0]), result)
            except ValueError:
                pass
    return result + 1

def oxlamon_matrix(prompt, seed, batch_size):
    pattern = re.compile(r'(,\s){2,}')

    class PromptItem:
        def __init__(self, text, parts, item):
            self.text = text
            self.parts = parts
            if item:
                self.parts.append( item )

    def clean(txt):
        return re.sub(pattern, ', ', txt)

    def repliter( txt ):
        for data in re.finditer( ".*?\\((.*?)\\).*", txt ):
            if data:
                r = data.span(1)
                for item in data.group(1).split("|"):
                    yield (clean(txt[:r[0]-1] + item.strip() + txt[r[1]+1:]), item.strip())
            break

    def iterlist( items ):
        outitems = []
        for item in items:
            for newitem, newpart in repliter(item.text):
                outitems.append( PromptItem(newitem, item.parts.copy(), newpart) )

        return outitems

    def getmatrix( prompt ):
        dataitems = [ PromptItem( prompt[1:].strip(), [], None ) ]
        while True:
            newdataitems = iterlist( dataitems )
            if len( newdataitems ) == 0:
                return dataitems
            dataitems = newdataitems

    def classToArrays( items ):
        texts = []
        parts = []

        for item in items:
            texts.append( item.text )
            parts.append( "\n".join(item.parts) )
        return texts, parts

    all_prompts, prompt_matrix_parts = classToArrays(getmatrix( prompt ))
    n_iter = math.ceil(len(all_prompts) / batch_size)
    all_seeds = len(all_prompts) * [seed]
    return all_seeds, n_iter, prompt_matrix_parts, all_prompts, None


def process_images(
        outpath, func_init, func_sample, prompt, seed, sampler_name, skip_grid, skip_save, batch_size,
        n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, use_RealESRGAN,use_GoBIG,gstrength,gsteps, upmodel_type, realesrgan_model_name, esrgan_scale,
        fp, ddim_eta=0.0, do_not_save_grid=False, normalize_prompt_weights=True, init_img=None, init_mask=None,
        keep_mask=False, mask_blur_strength=3, denoising_strength=0.75, resize_mode=None, uses_loopback=False,
        uses_random_seed_loopback=False, sort_samples=True, write_info_files=True, jpg_sample=False):
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

    if not ("|" in prompt) and prompt.startswith("@"):
        prompt = prompt[1:]

    comments = []

    prompt_matrix_parts = []
    if prompt_matrix:
        if prompt.startswith("@"):
            all_seeds, n_iter, prompt_matrix_parts, all_prompts, frows = oxlamon_matrix(prompt, seed, batch_size)
        else:
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
    output_seeds = []
    stats = []
    output_name = []
    output_pixels = []
    tic = time.time()
    for n in range(n_iter):
        with torch.no_grad(), precision_scope("cuda"), (model.ema_scope() if not opt.optimized else nullcontext()):
            init_data = func_init()
            prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
            seeds = all_seeds[n * batch_size:(n + 1) * batch_size]

            if opt.optimized:
                modelCS.to(device)
            uc = (model if not opt.optimized else modelCS).get_learned_conditioning(len(prompts) * [""])
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
                    c = torch.add(c, (model if not opt.optimized else modelCS).get_learned_conditioning(subprompts[i]), alpha=weight)
            else: # just behave like usual
                c = (model if not opt.optimized else modelCS).get_learned_conditioning(prompts)

            shape = [opt_C, height // opt_f, width // opt_f]

            if opt.optimized:
                mem = torch.cuda.memory_allocated()/1e6
                modelCS.to("cpu")
                while(torch.cuda.memory_allocated()/1e6 >= mem):
                    time.sleep(1)

            # we manually generate all input noises because each one should have a specific seed
            x = create_random_tensors([opt_C, height // opt_f, width // opt_f], seeds=seeds)
            samples_ddim = func_sample(init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc, sampler_name=sampler_name)
            if opt.optimized:
                modelFS.to(device)


            x_samples_ddim = (model if not opt.optimized else modelFS).decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            for i, x_sample in enumerate(x_samples_ddim):
                sanitized_prompt = prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})
                if sort_samples:
                    sanitized_prompt = sanitized_prompt[:128] #200 is too long
                    sample_path_i = os.path.join(sample_path, sanitized_prompt)
                    os.makedirs(sample_path_i, exist_ok=True)
                    base_count = get_next_sequence_number(sample_path_i)
                    filename = f"{base_count:05}-{steps}_{sampler_name}_{seeds[i]}"
                else:
                    sample_path_i = sample_path
                    base_count = get_next_sequence_number(sample_path_i)
                    sanitized_prompt = sanitized_prompt
                    filename = f"{base_count:05}-{seeds[i]}_{sanitized_prompt}"[:128] #same as before
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                x_sample = x_sample.astype(np.uint8)
                if use_GFPGAN and init_img is not None:
                    DynamicLoad_GFPGAN()
                    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(x_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
                    DynamicUnload_GFPGAN()
                    x_sample = restored_img[:,:,::-1]
                    image = Image.fromarray(x_sample)
                    filename = filename + '-gfpgan'
                    save_sample(image, sample_path_i, filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img, uses_loopback, uses_random_seed_loopback, skip_save,
skip_grid, sort_samples, sampler_name, ddim_eta, n_iter, batch_size, i, denoising_strength, resize_mode)


                image = Image.fromarray(x_sample)

                if init_mask:
                    #init_mask = init_mask if keep_mask else ImageOps.invert(init_mask)
                    init_mask = init_mask.filter(ImageFilter.GaussianBlur(mask_blur_strength))
                    init_mask = init_mask.convert('L')
                    init_img = init_img.convert('RGB')
                    image = image.convert('RGB')


                    if use_RealESRGAN and init_img is not None:

                        #DynamicLoad_RealESRGAN(realesrgan_model_name)
                        #if opt.extern_upscaler:
                        init_img = run_RealESRGAN(init_image,upmodel_type, realesrgan_model_name, True, 'RGB',esrgan_scale)
                        init_mask = run_RealESRGAN(init_mask,upmodel_type,realesrgan_model_name, True, 'L',esrgan_scale)
                        #else:
                        #    output, img_mode = RealESRGAN.enhance(np.array(init_img, dtype=np.uint8),outscale=esrgan_scale)
                        #    init_img = Image.fromarray(output)
                        #    init_img = init_img.convert('RGB')

                        #   output, img_mode = RealESRGAN.enhance(np.array(init_mask, dtype=np.uint8),outscale=esrgan_scale)
                        #    init_mask = Image.fromarray(output)
                        #    init_mask = init_mask.convert('L')
                        #DynamicUnload_RealESRGAN()
                    image = Image.composite(init_img, image, init_mask)


                if use_GoBIG:
                    original_sample = x_sample
                    original_filename = filename
                    def addalpha(im, mask):
                        imr, img, imb, ima = im.split()
                        mmr, mmg, mmb, mma = mask.split()
                        im = Image.merge('RGBA', [imr, img, imb, mma])  # we want the RGB from the original, but the transparency from the mask
                        return(im)
                    def grid_merge(source, slices):
                        source.convert("RGBA")
                        for slice, posx, posy in slices: # go in reverse to get proper stacking
                            source.alpha_composite(slice, (posx, posy))
                        return source
                    def grid_slice(source, overlap, og_size, maximize=False):
                        def grid_coords(target, original, overlap):
                            #generate a list of coordinate tuples for our sections, in order of how they'll be rendered
                            #target should be the size for the gobig result, original is the size of each chunk being rendered
                            center = []
                            target_x, target_y = target
                            center_x = int(target_x / 2)
                            center_y = int(target_y / 2)
                            original_x, original_y = original
                            x = center_x - int(original_x / 2)
                            y = center_y - int(original_y / 2)
                            center.append((x,y)) #center chunk
                            uy = y #up
                            uy_list = []
                            dy = y #down
                            dy_list = []
                            lx = x #left
                            lx_list = []
                            rx = x #right
                            rx_list = []
                            while uy > 0: #center row vertical up
                                uy = uy - original_y + overlap
                                uy_list.append((lx, uy))
                            while (dy + original_y) <= target_y: #center row vertical down
                                dy = dy + original_y - overlap
                                dy_list.append((rx, dy))
                            while lx > 0:
                                lx = lx - original_x + overlap
                                lx_list.append((lx, y))
                                uy = y
                                while uy > 0:
                                    uy = uy - original_y + overlap
                                    uy_list.append((lx, uy))
                                dy = y
                                while (dy + original_y) <= target_y:
                                    dy = dy + original_y - overlap
                                    dy_list.append((lx, dy))
                            while (rx + original_x) <= target_x:
                                rx = rx + original_x - overlap
                                rx_list.append((rx, y))
                                uy = y
                                while uy > 0:
                                    uy = uy - original_y + overlap
                                    uy_list.append((rx, uy))
                                dy = y
                                while (dy + original_y) <= target_y:
                                    dy = dy + original_y - overlap
                                    dy_list.append((rx, dy))
                            # calculate a new size that will fill the canvas, which will be optionally used in grid_slice and go_big
                            last_coordx, last_coordy = dy_list[-1:][0]
                            render_edgey = last_coordy + original_y # outer bottom edge of the render canvas
                            render_edgex = last_coordx + original_x # outer side edge of the render canvas
                            scalarx = render_edgex / target_x
                            scalary = render_edgey / target_y
                            if scalarx <= scalary:
                                new_edgex = int(target_x * scalarx)
                                new_edgey = int(target_y * scalarx)
                            else:
                                new_edgex = int(target_x * scalary)
                                new_edgey = int(target_y * scalary)
                            # now put all the chunks into one master list of coordinates (essentially reverse of how we calculated them so that the central slices will be on top)
                            result = []
                            for coords in dy_list[::-1]:
                                result.append(coords)
                            for coords in uy_list[::-1]:
                                result.append(coords)
                            for coords in rx_list[::-1]:
                                result.append(coords)
                            for coords in lx_list[::-1]:
                                result.append(coords)
                            result.append(center[0])
                            return result, (new_edgex, new_edgey)
                        def get_resampling_mode():
                            try:
                                from PIL import __version__, Image
                                major_ver = int(__version__.split('.')[0])
                                if major_ver >= 9:
                                    return Image.Resampling.LANCZOS
                                else:
                                    return LANCZOS
                            except Exception as ex:
                                return 1  # 'Lanczos' irrespective of version
                        width, height = og_size # size of the slices to be rendered
                        coordinates, new_size = grid_coords(source.size, og_size, overlap)
                        if maximize == True:
                            source = source.resize(new_size, get_resampling_mode()) # minor concern that we're resizing twice
                            coordinates, new_size = grid_coords(source.size, og_size, overlap) # re-do the coordinates with the new canvas size
                        # loc_width and loc_height are the center point of the goal size, and we'll start there and work our way out
                        slices = []
                        for coordinate in coordinates:
                            x, y = coordinate
                            slices.append(((source.crop((x, y, x+width, y+height))), x, y))
                        global slices_todo
                        slices_todo = len(slices) - 1
                        return slices, new_size
                    def convert_pil_img(image):
                        w, h = image.size
                        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
                        image = image.resize((w, h), resample=LANCZOS)
                        image = np.array(image).astype(np.float32) / 255.0
                        image = image[None].transpose(0, 3, 1, 2)
                        image = torch.from_numpy(image)
                        return 2.*image - 1.

                    torch_gc()

                    #DynamicLoad_RealESRGAN(realesrgan_model_name)
                    #if opt.extern_upscaler:
                    init_image = Image.fromarray(x_sample)
                    output = run_RealESRGAN(init_image,upmodel_type, realesrgan_model_name, True, 'RGB',4,False) # no change in scaling with Gobig
                    res = output
                    #else:
                    #    output,img_mode = RealESRGAN.enhance(x_sample[:,:,::-1])
                    #    x_sample2 = output[:,:,::-1]
                    #    res = Image.fromarray(x_sample2)
                    #DynamicUnload_RealESRGAN()



                    X2_Output = res.resize((int(res.width/2), int(res.height/2)), LANCZOS)
                    filename = original_filename
                    #output_images.append(X2_Output)



                    sampler = DDIMSampler(model)
                    data = [batch_size * [prompt]]
                    gobig_overlap = 64

                    for _ in trange(1, desc="Passes"):

                        source_image = X2_Output
                        og_size = (int(source_image.size[0] / 2), int(source_image.size[1] / 2))
                        slices, _ = grid_slice(source_image, gobig_overlap, og_size, False)

                        betterslices = []

                        for _, chunk_w_coords in tqdm(enumerate(slices), "Slices"):
                            chunk, coord_x, coord_y = chunk_w_coords
                            init_image = convert_pil_img(chunk).to(device)
                            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

                            sampler.make_schedule(ddim_num_steps=int(gsteps), ddim_eta=0, verbose=False)

                            assert 0. <= gstrength <= 1., 'can only work with strength in [0.0, 1.0]'
                            t_enc = int(gstrength*gsteps)

                            with torch.no_grad():
                                with precision_scope("cuda"):
                                    with model.ema_scope():
                                        for prompts in tqdm(data, desc="data"):
                                            uc = (model if not opt.optimized else modelCS).get_learned_conditioning(len(prompts) * [""])
                                            if isinstance(prompts, tuple):
                                                prompts2 = list(prompts)
                                            else:
                                                prompts2 = prompts
                                            c = model.get_learned_conditioning(prompts2)

                                            # encode (scaled latent)
                                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                                            # decode it
                                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                                    unconditional_conditioning=uc,)

                                            x_samples = model.decode_first_stage(samples)
                                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                                            for x_sample2 in x_samples:
                                                x_sample2 = 255. * rearrange(x_sample2.cpu().numpy(), 'c h w -> h w c')
                                                resultslice = Image.fromarray(x_sample2.astype(np.uint8)).convert('RGBA')
                                                betterslices.append((resultslice.copy(), coord_x, coord_y))

                        alpha = Image.new('L', og_size, color=0xFF)
                        alpha_gradient = ImageDraw.Draw(alpha)
                        a = 0
                        ia = 0
                        overlap = gobig_overlap
                        shape = (og_size, (0,0))
                        while ia < overlap:
                            alpha_gradient.rectangle(shape, fill = a)
                            a += 4
                            ia += 1
                            shape = ((og_size[0] - ia, og_size[1]- ia), (ia,ia))
                        mask = Image.new('RGBA', og_size, color=0)
                        mask.putalpha(alpha)
                        finished_slices = []
                        for betterslice, x, y in betterslices:
                            finished_slice = addalpha(betterslice, mask)
                            finished_slices.append((finished_slice, x, y))
                        # # Once we have all our images, use grid_merge back onto the source, then save
                        goBig_output = grid_merge(source_image.convert("RGBA"), finished_slices).convert("RGB")
                        filename = original_filename
                        image = goBig_output


                if not use_GoBIG:
                    image = Image.fromarray(x_sample)


                filename = f"{base_count:05}-{seeds[i]}_{prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.png"
                if not skip_save and init_img is not None:
                    image.save(os.path.join(sample_path, filename))

                output_name.append(filename)
                output_images.append(image)
                torch_gc()


        



    toc = time.time()

    if (prompt_matrix or not skip_grid) and not do_not_save_grid:
        if prompt_matrix:
            if prompt.startswith("@"):
                grid = image_grid(output_images, batch_size, force_n_rows=frows, captions=prompt_matrix_parts)
            else:
                grid = image_grid(output_images, batch_size, force_n_rows=1 << ((len(prompt_matrix_parts)-1)//2))
                try:
                    grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
                except:
                    import traceback
                    print("Error creating prompt_matrix text:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
        else:
            grid = image_grid(output_images, batch_size)
 
        if grid:
            output_images.insert(0, grid)
    
        grid_count = get_next_sequence_number(outpath, 'grid-')
        grid_file = f"grid-{grid_count:05}-{seed}_{prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.{grid_ext}"
        grid.save(os.path.join(outpath, grid_file), grid_format, quality=grid_quality, lossless=grid_lossless, optimize=True)

        grid_count += 1
    if init_img is None:
        print(len(output_images))
        print(len(output_name))


        for n in range(0,len(output_images)):
            if use_GFPGAN:
                if GFPGAN is None:
                    DynamicLoad_GFPGAN()
                image = output_images[n]
                output_images[n] = run_GFPGAN(image, 1.0, False)

        if GFPGAN is not None:
            DynamicUnload_GFPGAN()

        for n in range(0,len(output_images)):
            if use_RealESRGAN and not use_GoBIG:
                #if RealESRGAN is None:
                #    DynamicLoad_RealESRGAN(realesrgan_model_name)
                image = output_images[n]
                output_images[n] = run_RealESRGAN(image,upmodel_type, realesrgan_model_name, False, 'RGB', esrgan_scale)
            if not skip_save:
                output_images[n].save(os.path.join(sample_path, output_name[n]))

        if RealESRGAN is not None:
            DynamicUnload_RealESRGAN()

        if opt.optimized:
            mem = torch.cuda.memory_allocated()/1e6
            modelFS.to("cpu")
            while(torch.cuda.memory_allocated()/1e6 >= mem):
                time.sleep(1)

        toc = time.time()

    mem_max_used, mem_total = mem_mon.read_and_stop()
    time_diff = time.time()-start_time



    info = f"""
{prompt}
Steps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, {f'Denoising Strength:{denoising_strength}' if init_img is not None else ''}Size: {width} x {height}, Batch Seed: {all_seeds}{', GFPGAN: Enabled' if use_GFPGAN else ', GFPGAN: Disabled'}{f', Upscaler:{upmodel_type}'if use_RealESRGAN else 'None'}{f', Model:{realesrgan_model_name}' if use_RealESRGAN else ''}{', Prompt Matrix Mode.' if prompt_matrix else ''}""".strip()
    stats = f'''
Took { round(time_diff, 2) }s total ({ round(time_diff/(len(all_prompts)),2) }s per image)
Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%'''

    for comment in comments:
        info += "\n\n" + comment

    #mem_mon.stop()
    #del mem_mon
    torch_gc()

    return output_images, seed, info, stats


old_images = []
new_images = []
old_info = f""
new_info = f""
old_params = []
new_params = []
def txt2img(prompt: str, ddim_steps: int, sampler_name: str, toggles: List[int], upmodel_type, realesrgan_model_name: str,esrgan_scale:int, ddim_eta: float, n_iter: int,
            batch_size: int,cfg_choice: int, cfg_scale: float,pcfg_scale: float,gstrength: float,gsteps: float, seed: Union[int, str, None], height: int, width: int, fp):
    outpath = opt.outdir_txt2img or opt.outdir or "outputs/txt2img-samples"
    err = False
    seed = seed_to_int(seed)

    prompt_matrix = 0 in toggles
    normalize_prompt_weights = 1 in toggles
    skip_save = 2 not in toggles
    skip_grid = 3 not in toggles
    sort_samples = 4 in toggles
    write_info_files = 5 in toggles
    jpg_sample = 6 in toggles
    use_GoBIG = 7 in toggles
    use_GFPGAN = 8 in toggles
    use_RealESRGAN = 9 in toggles


    used_cfg = cfg_scale if cfg_choice == 0 else pcfg_scale
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

    global old_images
    global new_images
    global old_info
    global new_info
    global old_params
    global new_params
    #if replace_old:
    #    old_images = new_images
    #    old_info = new_info
    #    old_params = new_params
    new_images = []
    new_info = []

    def init():
        pass

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=used_cfg, unconditional_conditioning=unconditional_conditioning, eta=ddim_eta, x_T=x)
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
            cfg_scale=used_cfg,
            width=width,
            height=height,
            prompt_matrix=prompt_matrix,
            use_GFPGAN=use_GFPGAN,
            use_RealESRGAN=use_RealESRGAN,
            use_GoBIG=use_GoBIG,
            gstrength=gstrength,
            gsteps=gsteps,
            upmodel_type=upmodel_type,
            realesrgan_model_name=realesrgan_model_name,
            esrgan_scale=esrgan_scale,
            fp=fp,
            ddim_eta=ddim_eta,
            normalize_prompt_weights=normalize_prompt_weights,
            sort_samples=sort_samples,
            write_info_files=write_info_files,
            jpg_sample=jpg_sample,
        )

        del sampler
        new_images = output_images
        new_info = info
        new_params = (prompt, ddim_steps, sampler_name, toggles, realesrgan_model_name, ddim_eta, n_iter, batch_size, cfg_choice, cfg_scale, pcfg_scale, height, width, fp, seed )
        return output_images, seed, info, stats
    except RuntimeError as e:
        err = e
        err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
        stats = err_msg
        return [], seed, 'err', stats
    finally:
        if err:
            crash(err, '!!Runtime error (txt2img)!!')

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
    return old_images, old_info
def SaveToCsv(images, image_index:int, use_history=False):
    import csv
    if len(images) == 0:
        return
    global old_params
    global new_params
    os.makedirs("log/images", exist_ok=True)
        # those must match the "txt2img" function !! + images, seed, comment, stats !! NOTE: changes to UI output must be reflected here too

    filenames = []
    if not image_index-1 < len(images):
        image_index = len(images)
    if not image_index-1 >=0:
        image_index = 0


    savedImgs = images[int(image_index-1)]
    params = new_params if not use_history else old_params
    prompt, ddim_steps, sampler_name, toggles, realesrgan_model_name, ddim_eta, n_iter, batch_size, cfg_choice, cfg_scale, pcfg_scale, height, width, fp, seed = params
    seed = seed+(image_index-1)
    cfg = cfg_scale if cfg_choice == 0 else pcfg_scale
    with open("log/Resultlog.csv", "a", encoding="utf8", newline='') as file:
        import time
        import base64

        at_start = file.tell() == 0
        writer = csv.writer(file, delimiter =",")
        if at_start:
            writer.writerow(["prompt", "seed", "width", "height", "sampler", "toggles", "n_iter", "n_samples", "cfg_scale", "steps", "filename", "RealESRGAN name"])

        filename_base = str(int(time.time() * 1000))

        filename = "log/images/"+filename_base + "-"+str(int(image_index)) + ".png"

        if savedImgs.startswith("data:image/png;base64,"):
            savedImgs = savedImgs[len("data:image/png;base64,"):]

        with open(filename, "wb") as imgfile:
            imgfile.write(base64.decodebytes(savedImgs.encode('utf-8')))

        use_RealESRGAN = 8 in toggles

        writer.writerow([prompt if prompt else " ", seed, width, height, sampler_name, toggles, n_iter, batch_size, cfg, ddim_steps, filename, realesrgan_model_name if use_RealESRGAN else 'Disabled'])
    return gr.update(value ="log/Resultlog.csv")


def SaveToCsvHistory(old_img,image_index):
    return SaveToCsv(old_img,image_index, True)


class Flagging(gr.FlaggingCallback):

    def setup(self, components, flagging_dir: str):
        pass

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None):
        import csv

        os.makedirs("log/images", exist_ok=True)

        # those must match the "txt2img" function !! + images, seed, comment, stats !! NOTE: changes to UI output must be reflected here too
        prompt, ddim_steps, sampler_name, toggles, ddim_eta, n_iter, batch_size, cfg_pecision, cfg_scale, pcfg_scale, seed, height, width, fp, old_img, old_info,images, seed, comment, stats = flag_data

        filenames = []
        cfg = cfg_scale if cfg_choice == 0 else pcfg_scale

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

            writer.writerow([prompt, seed, width, height, sampler_name, toggles, n_iter, batch_size, cfg, ddim_steps, filenames[0]])

        print("Logged:", filenames[0])


def img2img(prompt: str, image_editor_mode: str, init_info, mask_mode: str, mask_blur_strength: int, ddim_steps: int, sampler_name: str,
            toggles: List[int], upmodel_type, realesrgan_model_name: str,esrgan_scale: int, n_iter: int, batch_size: int, cfg_scale: float,gstrength:float, gsteps:float, denoising_strength: float,
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
    sort_samples = 6 in toggles
    write_info_files = 7 in toggles
    jpg_sample = 8 in toggles
    use_GFPGAN = 9 in toggles
    use_RealESRGAN = 10 in toggles

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
        init_img = init_info["image"]
        init_img = init_img.convert("RGB")
        init_img = resize_image(resize_mode, init_img, width, height)
        init_mask = init_info["mask"]
        init_mask = init_mask.convert("RGB")
        init_mask = resize_image(resize_mode, init_mask, width, height)
        keep_mask = mask_mode == 0
        init_mask = init_mask if keep_mask else ImageOps.invert(init_mask)
    else:
        init_img = init_info
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

        if opt.optimized:
            modelFS.to(device)

        init_image = 2. * image - 1.
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = (model if not opt.optimized else modelFS).get_first_stage_encoding((model if not opt.optimized else modelFS).encode_first_stage(init_image))  # move to latent space

        if opt.optimized:
            mem = torch.cuda.memory_allocated()/1e6
            modelFS.to("cpu")
            while(torch.cuda.memory_allocated()/1e6 >= mem):
                time.sleep(1)

        return init_latent,

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        if sampler_name != 'DDIM':
            x0, = init_data

            sigmas = sampler.model_wrap.get_sigmas(ddim_steps)
            noise = x * sigmas[ddim_steps - t_enc - 1]

            xi = x0 + noise
            sigma_sched = sigmas[ddim_steps - t_enc - 1:]
            model_wrap_cfg = CFGDenoiser(sampler.model_wrap)
            samples_ddim = K.sampling.__dict__[f'sample_{sampler.get_sampler_name()}'](model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': cfg_scale}, disable=False)
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
                    use_GoBIG=use_GoBIG,
                    gstrength=gstrength,
                    gsteps=gsteps,
                    upmodel_type=upmodel_type,
                    realesrgan_model_name=realesrgan_model_name,
                    esrgan_scale=esrgan_scale,
                    fp=fp,
                    do_not_save_grid=True,
                    normalize_prompt_weights=normalize_prompt_weights,
                    init_img=init_img,
                    init_mask=init_mask,
                    keep_mask=keep_mask,
                    mask_blur_strength=mask_blur_strength,
                    denoising_strength=denoising_strength,
                    resize_mode=resize_mode,
                    uses_loopback=loopback,
                    uses_random_seed_loopback=random_seed_loopback,
                    sort_samples=sort_samples,
                    write_info_files=write_info_files,
                    jpg_sample=jpg_sample,
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
                grid_count = get_next_sequence_number(outpath, 'grid-')
                grid = image_grid(history, batch_size, force_n_rows=1)
                grid_file = f"grid-{grid_count:05}-{seed}_{prompt.replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.{grid_ext}"
                grid.save(os.path.join(outpath, grid_file), grid_format, quality=grid_quality, lossless=grid_lossless, optimize=True)


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
                use_GoBIG=use_GoBIG,
                gstrength=gstrength,
                gsteps=gsteps,
                upmodel_type=upmodel_type,
                realesrgan_model_name=realesrgan_model_name,
                esrgan_scale=esrgan_scale,
                fp=fp,
                normalize_prompt_weights=normalize_prompt_weights,
                init_img=init_img,
                init_mask=init_mask,
                keep_mask=keep_mask,
                mask_blur_strength=mask_blur_strength,
                denoising_strength=denoising_strength,
                resize_mode=resize_mode,
                uses_loopback=loopback,
                sort_samples=sort_samples,
                write_info_files=write_info_files,
                jpg_sample=jpg_sample,
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
    print(f"weight : '{weights}' for text: '{prompts}'")
    return prompts, weights

def run_GFPGAN(image, strength, autoload = True):

    if autoload:
        DynamicLoad_GFPGAN()
    image = image.convert("RGB")

    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(np.array(image, dtype=np.uint8), has_aligned=False, only_center_face=False, paste_back=True)
    res = Image.fromarray(restored_img)

    if strength < 1.0:
        res = Image.blend(image, res, strength)
    if autoload:
        DynamicUnload_GFPGAN()

    return res

def run_RealESRGAN(image, model_type: str, model_name: str, autoload= True, Itype='RGB', scale_ratio=4, do_array=False):
    Internal = model_type == 'Internal'
    index= upscalers_type.index(model_type)
    arg = upscalers_args[index]
    scale_list = upscaler_scale[index]
    try:

        scale_id = scale_list.index(scale_ratio)
    except ValueError:
        scale_ratio = scale_list[len(scale_list)-1]
    #if opt.extern_upscaler and not Internal:
    if not Internal:
        torch_gc()
        time.sleep(2)
        if not os.path.exists('./TMP'):
            os.mkdir('./TMP')
        image.save(f'./TMP/original.png')
        try:
            print(f'NAME={model_name}')
            subprocess.run(
                ['./'+model_type, '-i', './TMP/original.png', '-o', './TMP/output.png', str(arg), str(model_name), '-s', str(int(scale_ratio))],
                stdout=subprocess.PIPE
            ).stdout.decode('utf-8')
            output = Image.open('./TMP/output.png').convert(Itype)

            return output
        except Exception as e:
            print('ESRGAN resize failed. Make sure realesrgan-ncnn-vulkan is in your path (or in this directory)')
            print(e)
            return None
    else:

        DynamicLoad_RealESRGAN(model_name)


        image = image.convert(Itype)


        output, img_mode = RealESRGAN.enhance(np.array(image, dtype=np.uint8), outscale=scale_ratio)
        if not do_array:
            res = Image.fromarray(output)

        else:
            res = Image.fromarray(output[:,:,::-1])
        if autoload:
            DynamicUnload_RealESRGAN()

    return res

def run_esrganScaled(image,model_type, model_name: str, scale_ratio):
    return run_RealESRGAN(image,model_type, model_name, True, 'RGB', scale_ratio)

def Get_all_samples(GenType: int, ImgType: int):

    GenPath = ["outputs/txt2img-samples", "outputs/img2img-samples"]
    ImgPath = ["samples", ""]
    fullpath = os.path.join(GenPath[GenType], ImgPath[ImgType])
    output_images = []
    output_name = []
    for file in os.listdir(fullpath):
            if file.endswith("png"):
                output_images.append(str(os.path.join(fullpath, file)))
                #output_images.append(Image.open(os.path.join(fullpath, file)))
    return output_images


def run_goBIG(image, upmodel_type, model_name: str, gstrength:float, gsteps:int, prompt: str):
    outpath = opt.outdir_goBig or opt.outdir or "outputs/gobig-samples"
    os.makedirs(outpath, exist_ok=True)
    def addalpha(im, mask):
        imr, img, imb, ima = im.split()
        mmr, mmg, mmb, mma = mask.split()
        im = Image.merge('RGBA', [imr, img, imb, mma])  # we want the RGB from the original, but the transparency from the mask
        return(im)
    def grid_merge(source, slices):
        source.convert("RGBA")
        for slice, posx, posy in slices: # go in reverse to get proper stacking
            source.alpha_composite(slice, (posx, posy))
        return source
    def grid_slice(source, overlap, og_size, maximize=False):
        def grid_coords(target, original, overlap):
            #generate a list of coordinate tuples for our sections, in order of how they'll be rendered
            #target should be the size for the gobig result, original is the size of each chunk being rendered
            center = []
            target_x, target_y = target
            center_x = int(target_x / 2)
            center_y = int(target_y / 2)
            original_x, original_y = original
            x = center_x - int(original_x / 2)
            y = center_y - int(original_y / 2)
            center.append((x,y)) #center chunk
            uy = y #up
            uy_list = []
            dy = y #down
            dy_list = []
            lx = x #left
            lx_list = []
            rx = x #right
            rx_list = []
            while uy > 0: #center row vertical up
                uy = uy - original_y + overlap
                uy_list.append((lx, uy))
            while (dy + original_y) <= target_y: #center row vertical down
                dy = dy + original_y - overlap
                dy_list.append((rx, dy))
            while lx > 0:
                lx = lx - original_x + overlap
                lx_list.append((lx, y))
                uy = y
                while uy > 0:
                    uy = uy - original_y + overlap
                    uy_list.append((lx, uy))
                dy = y
                while (dy + original_y) <= target_y:
                    dy = dy + original_y - overlap
                    dy_list.append((lx, dy))
            while (rx + original_x) <= target_x:
                rx = rx + original_x - overlap
                rx_list.append((rx, y))
                uy = y
                while uy > 0:
                    uy = uy - original_y + overlap
                    uy_list.append((rx, uy))
                dy = y
                while (dy + original_y) <= target_y:
                    dy = dy + original_y - overlap
                    dy_list.append((rx, dy))
            # calculate a new size that will fill the canvas, which will be optionally used in grid_slice and go_big
            last_coordx, last_coordy = dy_list[-1:][0]
            render_edgey = last_coordy + original_y # outer bottom edge of the render canvas
            render_edgex = last_coordx + original_x # outer side edge of the render canvas
            scalarx = render_edgex / target_x
            scalary = render_edgey / target_y
            if scalarx <= scalary:
                new_edgex = int(target_x * scalarx)
                new_edgey = int(target_y * scalarx)
            else:
                new_edgex = int(target_x * scalary)
                new_edgey = int(target_y * scalary)
            # now put all the chunks into one master list of coordinates (essentially reverse of how we calculated them so that the central slices will be on top)
            result = []
            for coords in dy_list[::-1]:
                result.append(coords)
            for coords in uy_list[::-1]:
                result.append(coords)
            for coords in rx_list[::-1]:
                result.append(coords)
            for coords in lx_list[::-1]:
                result.append(coords)
            result.append(center[0])
            return result, (new_edgex, new_edgey)
        def get_resampling_mode():
            try:
                from PIL import __version__, Image
                major_ver = int(__version__.split('.')[0])
                if major_ver >= 9:
                    return Image.Resampling.LANCZOS
                else:
                    return LANCZOS
            except Exception as ex:
                return 1  # 'Lanczos' irrespective of version
        width, height = og_size # size of the slices to be rendered
        coordinates, new_size = grid_coords(source.size, og_size, overlap)
        if maximize == True:
            source = source.resize(new_size, get_resampling_mode()) # minor concern that we're resizing twice
            coordinates, new_size = grid_coords(source.size, og_size, overlap) # re-do the coordinates with the new canvas size
        # loc_width and loc_height are the center point of the goal size, and we'll start there and work our way out
        slices = []
        for coordinate in coordinates:
            x, y = coordinate
            slices.append(((source.crop((x, y, x+width, y+height))), x, y))
        global slices_todo
        slices_todo = len(slices) - 1
        return slices, new_size
    def convert_pil_img(image):
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.*image - 1.

    #DynamicLoad_RealESRGAN(model_name)
    image = image.convert("RGB")
    #if opt.extern_upscaler:
    res = run_RealESRGAN(image,upmodel_type, model_name, True, 'RGB')
    #else:
    #    output, img_mode = RealESRGAN.enhance(np.array(image, dtype=np.uint8))
    #    output = Image.fromarray(output)
    #DynamicUnload_RealESRGAN()
    #resize output to half size
    #convert output to single segment array


    res = res.resize((int(res.width/2), int(res.height/2)), LANCZOS)

    sampler = DDIMSampler(model)

    gobig_overlap = 64
    batch_size = 1
    data = [batch_size * [prompt]]
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    base_filename = 'sampleTest'
    res.save(os.path.join(outpath, f"{base_filename}.png"))
    image.save(os.path.join(outpath, f"{base_filename}ORG.png"))


    with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for _ in trange(1, desc="Passes"):
                        #realesrgan2x(opt.realesrgan, os.path.join(sample_path, f"{base_filename}.png"), os.path.join(sample_path, f"{base_filename}u.png"))
                        base_filename = f"{base_filename}u"

                        source_image = res
                        og_size = (int(source_image.size[0] / 2), int(source_image.size[1] / 2))
                        slices, _ = grid_slice(source_image, gobig_overlap, og_size, False)

                        betterslices = []
                        for _, chunk_w_coords in tqdm(enumerate(slices), "Slices"):
                            chunk, coord_x, coord_y = chunk_w_coords
                            init_image = convert_pil_img(chunk).to(device)
                            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

                            sampler.make_schedule(ddim_num_steps=int(gsteps), ddim_eta=0, verbose=False)

                            assert 0. <= gstrength <= 1., 'can only work with strength in [0.0, 1.0]'
                            t_enc = int(gstrength * gsteps)

                            with torch.no_grad():
                                with precision_scope("cuda"):
                                    with model.ema_scope():
                                        for prompts in tqdm(data, desc="data"):
                                            uc = model.get_learned_conditioning(batch_size * [""]) #'4k'
                                            if isinstance(prompts, tuple):
                                                prompts = list(prompts)
                                            c = model.get_learned_conditioning(prompts)

                                            # encode (scaled latent)
                                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                                            # decode it
                                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                                    unconditional_conditioning=uc,)

                                            x_samples = model.decode_first_stage(samples)
                                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                                            for x_sample in x_samples:
                                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                                resultslice = Image.fromarray(x_sample.astype(np.uint8)).convert('RGBA')
                                                betterslices.append((resultslice.copy(), coord_x, coord_y))

                        alpha = Image.new('L', og_size, color=0xFF)
                        alpha_gradient = ImageDraw.Draw(alpha)
                        a = 0
                        i = 0
                        overlap = gobig_overlap
                        shape = (og_size, (0,0))
                        while i < overlap:
                            alpha_gradient.rectangle(shape, fill = a)
                            a += 4
                            i += 1
                            shape = ((og_size[0] - i, og_size[1]- i), (i,i))
                        mask = Image.new('RGBA', og_size, color=0)
                        mask.putalpha(alpha)
                        finished_slices = []
                        for betterslice, x, y in betterslices:
                            finished_slice = addalpha(betterslice, mask)
                            finished_slices.append((finished_slice, x, y))
                        # # Once we have all our images, use grid_merge back onto the source, then save
                        final_output = grid_merge(source_image.convert("RGBA"), finished_slices).convert("RGB")
                        final_output.save(os.path.join(outpath, f"{base_filename}d.png"))
                        base_filename = f"{base_filename}d"

                        torch_gc()

                        #put_watermark(final_output, wm_encoder)
                        final_output.save(os.path.join(outpath, f"{base_filename}.png"))



    output, img_mode = RealESRGAN.enhance(np.array(image, dtype=np.uint8))
    res = Image.fromarray(output)

    return res


if opt.defaults is not None and os.path.isfile(opt.defaults):
    try:
        with open(opt.defaults, "r", encoding="utf8") as f:
            user_defaults = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        print(f"Error loading defaults file {opt.defaults}:", e, file=sys.stderr)
        print("Falling back to program defaults.", file=sys.stderr)
        user_defaults = {}
else:
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
txt2img_toggles.append('Upscale images using goBig')
txt2img_toggles.append('Fix faces using GFPGAN')
txt2img_toggles.append('Upscale images')
#txt2img_toggles.append('Replace Old Image with newly generated Image (GUI ONLY)')



txt2img_defaults = {
    'prompt': '',
    'ddim_steps': 50,
    'toggles': [1, 2, 3, 6],
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
img2img_toggles.append('Upscale images goBig')
img2img_toggles.append('Fix faces using GFPGAN')

img2img_toggles.append('Upscale images')

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

if 'img2img' in user_defaults:
    img2img_defaults.update(user_defaults['img2img'])

img2img_toggle_defaults = [img2img_toggles[i] for i in img2img_defaults['toggles']]
img2img_image_mode = 'sketch'

def change_model(model_type: str):
    global used_upscaler
    used_upscaler = model_type
    if model_type == 'realesrgan-ncnn-vulkan':
        return gr.update(choices=ext_esrgan_models, value=ext_esrgan_models[0])
    if model_type == 'waifu2x-ncnn-vulkan':
        return gr.update(choices=ext_waifu_models, value=ext_waifu_models[0])
    if model_type == 'realsr-ncnn-vulkan':
        return gr.update(choices=ext_srgan_models, value=ext_srgan_models[0])
    if model_type == 'Internal':
        return gr.update(choices=internal_esrgan_model, value=internal_esrgan_model[0])

def change_image_editor_mode(choice, cropped_image, resize_mode, width, height):
    if choice == "Mask":
        return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)]
    return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]

def update_image_mask(cropped_image, resize_mode, width, height):
    resized_cropped_image = resize_image(resize_mode, cropped_image, width, height) if cropped_image else None
    return gr.update(value=resized_cropped_image)

def copy_img_to_input(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        tab_update = gr.update(selected='img2img_tab')
        img_update = gr.update(value=processed_image)
        return {img2img_image_mask: processed_image, img2img_image_editor: img_update, tabs: tab_update}
    except IndexError:
        return [None, None]


def copy_img_to_upscale_esrgan(img):
    update = gr.update(selected='realesrgan_tab')
    image_data = re.sub('^data:image/.+;base64,', '', img)
    processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return {realesrgan_source: processed_image, tabs: update}


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


def show_help():
    return [gr.update(visible=False), gr.update(visible=True), gr.update(value=help_text)]

def hide_help():
    return [gr.update(visible=True), gr.update(visible=False), gr.update(value="")]

def switch_history_vis(value:bool):
    return gr.update(visible=value)

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
#CSV_DRAW{
width: 100%;
max-height: 800px;
overflow: scroll;
flex: unset;
display: flex;

}
input[type=number]:disabled { -moz-appearance: textfield;+ }
"""

def show_gobig(toggles):
    show = 7 in toggles
    return [gr.update(visible=show), gr.update(visible=show)]

css = styling if opt.no_progressbar_hiding else styling + css_hide_progressbar
# This is the code that finds which selected item the user has in the gallery
js_part="""let getIndex = function(){
        let selected = document.querySelector('gradio-app').shadowRoot.querySelector('#gallery_output .\\\\!ring-2');
        return selected ? [...selected.parentNode.children].indexOf(selected) : 0;
    };"""
return_selected_img_js = "(x) => {" + js_part+ " document.querySelector('gradio-app').shadowRoot.querySelector('#img2img_editor .modify-upload button:last-child')?.click();return [x[getIndex()].replace('data:;','data:image/png;')]}"
copy_selected_img_js = "async (x) => {" + js_part+ """ 
let data = x[getIndex()];
const blob = await (await fetch(data.replace('data:;','data:image/png;'))).blob(); 
let item = new ClipboardItem({'image/png': blob})
navigator.clipboard.write([item]);
return x
}"""

with gr.Blocks(css=css, analytics_enabled=False, title="Stable Diffusion WebUI") as demo:
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
                
            with gr.Row(elem_id='body').style(equal_height=False):
                with gr.Column():
                    txt2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=txt2img_defaults["height"])
                    txt2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=txt2img_defaults["width"])
                    txt2img_cfgPrecision =gr.Radio(label='CFG Precision', choices=["Normal", "Precise"],type="index", value="Normal")
                    txt2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=txt2img_defaults['cfg_scale'])
                    txt2img_pcfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.1, label='Precise Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=txt2img_defaults['cfg_scale'])
                    txt2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, max_lines=1, value=txt2img_defaults["seed"])
                    txt2img_batch_count = gr.Slider(minimum=1, maximum=20, step=1, label='Batch count (how many batches of images to generate)', value=txt2img_defaults['n_iter'])
                    txt2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=txt2img_defaults['batch_size'])
                    txt2img_goBig_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='GoBIG Detail Enhancment (Lower will look more like the original)', value=0.3,interactive=True)
                    txt2img_goBig_steps = gr.Slider(minimum=1, maximum=300, step=1, label='GoBIG Sampling Steps', value=150,interactive=True)

                with gr.Column():
                    output_txt2img_history = gr.Box()

                    with output_txt2img_history:
                        gr.Markdown("Image History")
                        output_txt2img_showHostory = gr.Checkbox(label="Open", value=True)
                        output_txt2img_history_intern = gr.Box()
                        with output_txt2img_history_intern:
                            output_txt2img_oldImg = gr.Gallery(label="Old Images", elem_id="gallery_output").style(grid=[4,4])
                            with gr.Row():
                                with gr.Group():
                                    output_txt2img_select_imageold = gr.Number(label='Select image number from results for copy/Save', value=1, precision=None)
                                    output_txt2img_copy_to_input_btnold = gr.Button("Copy selected image to img2img input")
                                    output_txt2img_save_to_csv_btnold = gr.Button("Save the Selected Image parameters that are currently in the history").style(rounded=[False, True, True, True])

                            with gr.Row():
                                output_txt2img_oldParam = gr.Textbox(label="generation parameters History")




                    with gr.Box():
                        output_txt2img_NewImg = gr.Gallery(label="Images", elem_id="gallery_output").style(grid=[4,4])
                        with gr.Row():
                            with gr.Group():
                                output_txt2img_seed = gr.Number(label='Seed', interactive=False)
                                output_txt2img_copy_seed = gr.Button("Copy").click(inputs=output_txt2img_seed, outputs=[], _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                            with gr.Group():
                                output_txt2img_select_imagenew = gr.Number(label='Select image number from results for Copy/Save', value=1, precision=None)
                                output_txt2img_copy_to_input_btnnew = gr.Button("Push to img2img").style(full_width=True)
                                output_txt2img_save_to_csv_btnnew = gr.Button("Save the Selected Image parameters").style(rounded=[False,False,False,True])
                                output_txt2img_copy_to_gobig_input_btn = gr.Button("Copy selected image to goBig input").style(rounded=[False,False,True,False])

                        with gr.Group():
                            output_txt2img_params = gr.Textbox(label="Copy-paste generation parameters", interactive=False)
                            output_txt2img_save_to_history_btn= gr.Button("Save Images Into History").style(rounded=[False,False,False,True])
                            output_txt2img_copy_params = gr.Button("Copy").style(rounded=[False,False,True,False]).click(inputs=output_txt2img_params, outputs=[], _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                    output_txt2img_stats = gr.HTML(label='Stats')


                with gr.Column():
                    txt2img_btn = gr.Button("Generate", elem_id="generate", variant="primary")
                    txt2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=txt2img_defaults['ddim_steps'])
                    txt2img_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=txt2img_defaults['sampler_name'])
                    with gr.Tabs():
                        with gr.TabItem('Simple'):
                            txt2img_submit_on_enter = gr.Radio(['Yes', 'No'], label="Submit on enter? (no means multiline)", value=txt2img_defaults['submit_on_enter'], interactive=True)
                            txt2img_submit_on_enter.change(lambda x: gr.update(max_lines=1 if x == 'Single' else 25) , txt2img_submit_on_enter, txt2img_prompt)
                        with gr.TabItem('Advanced'):
                            txt2img_togglesBox = gr.CheckboxGroup(label='', choices=txt2img_toggles, value=txt2img_toggle_defaults, type="index")
                            txt2img_realesrgan_model_type = gr.Dropdown(label='Upscaler type (internal is RealESRGAN)', choices=upscalers_type, value=upscalers_type[0])
                            txt2img_realesrgan_model_name = gr.Dropdown(label='Upscaler model', choices=internal_esrgan_model, value=internal_esrgan_model[0]) # TODO: Feels like I shouldnt slot it in here.
                            txt2img_realesrgan_scale = gr.Slider(minimum=2.0, maximum=4.0, step=1, label="Upscale Ratio", value=4)
                            txt2img_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=txt2img_defaults['ddim_eta'], visible=False)
                    txt2img_embeddings = gr.File(label = "Embeddings file for textual inversion", visible=hasattr(model, "embedding_manager"))


            txt2img_realesrgan_model_type.change(
                change_model,
                [txt2img_realesrgan_model_type],
                [txt2img_realesrgan_model_name]
            )
            output_txt2img_showHostory.change(
                switch_history_vis,
                [output_txt2img_showHostory],
                [output_txt2img_history_intern]
            )
            txt2img_togglesBox.change(
                show_gobig,
                [txt2img_togglesBox],
                [txt2img_goBig_strength, txt2img_goBig_steps]
            )
            txt2img_btn.click(
                txt2img,
                [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_togglesBox, txt2img_realesrgan_model_type,txt2img_realesrgan_model_name,txt2img_realesrgan_scale, txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfgPrecision, txt2img_cfg, txt2img_pcfg,txt2img_goBig_strength,txt2img_goBig_steps, txt2img_seed, txt2img_height, txt2img_width, txt2img_embeddings],
                [output_txt2img_NewImg, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]
            )
            txt2img_prompt.submit(
                txt2img,
                [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_togglesBox, txt2img_realesrgan_model_type,txt2img_realesrgan_model_name,txt2img_realesrgan_scale, txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfgPrecision, txt2img_cfg, txt2img_pcfg,txt2img_goBig_strength,txt2img_goBig_steps, txt2img_seed, txt2img_height, txt2img_width, txt2img_embeddings],
                [output_txt2img_NewImg, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]
            )


            output_txt2img_save_to_history_btn.click(
                SaveToHistory,
                None,
                [output_txt2img_oldImg,output_txt2img_oldParam]
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
                img2img_btn_mask = gr.Button("Generate",variant="primary", visible=False, elem_id="img2img_mask_btn")
                img2img_btn_editor = gr.Button("Generate",variant="primary", elem_id="img2img_editot_btn")
            with gr.Row().style(equal_height=False):
                with gr.Column():

                    img2img_image_editor_mode = gr.Radio(choices=["Mask", "Crop"], label="Image Editor Mode", value="Crop")
                    img2img_show_help_btn = gr.Button("Show Hints")
                    img2img_hide_help_btn = gr.Button("Hide Hints", visible=False)
                    img2img_help = gr.Markdown(visible=False, value="")
                    with gr.Row():
                        img2img_painterro_btn = gr.Button("Advanced Editor")
                        img2img_copy_from_painterro_btn = gr.Button(value="Get Image from Advanced Editor")
                    img2img_image_editor = gr.Image(value=sample_img2img, source="upload", interactive=True, type="pil", tool="select", elem_id="img2img_editor")
                    img2img_image_mask = gr.Image(value=sample_img2img, source="upload", interactive=True, type="pil", tool="sketch", visible=False, elem_id="img2img_mask")
                    img2img_mask = gr.Radio(choices=["Keep masked area", "Regenerate only masked area"], label="Mask Mode", type="index", value=img2img_mask_modes[img2img_defaults['mask_mode']], visible=False)
                    img2img_mask_blur_strength = gr.Slider(minimum=1, maximum=10, step=1, label="How much blurry should the mask be? (to avoid hard edges)", value=3, visible=False)
                    img2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=img2img_defaults['ddim_steps'])
                    img2img_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=img2img_defaults['sampler_name'])
                    img2img_togglesBox = gr.CheckboxGroup(label='', choices=img2img_toggles, value=img2img_toggle_defaults, type="index")
                    img2img_realesrgan_model_type = gr.Dropdown(label='Upscaler type (internal is RealESRGAN)', choices=upscalers_type, value=upscalers_type[0])
                    img2img_realesrgan_model_name = gr.Dropdown(label='Upscaler model', choices=internal_esrgan_model, value=internal_esrgan_model[0]) # TODO: Feels like I shouldnt slot it in here.
                    img2img_realesrgan_scale = gr.Slider(minimum=2.0, maximum=4.0, step=1, label="Upscale Ratio", value=4)
                    img2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1, label='Batch count (how many batches of images to generate)', value=img2img_defaults['n_iter'])
                    img2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=img2img_defaults['batch_size'])
                    img2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=img2img_defaults['cfg_scale'])
                    img2img_goBig_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='GoBIG Detail Enhancment (Lower will look more like the original)', value=0.3,interactive=True)
                    img2img_goBig_steps = gr.Slider(minimum=1, maximum=300, step=1, label='GoBIG Sampling Steps', value=150,interactive=True)
                    img2img_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength', value=img2img_defaults['denoising_strength'])
                    img2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, value=img2img_defaults["seed"])
                    img2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=img2img_defaults["height"])
                    img2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=img2img_defaults["width"])
                    img2img_resize = gr.Radio(label="Resize mode", choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value=img2img_resize_modes[img2img_defaults['resize_mode']])
                    img2img_embeddings = gr.File(label = "Embeddings file for textual inversion", visible=hasattr(model, "embedding_manager"))

                with gr.Column():
                    output_img2img_gallery = gr.Gallery(label="Images")
                    """
                    with gr.Tabs():
                        with gr.TabItem("Generated image actions", id="img2img_actions_tab"):
                            output_img2img_select_image = gr.Number(label='Select image number from results for copying', value=1, precision=None)
                            gr.Markdown("Clear the input image before copying your output to your input. It may take some time to load the image.")
                            output_img2img_copy_to_input_btn = gr.Button("Copy selected image to input")
                        with gr.TabItem("Output info", id="img2img_output_info_tab"):
                            output_img2img_params = gr.Textbox(label="Generation parameters")
                            with gr.Row():
                                output_img2img_copy_params = gr.Button("Copy full parameters").click(inputs=output_img2img_params, outputs=[], _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                output_img2img_seed = gr.Number(label='Seed', interactive=False, visible=False)
                                output_img2img_copy_seed = gr.Button("Copy only seed").click(inputs=output_img2img_seed, outputs=[], _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                            output_img2img_stats = gr.HTML(label='Stats')
                    """
                    output_img2img_select_image = gr.Number(label='Select image number from results for copying', value=1, precision=None)
                    gr.Markdown("Clear the input image before copying your output to your input. It may take some time to load the image.")
                    output_img2img_copy_to_input_btn = gr.Button("Copy selected image to input")


                    output_img2img_seed = gr.Number(label='Seed')
                    output_img2img_params = gr.Textbox(label="Copy-paste generation parameters")
                    output_img2img_stats = gr.HTML(label='Stats')
            img2img_realesrgan_model_type.change(
                change_model,
                [img2img_realesrgan_model_type],
                [img2img_realesrgan_model_name]
            )
            img2img_togglesBox.change(
                    show_gobig,
                    [img2img_togglesBox],
                    [img2img_goBig_strength, img2img_goBig_steps]
            )
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
                [output_img2img_gallery],
                [img2img_image_editor, img2img_image_mask, tabs],
                _js=return_selected_img_js
            )

            output_txt2img_copy_to_input_btnold.click(
                copy_img_to_input,
                [output_txt2img_select_imageold, output_txt2img_oldImg],
                [img2img_image_editor, img2img_image_mask,tabs]
            )
            output_txt2img_copy_to_input_btnnew.click(
                copy_img_to_input,
                [output_txt2img_select_imagenew, output_txt2img_NewImg],
                [img2img_image_editor, img2img_image_mask,tabs]
            )
            img2img_btn_mask.click(
                img2img,
                [img2img_prompt, img2img_image_editor_mode, img2img_image_mask, img2img_mask, img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_togglesBox, img2img_realesrgan_model_type, img2img_realesrgan_model_name, img2img_realesrgan_scale, img2img_batch_count, img2img_batch_size, img2img_cfg,txt2img_goBig_strength,txt2img_goBig_steps, img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize, img2img_embeddings],
                [output_img2img_gallery, output_img2img_seed, output_img2img_params, output_img2img_stats]
            )

            img2img_btn_editor.click(
                img2img,
                [img2img_prompt, img2img_image_editor_mode, img2img_image_editor, img2img_mask, img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_togglesBox, img2img_realesrgan_model_type, img2img_realesrgan_model_name, img2img_realesrgan_scale, img2img_batch_count, img2img_batch_size, img2img_cfg,txt2img_goBig_strength,txt2img_goBig_steps, img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize, img2img_embeddings],
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


            gfpgan_defaults = {
                'strength': 100,
            }

        if 'gfpgan' in user_defaults:
            gfpgan_defaults.update(user_defaults['gfpgan'])

        with gr.TabItem("GFPGAN",id='cfpgan_tab'):
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

        with gr.TabItem("RealESRGAN"):
            gr.Markdown("Upscale images")
            with gr.Row():
                with gr.Column():
                    realesrgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                    realesrgan_model_type = gr.Dropdown(label='Upscaler type (internal is RealESRGAN)', choices=upscalers_type, value=upscalers_type[0])
                    realesrgan_model_name = gr.Dropdown(label='Upscaler model', choices=internal_esrgan_model, value=internal_esrgan_model[0])
                    realesrgan_scale = gr.Slider(minimum=2.0, maximum=4.0, step=1, label="Upscale Ratio", value=4)
                    realesrgan_btn = gr.Button("Generate")
                with gr.Column():
                    realesrgan_output = gr.Image(label="Output")
            realesrgan_model_type.change(
                change_model,
                [realesrgan_model_type],
                [realesrgan_model_name]
            )
            realesrgan_btn.click(
                run_esrganScaled,
                [realesrgan_source,realesrgan_model_type, realesrgan_model_name, realesrgan_scale],
                [realesrgan_output]
            )
        with gr.TabItem("goBIG"):
            gr.Markdown("Upscale and detail images")
            with gr.Row():
                with gr.Column():
                    realesrganGoBig_source = gr.Image(source="upload", interactive=True, type="pil", tool="select")
                    realesrganGoBig_model_type = gr.Dropdown(label='Upscaler type (internal is RealESRGAN)', choices=upscalers_type, value=upscalers_type[0])
                    realesrganGoBig_model_name = gr.Dropdown(label='Upscaler model', choices=internal_esrgan_model, value=internal_esrgan_model[0])
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
            realesrgan_model_type.change(
                change_model,
                [realesrganGoBig_model_type],
                [realesrganGoBig_model_name]
            )
            realesrganGoBig_btn.click(
                run_goBIG,
                [realesrganGoBig_source, realesrganGoBig_model_type, realesrganGoBig_model_name, GoBig_strength, GoBig_steps, realesrganGoBig_prompt],
                [realesrganGoBig_output]
            )
            output_txt2img_copy_to_gobig_input_btn.click(
                copy_img_to_input,
                [output_txt2img_select_imagenew, output_txt2img_NewImg],
                [realesrganGoBig_source,realesrganGoBig_source]
            )
        with gr.TabItem("Parameter History"):
            def refresh():
                return gr.update(value ="log/Resultlog.csv")
            with gr.Column():
                csv_file = gr.Dataframe(value="log/Resultlog.csv", elem_id="CSV_DRAW").style(rounded=[True, True,False,False])
                csv_btn = gr.Button("Refresh", elem_id="generate").style(rounded=[False, False,True,True],full_width=True).click(refresh, None, [csv_file])

            output_txt2img_save_to_csv_btnold.click(
                SaveToCsvHistory,
                [output_txt2img_oldImg, output_txt2img_select_imageold],
                [csv_file]
            )
            output_txt2img_save_to_csv_btnnew.click(
                SaveToCsv,
                [output_txt2img_NewImg,output_txt2img_select_imagenew],
                [csv_file]
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
        gradio_params = {
            'show_error': True,
            'server_name': '0.0.0.0',
            'share': opt.share
        }
        if not opt.share:
            demo.queue(concurrency_count=1)
        if opt.share and opt.share_password:
            gradio_params['auth'] = ('webui', opt.share_password)
        self.demo.launch(**gradio_params)

    def stop(self):
        self.demo.close() # this tends to hang

def launch_server():
    server_thread = ServerLauncher(demo)
    server_thread.start()

    try:
        while server_thread.is_alive():
            time.sleep(60)
    except (KeyboardInterrupt, OSError) as e:
        print('\n', e)
        print('exiting...calling os._exit(0)')
        t = threading.Timer(0.25, os._exit, args=[0])
        t.start()

def run_headless():
    with open(opt.cli, 'r', encoding='utf8') as f:
        kwargs = yaml.safe_load(f)
    target = kwargs.pop('target')
    if target == 'txt2img':
        target_func = txt2img
    elif target == 'img2img':
        target_func = img2img
        raise NotImplementedError()
    else:
        raise ValueError(f'Unknown target: {target}')
    prompts = kwargs.pop("prompt")
    prompts = prompts if type(prompts) is list else [prompts]
    for i, prompt_i in enumerate(prompts):
        print(f"===== Prompt {i+1}/{len(prompts)}: {prompt_i} =====")
        output_images, seed, info, stats = target_func(prompt=prompt_i, **kwargs)
        print(f'Seed: {seed}')
        print(info)
        print(stats)
        print()

if __name__ == '__main__':
    if opt.cli is None:
        launch_server()
    else:
        run_headless()
