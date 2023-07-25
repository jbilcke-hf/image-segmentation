import warnings
warnings.filterwarnings('ignore')

import subprocess, io, os, sys, time
os.system("pip install gradio==3.36.1")
import gradio as gr

from loguru import logger

# os.system("pip install diffuser==0.6.0")
# os.system("pip install transformers==4.29.1")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if os.environ.get('IS_MY_DEBUG') is None:
    result = subprocess.run(['pip', 'install', '-e', 'GroundingDINO'], check=True)
    print(f'pip install GroundingDINO = {result}')

# result = subprocess.run(['pip', 'list'], check=True)
# print(f'pip list = {result}')

sys.path.insert(0, './GroundingDINO')

import argparse
import copy

import numpy as np
import torch
from PIL import Image

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


def load_image(image_path):
    # # load image
    if isinstance(image_path, PIL.Image.Image):
        image_pil = image_path
    else:
        image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def run_inference(input_image, text_prompt, box_threshold, text_threshold, config_file, ckpt_repo_id, ckpt_filenmae):

    # Load the Grounding DINO model
    model = load_model_hf(config_file, ckpt_repo_id, ckpt_filenmae)

    # Load the input image
    image_pil, image = load_image(input_image)

    # Run the object detection and grounding model
    boxes, labels = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold)

    # Convert the boxes and labels to a JSON format
    result = []
    for box, label in zip(boxes, labels):
        result.append({
            "box": box.tolist(),
            "label": label
        })

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded SAM demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    args = parser.parse_args()
    print(f'args = {args}')

    model_config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    model_ckpt_repo_id = "ShilongLiu/GroundingDINO"
    model_ckpt_filenmae = "groundingdino_swint_ogc.pth"

    def inference_func(input_image, text_prompt):
        result = run_inference(input_image, text_prompt, 0.3, 0.25, model_config_file, model_ckpt_repo_id, model_ckpt_filenmae)
        return result

    # Create the Gradio interface for the model
    interface = gr.Interface(
        fn=inference_func,
        inputs=[
            gr.inputs.Image(label="Input Image"),
            gr.inputs.Textbox(label="Detection Prompt")
        ],
        outputs=gr.outputs.Dataframe("pandas"),
        title="Object Detection and Grounding",
        description="A Gradio app to detect objects in an image and ground them to captions using Grounding DINO.",
        server_name='0.0.0.0',
        debug=args.debug,
        share=args.share
    )

    # Launch the interface
    interface.launch()