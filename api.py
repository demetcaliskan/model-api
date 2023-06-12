from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import PIL
import requests
import random
from auth_token import auth_token
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory
from PIL import Image
import base64
from io import BytesIO
from pydantic import BaseModel


class ImageItem(BaseModel):
    address: str


class TransformItem(BaseModel):
    title: str
    url: str
    prompt: str
    neg_prompt: str
    num_inference_steps: int


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

device = "mps"
model_id = "timbrooks/instruct-pix2pix"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, safety_checker=None, use_auth_token=auth_token)

pipe.to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config)
pipe.enable_attention_slicing()


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image.thumbnail((500, 500))
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


@app.post("/img-upload")
def uploadImage(item: ImageItem):
    newSrc = item.address[22:]
    bytes_decoded = base64.b64decode(str.encode(newSrc))
    img = Image.open(BytesIO(bytes_decoded))
    convertedImage = img.convert("RGB")
    num = random.randint(0, 10000000000)
    imgPath = f"images/{str(num)}.png"
    convertedImage.save(imgPath)
    image = Image.open(imgPath)
    image.thumbnail((1000, 1000))
    image.save(imgPath)
    imageSource = f"http://127.0.0.1:8000/{imgPath}"
    resData = {"image": {
        "src": imageSource,
        "alt": ""
    }
    }
    return JSONResponse(content=jsonable_encoder(resData))


@app.post("/image-process")
def generate(item: TransformItem):
    img = download_image(item.url)
    image = pipe(item.prompt, negative_prompt=item.neg_prompt, image=img, num_inference_steps=item.num_inference_steps,
                 image_guidance_scale=1.65, guidance_scale=12).images[0]
    num = random.randint(0, 10000000000)
    imgPath = f"images/{str(num) + item.title}.png"
    image.save(imgPath)
    imageSource = f"http://127.0.0.1:8000/{imgPath}"
    resData = {"image": {
        "src": imageSource,
        "alt": item.title
    }
    }
    return JSONResponse(content=jsonable_encoder(resData))


@app.get("/translation")
def generate2(input_text: str):
    DetectorFactory.seed = 0
    model_name = f'Helsinki-NLP/opus-mt-tr-en'
    tokenizer = MarianTokenizer.from_pretrained(
        model_name, use_auth_token=auth_token)
    model = MarianMTModel.from_pretrained(
        model_name, use_auth_token=auth_token)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs)
    output_text = tokenizer.batch_decode(
        outputs, skip_special_tokens=True)[0]
    print("output", output_text)
    resData = {"translation": {
        "translatedText": output_text
    }
    }
    return JSONResponse(content=jsonable_encoder(resData))


app.mount("/images", StaticFiles(directory="images"), name="images")
