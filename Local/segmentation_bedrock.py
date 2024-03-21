######################################## 사용법 #################################################
## ./local/segmentation_bedrock.py --config_file ./local/config/config.yaml --img_path ./test_images/andy_portrait_2.jpg
###############################################################################################
import base64
import io
import json
import os
import sys
import random

import boto3
from botocore.config import Config

import cv2
import copy

import argparse
import yaml

import numpy as np

## pip install git+https://github.com/facebookresearch/segment-anything.git
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image, ImageDraw, ExifTags, ImageColor


def model_load(cfg):    
    sam = sam_model_registry["vit_h"](checkpoint=cfg['checkpoint'])
    if cfg['use_cuda']:
        sam.to(device='cuda')
    
    predictor = SamPredictor(sam)
    return predictor


def boto3_runtime(service_name, target_region):
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )

    session = boto3.Session(**session_kwargs)

    boto3_runtime = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )
    
    return boto3_runtime


def show_faces(img_path, target_region):
    client = boto3_runtime(
        service_name='rekognition',
        target_region=target_region
    )
    
    with open(img_path, 'rb') as image:
        #Call DetectFaces 
        response = client.detect_faces(Image={'Bytes': image.read()},Attributes=['ALL'])

        image = Image.open(img_path)
        imgWidth, imgHeight = image.size       
        ori_image = copy.deepcopy(image)

        for faceDetail in response['FaceDetails']:
            print('The detected face is between ' + str(faceDetail['AgeRange']['Low']) 
                  + ' and ' + str(faceDetail['AgeRange']['High']) + ' years old')

            box = faceDetail['BoundingBox']
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']


            print('Left: ' + '{0:.0f}'.format(left))
            print('Top: ' + '{0:.0f}'.format(top))
            print('Face Width: ' + "{0:.0f}".format(width))
            print('Face Height: ' + "{0:.0f}".format(height))

    return ori_image, imgWidth, imgHeight, int(left), int(top), int(width), int(height), response


## 모델 별 label list : https://docs.aws.amazon.com/rekognition/latest/dg/labels.html
def show_labels(img_path, target_label=None, target_region='us-west-2'):
    client = boto3_runtime(
        service_name='rekognition',
        target_region=target_region
    )
    
    if target_label is None:
        Settings = {"GeneralLabels": {"LabelInclusionFilters":[]},"ImageProperties": {"MaxDominantColors":1}}
        # print(f"target_label_None : {target_label}")
    else:
        Settings = {"GeneralLabels": {"LabelInclusionFilters":[target_label]},"ImageProperties": {"MaxDominantColors":1}}
        # print(f"target_label : {target_label}")
    
    box = None
    with open(img_path, 'rb') as image:
        #Call DetectFaces 
        response = client.detect_labels(Image={'Bytes': image.read()},
                                        MaxLabels=15,
                                        MinConfidence=0.7,
                                        # Uncomment to use image properties and filtration settings
                                        Features=["GENERAL_LABELS", "IMAGE_PROPERTIES"],
                                        Settings=Settings
                                       )

        image = Image.open(img_path).convert('RGB')
        imgWidth, imgHeight = image.size       
        ori_image = copy.deepcopy(image)
        color = 'white'
    
        for item in response['Labels']:
            # print(item)
            if len(item['Instances']) > 0:
                print(item)
                print(item['Name'], item['Confidence'])

                for sub_item in item['Instances']:
                    color = sub_item['DominantColors'][0]['CSSColor']
                    box = sub_item['BoundingBox']
                    break
            break
        try:
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']


            print('Left: ' + '{0:.0f}'.format(left))
            print('Top: ' + '{0:.0f}'.format(top))
            print('Object Width: ' + "{0:.0f}".format(width))
            print('Object Height: ' + "{0:.0f}".format(height))
            return ori_image, imgWidth, imgHeight, int(left), int(top), int(width), int(height), color, response
        except:
            print("There is no target label in the image.")
            return _, _, _, _, _, _, _, _, _



def img_resize(image):
    imgWidth, imgHeight = image.size 


    if imgWidth < imgHeight:
        imgWidth = int(1024/imgHeight*imgWidth)
        imgWidth = imgWidth-imgWidth%64
        imgHeight = 1024
    else:
        imgHeight = int(1024/imgWidth*imgHeight)
        imgHeight = imgHeight-imgHeight%64
        imgWidth = 1024 

    image = image.resize((imgWidth, imgHeight), resample=0)
    return image


def image_to_base64(img) -> str:
    """Converts a PIL Image or local image file path to a base64 string"""
    if isinstance(img, str):
        if os.path.isfile(img):
            print(f"Reading image from file: {img}")
            with open(img, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise FileNotFoundError(f"File {img} does not exist")
    elif isinstance(img, Image.Image):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Expected str (filename) or PIL Image. Got {type(img)}")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="")
    parser.add_argument('--img_path', type=str, default="")
    parser.add_argument('--target_label', type=str, default="face")
    args = parser.parse_args()
    
    
    with open(args.config_file, mode="r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.target_label == 'face':
        f_image, width, height, f_left, f_top, f_width, f_height, human_res = show_faces(img_path=args.img_path, target_region=cfg['target_region'])
    else:
        f_image, width, height, f_left, f_top, f_width, f_height, color, human_res = show_labels(img_path=args.img_path, target_label=args.target_label, target_region=cfg['target_region'])
        
    ## PIL to OpenCV
    numpy_image = np.array(f_image)
    
    input_box = np.array([f_left, f_top, f_left+f_width, f_top+f_height])
    input_label = np.array([0])

    predictor = model_load(cfg)
    predictor.set_image(numpy_image)
    masks, _, _ = predictor.predict(box=input_box,
                                    point_labels=input_label,
                                    multimask_output=False)



    mask = (np.array(masks[0])) * 255.0
    mask = 255 - mask
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.convert('RGB')
    
    f_image = img_resize(f_image)
    mask_img = img_resize(mask_img)
    
    outpaint_prompt = random.choice(cfg['candidate_prompts'])
    seed_range = cfg['seed_range'].split(',')
    seed = random.randint(int(seed_range[0]),int(seed_range[1]))
    # outpaint_prompt = 'a man in a well tailored business suit in front of a plain background'
    # seed = 34
    print(f"outpaint_prompt : {outpaint_prompt}, seed : {seed}")
    # Payload creation
    body = json.dumps({
        "taskType": "OUTPAINTING",
        "outPaintingParams": {
            "text": outpaint_prompt,              # Optional
            # "negativeText": negative_prompts,    # Optional
            "image": image_to_base64(f_image),      # Required
            # "maskPrompt": mask_prompt,               # One of "maskImage" or "maskPrompt" is required
            "maskImage": image_to_base64(mask_img),  # Input maskImage based on the values 0 (black) or 255 (white) only
        },                                                 
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "premium",
            # "quality": "standard",
            "cfgScale": cfg['cfgScale'],
            "height": cfg['height'],
            "width": cfg['width'],
            "seed": seed  #42
        }
    })

    # Model invocation
    boto3_bedrock = boto3_runtime(
        service_name='bedrock-runtime',
        target_region=cfg['target_region']
    )

    response = boto3_bedrock.invoke_model(
        body=body,
        modelId=cfg['modelId'],
        accept="application/json", 
        contentType="application/json"
    )

    # Output processing
    response_body = json.loads(response.get("body").read())
    img_b64 = response_body["images"][0]
    print(f"Output: {img_b64[0:80]}...")

    # Decode + save
    img_result = Image.open(
        io.BytesIO(
            base64.decodebytes(
                bytes(img_b64, "utf-8")
            )
        )
    )
    if cfg['resize']:
        img_result = img_result.resize((360, 480), 0)
    output_path = cfg['result_path'] + '/generated_'+args.img_path.split('/')[-1]
    os.makedirs(cfg['result_path'], exist_ok=True)
    img_result.save(output_path)
    print(f"Gnerating an image in {output_path}")
        
        
if __name__ == '__main__':
    main()