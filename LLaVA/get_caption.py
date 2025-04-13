from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from transformers import CLIPTextModel, CLIPTokenizer
import argparse
import os
from tqdm import tqdm
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description="Evaluate model with given image and prompt")
parser.add_argument('--img_fol', type=str, required=True, help='Path to the input image')
parser.add_argument('--res_fol', type=str, required=True, help='Path to the input image')
args1 = parser.parse_args()
model_path = "liuhaotian/llava-v1.6-vicuna-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)  

root_dir = os.path.join(os.getcwd(), args1.img_fol)
subdirs= []
for root, dirs, files in os.walk(root_dir):
    for subdir in dirs:
        if ('_' in subdir and 'ipynb' not in subdir):
            subdirs.append(subdir)

res_dir = os.path.join(os.getcwd(), args1.res_fol)

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
def clip_embedding(out):
    inputs = clip_tokenizer(out, return_tensors="pt")
    outputs = clip_model(**inputs)
    return outputs.last_hidden_state

print(subdirs)
for sd in tqdm(subdirs):
    with open(os.path.join(res_dir, sd, 'edit.txt'), 'r') as file:
        inv_ins = file.read()
    prompt = f'''Give me one line description of an image generated after applying the edits between Image1 and image2 on input image. The edits between image1 and image2 are: "{inv_ins}". Write your caption in one line based on content of given input image. That is, if some part of edit between image1 and image2 is not applicable to this input image you should skip it (for example if edit between image1 and image2 is to replace a pen with pencil, but the given input image does not contain any pen; just ignore that edit). Make sure that your caption completely describe the edited image obtained on editing the input image. The caption should not contain more than 20 words. so try to summarize the caption and use only relevant information using given image. Write only one line caption with less than 20 words. Write output as if you are describing the image but in one line in less than 20 words. (do not mention image1, image2 in your response).'''
    
    image_file = os.path.join(root_dir, sd, '1_0.png')

    args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
    })()

    out = eval_model(args,tokenizer, model, image_processor, context_len)
    # Create a clip embedding of the output
    clip_embedding = clip_embedding(out)
    # Dimensions will be [1, 77, 768]
    # print(out)
    with open(os.path.join(res_dir, sd, 'inv_cap.txt'), 'w') as file:
        file.write(out)
    torch.save(clip_embedding, os.path.join(res_dir, sd, 'inv_cap_clip.pt'))