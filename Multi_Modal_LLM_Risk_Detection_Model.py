import os
import io
import re
import cv2
import csv
import torch
import base64
import numpy as np
import random
import pickle
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from openai import OpenAI
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq, AutoModelForImageTextToText
import pandas as pd

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

##################################### Setting #########################################
single_seq_frame_use = False # single frame 사용 여부 Flase이면 Sequence Image 사용
future_context_analyzing_use = True # 미래 예상되는 상황에 대한 분석 내용을 사용할지 여부, True이면 사용
past_context_analyzing_use = True # 과거 요약에 대한 사용할지 여부, True면 사용
LLM_Model_Select_multi_label_option_pass_context_model = 1 # 1 : GPT-4o, 2: InternVL3-8B, 3: Qwen2.5-VL-7B
multi_llm_decision_use = True ## True이면 최종 결정에서 3개의 llm을 사용해서 나온 결론을 vote로 확인할 경우, false이면 그냥 rule
image_dir = "/home/work/Dataset"
#######################################################################################################

################################### API Setting & pretrained model Setting ########################################
if LLM_Model_Select_multi_label_option_pass_context_model == 1 or multi_llm_decision_use :
    ############# GPT-4o Model -------------
    os.environ["OPENAI_API_KEY"] = "api key"
    client = OpenAI(timeout=60)
if LLM_Model_Select_multi_label_option_pass_context_model == 2 or multi_llm_decision_use :
    ############# InternVL3-8B Model -------           
    model_cache_dir = "/home/work/future_leejonghyun"
    internvl_model = AutoModel.from_pretrained('OpenGVLab/InternVL3-38B', cache_dir=model_cache_dir, torch_dtype=torch.bfloat16, load_in_8bit=False, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True, device_map="auto").eval()
    internvl_tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL3-38B', cache_dir=model_cache_dir, trust_remote_code=True, use_fast=False)
if LLM_Model_Select_multi_label_option_pass_context_model == 3 or multi_llm_decision_use :
    ############# Qwen2.5-VL-7B Model -------          
    Qwen_model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",  device_map="auto",trust_remote_code=True,torch_dtype="auto").eval()
    Qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",  trust_remote_code=True)

#########################################  필요한 function 정의 ########################################

# binary Class Report 관련 코드
def convert_to_binary_label(multilabel_arr, label_to_id):
    normal_idx = label_to_id["normal"]
    binary_labels = []
    for label_vector in multilabel_arr:
        label_vector = np.array(label_vector)
        if label_vector[normal_idx] == 1 and label_vector.sum() == 1:
            binary_labels.append("normal")
        else:
            binary_labels.append("risk")
    return binary_labels

def preprocess_image(image: Image.Image, input_size=448):
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),  # [0,1] 범위 float tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) 

def concat_images_horizontally(images, gap=10):
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths) + gap * (len(images) - 1)
    max_height = max(heights)
    new_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for i, img in enumerate(images):
        new_img.paste(img, (x_offset, 0))
        x_offset += img.size[0] + gap
    return new_img

def generate_caption_with_gpt4o_sequence(image_tensors, prompt, single_seq_frame_use, llm_option, task, sub_folder):
    concat_img = image_tensors
    buffered = io.BytesIO()
    concat_img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    img_base64_url = f"data:image/jpeg;base64,{img_base64}"
    
    messages = []

    if single_seq_frame_use :
        persona_block = ("You are a safety surveillance AI system that views CCTV images and analyzes them for risks.\n"
                "The goal is to quickly identify potential risk signs from the images.\n"
                f"Types of risk include {sub_folder}.\n"
                "The analysis must assume that there may be risk types situations, rather than always normal type situations.\n"
                "Even the smallest abnormality signals must be recognized.\n")
    else : 
        persona_block = (
                "You are a safety surveillance AI system that views 10 consecutive CCTV images and analyzes them for risks.\n"
                "The images are arranged from left (earliest) to right (most recent).\n"
                "The goal is to quickly identify potential risk signs from the images.\n"
                f"Types of risk include {sub_folder}.\n"
                "The analysis must assume that there may be risk types situations, rather than always normal type situations.\n"
                "Even the smallest abnormality signals must be recognized.\n")
    


    if llm_option == 0 : 
        if task == 'future' :
            if single_seq_frame_use :
                prompt2 = "Look at the image and predict the following future situation in three cases.\n"
            else :
                prompt2 = "Look at the past images from the left and predict the following future situation in three cases.\n"

            prompt3 = persona_block + prompt2 + prompt
            messages.append({"type": "text", "text": prompt3})
            messages.append({"type": "image_url", "image_url": {"url": img_base64_url}})
        elif task == 'past' :
            prompt = "Summarize the past image on the left.\n"
            prompt = persona_block + prompt
            messages.append({"type": "text", "text": prompt})
            messages.append({"type": "image_url", "image_url": {"url": img_base64_url}})
        elif task == 'end' :
            prompt = prompt
        else : 
            prompt2 = f"{task} has {sub_folder}, etc. Please note.\n"
            if task == 'normal' : prompt2 = f"No {task} has {sub_folder}, etc. Please note.\n"
            messages.append({"type": "text", "text": prompt})
            messages.append({"type": "text", "text": prompt2})
            #print(messages)
            messages.append({"type": "image_url", "image_url": {"url": img_base64_url}})
    elif llm_option == 1 : 
        prompt2 = (
            "Above are the opinions of three LLMs who reviewed the frame and responded.\n" 
            "Based on this, please summarize your final decision about the image below in one word.\n"
            "If both normal and swoon give their opinions in 3 llms, it is likely to be normal.\n"
            "**Start your output with one-word: 'normal' or 'assault' or 'swoon' or 'robbery'.**\n"
            "**After your one-word answer, briefly explain your reasoning.**\n"
        )
 
        messages.append({"type": "text", "text": prompt})
        messages.append({"type": "text", "text": prompt2})
        #print(messages)
        messages.append({"type": "image_url", "image_url": {"url": img_base64_url}})

    # --- retry 로직 추가 ---
    response = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                messages=[{"role": "user", "content": messages}]
            )
            break 
        except openai.InternalServerError:
            print(f"[경고] GPT-4o 호출 실패(500 error). {attempt+1}번째 재시도...")
            time.sleep(2 * (attempt + 1))  # 대기 후 재시도
    if response is None:
        raise Exception("GPT-4o API 호출이 3번 모두 실패했습니다.")

    return response.choices[0].message.content


def generate_caption_with_internvl3_sequence(image_tensors, prompt, single_seq_frame_use, llm_option, task, sub_folder):
    concat_img = image_tensors
    if single_seq_frame_use :
        persona_block = ("You are a safety surveillance AI system that views CCTV images and analyzes them for risks.\n"
                "The goal is to quickly identify potential risk signs from the images.\n"
                f"Types of risk include {sub_folder}.\n"
                "The analysis must assume that there may be risk types situations, rather than always normal type situations.\n"
                "Even the smallest abnormality signals must be recognized.\n")
    else : 
        persona_block = (
                "You are a safety surveillance AI system that views 5 consecutive CCTV images and analyzes them for risks.\n"
                "The images are arranged from left (earliest) to right (most recent).\n"
                "The goal is to quickly identify potential risk signs from the images.\n"
                f"Types of risk include {sub_folder}.\n"
                "The analysis must assume that there may be risk types situations, rather than always normal type situations.\n"
                "Even the smallest abnormality signals must be recognized.\n")

    if llm_option == 0 : 
        if task == 'future' :
            if single_seq_frame_use :
                prompt2 = "Look at the image and predict the following future situation in three cases.\n"
            else :
                prompt2 = "Look at the past images from the left and predict the following future situation in three cases.\n"
            prompt = persona_block + prompt2 + prompt
        elif task == 'past' : 
            prompt = "Summarize the past image on the left.\n"
            prompt = persona_block + prompt
        elif task == 'end' :
            prompt = prompt
        else : 
            prompt2 = f"{task} has {sub_folder}, etc. Please note.\n"
            if task == 'normal' : prompt2 = f"No {task} has {sub_folder}, etc. Please note.\n"
            prompt = prompt + prompt2
    elif llm_option == 1 : 
        prompt = "Here is the full analysis of the risk image. Based on this full analysis and the image, make a final judgment. The final judgment is made by considering the first word that is said for each task in the above analysis as the opinion(one-word) or 'normal'. If there is no risk signal at all, it is considered 'normal'. Select one or more words and briefly explain why."
    
    generation_config = dict(max_new_tokens=1024, do_sample=False, pad_token_id=internvl_tokenizer.eos_token_id)
    pixel_values = preprocess_image(concat_img).to(torch.bfloat16).cuda()
    response, history = internvl_model.chat(internvl_tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    return response


def generate_caption_with_Qwen_sequence(image_tensors, prompt, single_seq_frame_use, llm_option, task, sub_folder):
    concat_img = image_tensors
   

    if single_seq_frame_use :
        persona_block = ("You are a safety surveillance AI system that views CCTV images and analyzes them for risks.\n"
                "The goal is to quickly identify potential risk signs from the images.\n"
                f"Types of risk include {sub_folder}.\n"
                "The analysis must assume that there may be risk types situations, rather than always normal type situations.\n"
                "Even the smallest abnormality signals must be recognized.\n")
    else : 
        persona_block = (
                "You are a safety surveillance AI system that views 5 consecutive CCTV images and analyzes them for risks.\n"
                "The images are arranged from left (earliest) to right (most recent).\n"
                "The goal is to quickly identify potential risk signs from the images.\n"
                f"Types of risk include {sub_folder}.\n"
                "The analysis must assume that there may be risk types situations, rather than always normal type situations.\n"
                "Even the smallest abnormality signals must be recognized.\n")

    if llm_option == 0 :
        if task == 'future' :
            if single_seq_frame_use :
                prompt2 = "Look at the image and predict the following future situation in three cases.\n"
            else :
                prompt2 = "Look at the past images from the left and predict the following future situation in three cases.\n"
            prompt = persona_block + prompt2 + prompt
        elif task == 'past' :
            prompt = "Summarize the past image on the left.\n"
            prompt = persona_block + prompt
        elif task == 'end' :
            prompt = prompt
        else : 
            prompt2 = f"{task} has {sub_folder}, etc. Please note.\n"
            if task == 'normal' : prompt2 = f"No {task} has {sub_folder}, etc. Please note.\n"
            prompt = prompt + prompt2
    elif llm_option == 1 : 
        prompt = "Here is the full analysis of the risk image. Based on this full analysis and the image, make a final judgment. The final judgment is made by considering the first word that is said for each task in the above analysis as the opinion(one-word) or 'normal'. If there is no risk signal at all, it is considered 'normal'. Select one or more words and briefly explain why."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  
                {"type": "text", "text": prompt}
            ],
        }
    ]
    text  = Qwen_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = Qwen_processor(text=[text],  images=[concat_img], return_tensors="pt",return_pixel_values=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    generate_ids = Qwen_model.generate(**inputs, max_new_tokens=256, use_cache=True)
    response = Qwen_processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

    matches = list(re.finditer(r"assistant\s*", response))
    if matches:
        last_start = matches[-1].end()
        response = response[last_start:].strip()

    return response

def generate_prompt(single_seq_frame_use, LLM_Model_Select, task):
    if task != '' :
        if LLM_Model_Select == 2 :
            img = ("<image>\n")
        elif LLM_Model_Select == 3 :
            img = ("<|image|>\n")
        else : 
            img = ("")
        ######################################################### seqeunce version ############################
        task_block_seq = (
            "You are an AI system analyzing 5 consecutive cctv images.\n"
            "The images are arranged from left (earliest) to right (most recent).\n"
        )
        output_format_block_seq = (
            f"**Start output with one of the words '{task}' or 'No {task}' for the most recent image.**\n"
            "**After your answer, briefly explain your reasoning.**\n"
        )
        #####################################################################################################
    
        ######################################################### single version ############################
        task_block_single = (
            "You are an AI system analyzing cctv images.\n"
        )
        output_format_block_single = (
            f"**Start your output with one-word: either '{task}' or 'No {task}'.**\n"
            "**After your one-word answer, briefly explain your reasoning.**\n"
        )
        #####################################################################################################
        if task != 'normal' : 
            task_block_seq = ("") 
            task_block_single = ("")
        if single_seq_frame_use : 
            full_prompt = (
                img
                + task_block_single
                + output_format_block_single
            )
        else : 
            full_prompt = (
                img
                + task_block_seq
                + output_format_block_seq
            )   
    if task == 'end' :
        full_prompt = 'End the conversation.'
    return full_prompt
#######################################################################################################
# fewshot 저장 폴더 경로 
if single_seq_frame_use : 
    fewshot_root = os.path.join(image_dir, "fewshot_single_image")
    os.makedirs(fewshot_root, exist_ok=True)
else : 
    fewshot_root = os.path.join(image_dir, "fewshot_sequence_image")
    os.makedirs(fewshot_root, exist_ok=True)
    
label_folders = sorted([
    name for name in os.listdir(image_dir)
    if os.path.isdir(os.path.join(image_dir, name)) and "fewshot" not in name.lower()
])
#################################################################################
tasks_ordered = [] 
if 'normal' in label_folders: tasks_ordered.append('normal') 
middle_tasks = [task for task in label_folders if task not in ('normal', 'others')] 
tasks_ordered.extend(middle_tasks)
if 'others' in label_folders: tasks_ordered.append('others')
label_folders = tasks_ordered
#################################################################################
label_to_id = {label: idx for idx, label in enumerate(label_folders, start=0)}
num_classes = len(label_to_id)  
print("label_to_id:", label_to_id)
######################### sub folder list #######################################
subfolder_dict = {}
for label in label_folders:
    label_path = os.path.join(image_dir, label)
    if os.path.isdir(label_path):
        subfolders = [
            name for name in os.listdir(label_path)
            if os.path.isdir(os.path.join(label_path, name))
        ]
        subfolder_dict[label] = subfolders
all_video_paths = []
for label, subfolders in subfolder_dict.items():
    all_video_paths.append(subfolders)

all_video_paths = [item for sublist in all_video_paths for item in sublist]    
#####################################################################################
id_to_label = {v: k for k, v in label_to_id.items()}
min_id = min(id_to_label.keys())
max_id = max(id_to_label.keys())
target_names = [id_to_label[i] for i in range(min_id, max_id + 1)]

# one-hot
def decode_multilabel(arr):
    labels = []
    for row in arr:
        if row.sum() == 0:  
            labels.append("temp")
        else:
            row_labels = []
            for i, v in enumerate(row):
                if v == 1:
                    if i >= len(target_names):
                        row_labels.append("temp")
                    else:
                        row_labels.append(target_names[i])
            labels.append(",".join(row_labels))  
    return labels
#######################################################################################
final_labels_report = []
final_predicted_label_report_gpt4o  = []
final_predicted_label_report_internvl  = []
final_predicted_label_report_qwen = []
final_predicted_label_report_multi = []
final_predicted_label_report_multi2 = []
final_predicted_label_report_multi3 = []
final_predicted_label_report_multi4 = []
final_predicted_label_report = []
misclassified_explanations = []
save_caption1 = []
save_caption2 = []
final_true_label = []
final_predicted_label = []
final_predicted_label_gpt4o = []
final_predicted_label_internvl = []
final_predicted_label_qwen = []
final_predicted_label_multi = []
final_predicted_label_multi2 = []
final_predicted_label_multi3 = []
final_predicted_label_multi4 = []
concat_img_list = []
#######################################################################################
image_list = []
for label_name in label_folders:
    folder_path = os.path.join(image_dir, label_name)
    if not os.path.exists(folder_path):
        print(f"[경고] 폴더 없음: {folder_path}")
        continue

    for img_file in sorted(os.listdir(folder_path)):
        if img_file.lower().endswith((".jpg")):
            img_path = os.path.join(folder_path, img_file)
            image_list.append((img_path, label_to_id[label_name]))

print(f"총 {len(image_list)}개 이미지 발견")
#######################################################################################
with torch.no_grad():
    for img_path, class_id in tqdm(image_list, desc="Processing saved images"):
        gt_label = id_to_label[class_id]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[오류] 이미지 로드 실패: {img_path}, {e}")
            continue

        seq_frames = image 
        seq_frames2 = seq_frames
        
        caption = []    
        caption2 = []
        prompt_temp = []
        final_caption = []
        final_binary_multi_labels1 = []
        final_binary_multi_labels2 = []
        final_binary_multi_labels3 = []
        final_binary_multi_labels4 = []
        final_binary_multi_labels5 = []
        final_binary_multi_labels6 = []
        final_binary_multi_labels7 = []
        final_binary_multi_labels8 = []

        if past_context_analyzing_use :
            task = 'past'
            prompt = ''
            sub_folder = ', '.join(all_video_paths)
            if LLM_Model_Select_multi_label_option_pass_context_model == 1 : 
                generated_caption = generate_caption_with_gpt4o_sequence(seq_frames2, prompt, single_seq_frame_use, 0, task, sub_folder)
            elif LLM_Model_Select_multi_label_option_pass_context_model == 2 : 
                generated_caption = generate_caption_with_internvl3_sequence(seq_frames2, prompt, single_seq_frame_use, 0, task, sub_folder)
            elif LLM_Model_Select_multi_label_option_pass_context_model == 3 :
                generated_caption = generate_caption_with_Qwen_sequence(seq_frames2, prompt, single_seq_frame_use, 0, task, sub_folder)
            
            prompt_temp.append("Below is llm's summary of his thoughts on the past. Please refer to it when classifying the present.\n") 
            prompt_temp.append(generated_caption)
            prompt_temp.append("\n")

        if future_context_analyzing_use :
            task = 'future'
            prompt = ''.join(prompt_temp)
            sub_folder = ', '.join(all_video_paths)
            if LLM_Model_Select_multi_label_option_pass_context_model == 1 : 
                generated_caption = generate_caption_with_gpt4o_sequence(seq_frames2, prompt, single_seq_frame_use, 0, task, sub_folder)
            elif LLM_Model_Select_multi_label_option_pass_context_model == 2 : 
                generated_caption = generate_caption_with_internvl3_sequence(seq_frames2, prompt, single_seq_frame_use, 0, task, sub_folder)
            elif LLM_Model_Select_multi_label_option_pass_context_model == 3 :
                generated_caption = generate_caption_with_Qwen_sequence(seq_frames2, prompt, single_seq_frame_use, 0, task, sub_folder)
            
            prompt_temp.append("Below is llm's opinion, which predicts the future based on past context. Please refer to it when classifying the present.\n") 
            prompt_temp.append(generated_caption)
            prompt_temp.append("\n")

        caption_str = ''.join(prompt_temp) 
        label_vector_temp = [0] * len(label_to_id)
        label_vector_temp1 = [0] * len(label_to_id)
        label_vector_temp2 = [0] * len(label_to_id)
        label_vector_temp3 = [0] * len(label_to_id)
        label_vector_temp4 = [0] * len(label_to_id)
        label_vector_temp5 = [0] * len(label_to_id)
        label_vector_temp6 = [0] * len(label_to_id)
        label_vector_temp7 = [0] * len(label_to_id)
        temp = 0

        for task in label_folders + ['end']: 
            ############# prompt 
            prompt = generate_prompt(True, 3, task)
            #print(prompt)
            if task != 'end' :
                if task == 'normal' : sub_folder = ', '.join(all_video_paths)
                else : sub_folder = ', '.join(subfolder_dict[task])
                prompt = prompt + caption_str
            if not multi_llm_decision_use :
                ############# LLM 
                if LLM_Model_Select_multi_label_option_pass_context_model == 1 : 
                    generated_caption = generate_caption_with_gpt4o_sequence(seq_frames, prompt, True, 0, task, sub_folder)
                elif LLM_Model_Select_multi_label_option_pass_context_model == 2 : 
                    generated_caption = generate_caption_with_internvl3_sequence(seq_frames, prompt, True, 0, task, sub_folder)
                elif LLM_Model_Select_multi_label_option_pass_context_model == 3 :
                    generated_caption = generate_caption_with_Qwen_sequence(seq_frames, prompt, True, 0, task, sub_folder)
                final_caption.append(generated_caption)
            else :
                import time
                start = time.time()
                generated_caption = f"\nBelow is gpt4o, internvl, qwen answer of {task}.\n"
                generated_caption += "Below is gpt4o's answer.\n"
                generated_caption_gpt4o = generate_caption_with_gpt4o_sequence(seq_frames, prompt, True, 0, task, sub_folder)
                generated_caption += generated_caption_gpt4o
                end = time.time()
                print(f"gpt4o 실행 시간: {end - start:.2f}초")
                start = time.time()
                generated_caption += "\nBelow is internvl's answer.\n"  
                generated_caption_internvl = generate_caption_with_internvl3_sequence(seq_frames, prompt, True, 0, task, sub_folder)
                generated_caption += generated_caption_internvl
                end = time.time()
                print(f"internvl 실행 시간: {end - start:.2f}초")
                start = time.time()
                generated_caption += "\nBelow is qwen's answer.\n"
                generated_caption_qwen = generate_caption_with_Qwen_sequence(seq_frames, prompt, True, 0, task, sub_folder)
                generated_caption += generated_caption_qwen
                end = time.time()
                print(f"qwen 실행 시간: {end - start:.2f}초")
                final_caption.append(generated_caption)


                ###### multi llm 
                if task == 'end' : break
                first_line_gpt4o = ''.join(generated_caption_gpt4o).splitlines()[0].lower()
                first_line_internvl = ''.join(generated_caption_internvl).splitlines()[0].lower()
                first_line_qwen = ''.join(generated_caption_qwen).splitlines()[0].lower()
                
                if not re.search(rf"\bno\s+{re.escape(task)}\b", first_line_gpt4o) and re.search(rf"\b{re.escape(task)}\b", first_line_gpt4o) : 
                    for label, idx in label_to_id.items():
                        if re.search(rf"\b{re.escape(label)}\b", first_line_gpt4o) : 
                            label_vector_temp1[idx] = 1   
                            label_vector_temp4[idx] = 1  
                if not re.search(rf"\bno\s+{re.escape(task)}\b", first_line_internvl) and re.search(rf"\b{re.escape(task)}\b", first_line_internvl) : 
                    for label, idx in label_to_id.items():
                        if re.search(rf"\b{re.escape(label)}\b", first_line_internvl) : 
                            label_vector_temp2[idx] = 1 
                            label_vector_temp4[idx] = 1
                if not re.search(rf"\bno\s+{re.escape(task)}\b", first_line_qwen) and re.search(rf"\b{re.escape(task)}\b", first_line_qwen) : 
                    for label, idx in label_to_id.items():
                        if re.search(rf"\b{re.escape(label)}\b", first_line_qwen) : 
                            label_vector_temp3[idx] = 1 
                            label_vector_temp4[idx] = 1

                ###### multi llm에 대한 최종 결과 정리 부분
                first_lines = [
                    ''.join(generated_caption_gpt4o).splitlines()[0].lower(),
                    ''.join(generated_caption_internvl).splitlines()[0].lower(),
                    ''.join(generated_caption_qwen).splitlines()[0].lower()
                ]



                if (task == 'normal') and sum(
                    1 for line in first_lines 
                    if re.search(rf"\b{re.escape(task)}\b", line) 
                    and not re.search(rf"\bno\s+{re.escape(task)}\b", line)
                ) >= 2:
                    for label, idx in label_to_id.items():
                        if label == task:   
                            label_vector_temp5[idx] = 1

                if (task == 'normal') and sum(
                    1 for line in first_lines 
                    if re.search(rf"\b{re.escape(task)}\b", line) 
                    and not re.search(rf"\bno\s+{re.escape(task)}\b", line)
                ) >= 1:
                    for label, idx in label_to_id.items():
                        if label == task:  
                            label_vector_temp6[idx] = 1

                if (task == 'normal') and all(re.search(rf"\b{re.escape(task)}\b", line) and not re.search(rf"\bno\s+{re.escape(task)}\b", line) 
                    for line in first_lines):
                
                    for label, idx in label_to_id.items():
                        if label == task:   
                            label_vector_temp4[idx] = 1
                else:
                    temp = 1    
                    continue
                
            ###### 이부분은 multi llm 아닐때 
            if task == 'end' : break
            final_caption_str_temp = ''.join(generated_caption)
            first_line = final_caption_str_temp.splitlines()[0].lower()  
            if re.search(rf"\bno\s+{re.escape(task)}\b", first_line) : 
                continue
            if re.search(rf"\b{re.escape(task)}\b", first_line) : 
                for label, idx in label_to_id.items():
                    if re.search(rf"\b{re.escape(label)}\b", first_line) : 
                        label_vector_temp[idx] = 1 
        #########################################################################
        final_captions = ''.join(final_caption)
        generated_caption_gpt4o_mix = generate_caption_with_gpt4o_sequence(seq_frames, final_captions, True, 1, task, sub_folder)
        
        first_line_mix = ''.join(generated_caption_gpt4o_mix).splitlines()[0].lower()
        for task in label_folders : 
            if not re.search(rf"\bno\s+{re.escape(task)}\b", first_line_mix) and re.search(rf"\b{re.escape(task)}\b", first_line_mix) : 
                        for label, idx in label_to_id.items():
                            if re.search(rf"\b{re.escape(label)}\b", first_line_mix) : 
                                label_vector_temp7[idx] = 1   
        ############################################################################## 
        if sum(label_vector_temp) == 0 and "normal" in label_to_id : label_vector_temp[label_to_id["normal"]] = 1
        if sum(label_vector_temp1) == 0 and "normal" in label_to_id : label_vector_temp1[label_to_id["normal"]] = 1
        if sum(label_vector_temp2) == 0 and "normal" in label_to_id : label_vector_temp2[label_to_id["normal"]] = 1
        if sum(label_vector_temp3) == 0 and "normal" in label_to_id : label_vector_temp3[label_to_id["normal"]] = 1
        if sum(label_vector_temp4) == 0 and "normal" in label_to_id : label_vector_temp4[label_to_id["normal"]] = 1
        if sum(label_vector_temp7) == 0 and "normal" in label_to_id : label_vector_temp7[label_to_id["normal"]] = 1
        #################################################################################
        final_binary_multi_labels1.append(label_vector_temp)   
        final_binary_multi_labels1 = np.array(final_binary_multi_labels1).squeeze()
        final_predicted_label.append(final_binary_multi_labels1)
        final_true_label.append(gt_label)

        if multi_llm_decision_use : 
            final_binary_multi_labels2.append(label_vector_temp1)   
            final_binary_multi_labels2 = np.array(final_binary_multi_labels2).squeeze()
            final_predicted_label_gpt4o.append(final_binary_multi_labels2)
            final_binary_multi_labels3.append(label_vector_temp2)   
            final_binary_multi_labels3 = np.array(final_binary_multi_labels3).squeeze()
            final_predicted_label_internvl.append(final_binary_multi_labels3)
            final_binary_multi_labels4.append(label_vector_temp3)   
            final_binary_multi_labels4 = np.array(final_binary_multi_labels4).squeeze()
            final_predicted_label_qwen.append(final_binary_multi_labels4)
            final_binary_multi_labels5.append(label_vector_temp4)   
            final_binary_multi_labels5 = np.array(final_binary_multi_labels5).squeeze()
            final_predicted_label_multi.append(final_binary_multi_labels5)
            final_binary_multi_labels6.append(label_vector_temp5)   
            final_binary_multi_labels6 = np.array(final_binary_multi_labels6).squeeze()
            final_predicted_label_multi2.append(final_binary_multi_labels6)
            final_binary_multi_labels7.append(label_vector_temp6)   
            final_binary_multi_labels7 = np.array(final_binary_multi_labels7).squeeze()
            final_predicted_label_multi3.append(final_binary_multi_labels7)
            final_binary_multi_labels8.append(label_vector_temp7)   
            final_binary_multi_labels8 = np.array(final_binary_multi_labels8).squeeze()
            final_predicted_label_multi4.append(final_binary_multi_labels8)
        #################################################################################
        predicted_label = final_binary_multi_labels1
        if multi_llm_decision_use : predicted_label = final_binary_multi_labels5 

        save_caption1.append(final_caption)
        save_caption2.append(caption_str)
        
        filename = os.path.basename(img_path)
        name_only = os.path.splitext(filename)[0]
        fewshot_root_detail = os.path.join(fewshot_root, name_only)
        os.makedirs(fewshot_root_detail, exist_ok=True)

        if not any(p and g for p, g in zip(predicted_label, gt_label)):
            save_name = f"frame_{i}_Pred{predicted_label}_GT{gt_label}.jpg"
            save_path = os.path.join(fewshot_root_detail, save_name)
            to_pil = transforms.ToPILImage()
            pil_images = [to_pil(tensor) for tensor in seq_frames]
            
            if single_seq_frame_use :
                concat_img = to_pil(seq_frames[-1])
            else : 
                concat_img = concat_images_horizontally(pil_images)
            concat_img.save(save_path)
            
            frame_number = i
            misclassified_explanations.append((frame_number, final_caption))
        ##################################################################################################################

# 결과 설명 저장
explanation_save_path = os.path.join(fewshot_root_detail, "misclassified_explanations.txt")
with open(explanation_save_path, "w", encoding="utf-8") as f:
    for idx, text in misclassified_explanations:
        f.write(f"Frame {idx}: {text}\n")    
print("misclassified_explanations save end")

# 캡션 텍스트 저장
txt_save_path = video_path.replace(".mp4", "_multi_label_caption.txt")
with open(txt_save_path, "w", encoding="utf-8") as f:
    for i, (cap_text, label) in enumerate(zip(save_caption1, final_true_label)):
        f.write(f"[Frame {i}] {cap_text}\n")
        f.write(f"[GT Label] {label}\n")
# 캡션 텍스트 저장
txt_save_path = video_path.replace(".mp4", "_past_future_caption.txt")
with open(txt_save_path, "w", encoding="utf-8") as f:
    for i, (cap_text, label) in enumerate(zip(save_caption2, final_true_label)):
        f.write(f"[Frame {i}] {cap_text}\n")
        f.write(f"[GT Label] {label}\n")

image_save_path = video_path.replace(".mp4", "")
os.makedirs(image_save_path, exist_ok=True)

for i, frame_img in enumerate(concat_img_list):
    frame_filename = f"frame_{i:05d}.jpg"
    frame_path = os.path.join(image_save_path, frame_filename)
    frame_img.save(frame_path)  

final_labels_report.append(final_true_label)
final_predicted_label_report.append(final_predicted_label)
if multi_llm_decision_use :
    final_predicted_label_report_gpt4o.append(final_predicted_label_gpt4o)
    final_predicted_label_report_internvl.append(final_predicted_label_internvl)
    final_predicted_label_report_qwen.append(final_predicted_label_qwen)
    final_predicted_label_report_multi.append(final_predicted_label_multi)
    final_predicted_label_report_multi2.append(final_predicted_label_multi2)
    final_predicted_label_report_multi3.append(final_predicted_label_multi3)
    final_predicted_label_report_multi4.append(final_predicted_label_multi4)

if multi_llm_decision_use :

    y_true_temp = np.vstack([ np.stack(sublist) for sublist in final_true_label ])
    y_pred_gpt4o_temp = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_gpt4o ])
    y_pred_internvl_temp = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_internvl ])
    y_pred_qwen_temp = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_qwen ])
    y_pred_multi_temp = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_multi ])
    y_pred_multi2_temp = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_multi2 ])
    y_pred_multi3_temp = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_multi3 ])
    y_pred_multi4_temp = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_multi4 ])


    df_temp = pd.DataFrame({
        "y_true": decode_multilabel(y_true_temp),
        "y_pred_gpt4o": decode_multilabel(y_pred_gpt4o_temp),
        "y_pred_internvl": decode_multilabel(y_pred_internvl_temp),
        "y_pred_qwen": decode_multilabel(y_pred_qwen_temp),
        "y_pred_multi": decode_multilabel(y_pred_multi_temp),
        "y_pred_multi2": decode_multilabel(y_pred_multi2_temp),
        "y_pred_multi3": decode_multilabel(y_pred_multi3_temp),
        "y_pred_multi_mix": decode_multilabel(y_pred_multi4_temp),
    })

    image_dir = os.path.dirname(video_path) 
    video_name = os.path.splitext(os.path.basename(video_path))[0] 
    csv_save_path = os.path.join(image_dir, f"{video_name}_results.csv") 
    df_temp.to_csv(csv_save_path, index=False) 
    print(f"[저장 완료] {csv_save_path}")
####################################################################################################################
if not multi_llm_decision_use :
    # Multi Calss Report 관련 코드
    y_true = np.vstack([ np.stack(sublist) for sublist in final_labels_report ])
    y_pred = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_report ])

    # 변환
    binary_y_true = convert_to_binary_label(y_true, label_to_id)
    binary_y_pred = convert_to_binary_label(y_pred, label_to_id)

    # classification report 생성(1)
    print("=== Binary Classification Report (normal vs risk) ===")
    print(classification_report(binary_y_true, binary_y_pred, target_names=["normal", "risk"], zero_division=0))

    # classification report 생성(2)
    print("=== Multi label Classification Report (Sub Class) ===")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
####################################################################################################################
if multi_llm_decision_use :
    # Multi Calss Report 관련 코드
    y_true = np.vstack([ np.stack(sublist) for sublist in final_labels_report ])
    y_pred_gpt4o = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_report_gpt4o ])
    y_pred_internvl = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_report_internvl ])
    y_pred_qwen = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_report_qwen ])
    y_pred_multi = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_report_multi ])
    y_pred_multi2 = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_report_multi2 ])
    y_pred_multi3 = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_report_multi3 ])
    y_pred_multi4 = np.vstack([ np.stack(sublist) for sublist in final_predicted_label_report_multi4 ])

    # 변환
    binary_y_true = convert_to_binary_label(y_true, label_to_id)
    binary_y_pred_gpt4o = convert_to_binary_label(y_pred_gpt4o, label_to_id)
    binary_y_pred_internvl = convert_to_binary_label(y_pred_internvl, label_to_id)
    binary_y_pred_qwen = convert_to_binary_label(y_pred_qwen, label_to_id)
    binary_y_pred_multi = convert_to_binary_label(y_pred_multi, label_to_id)
    binary_y_pred_multi2 = convert_to_binary_label(y_pred_multi2, label_to_id)
    binary_y_pred_multi3 = convert_to_binary_label(y_pred_multi3, label_to_id)
    binary_y_pred_multi4 = convert_to_binary_label(y_pred_multi4, label_to_id)


    # classification report 생성(1)
    print("=== Milti LLM - Binary Classification Report_gpt4o (normal vs risk) ===")
    print(classification_report(binary_y_true, binary_y_pred_gpt4o, target_names=["normal", "risk"], zero_division=0))

    # classification report 생성(2)
    print("=== Milti LLM - Multi label Classification Report_gpt4o (Sub Class) ===")
    print(classification_report(y_true, y_pred_gpt4o, target_names=target_names, zero_division=0))
    acc = accuracy_score(y_true, y_pred_gpt4o)
    print(f"accuracy                           {acc:.2f}")

    # classification report 생성(3)
    print("=== Milti LLM - Binary Classification Report_internvl (normal vs risk) ===")
    print(classification_report(binary_y_true, binary_y_pred_internvl, target_names=["normal", "risk"], zero_division=0))

    # classification report 생성(4)
    print("=== Milti LLM - Multi label Classification Report_internvl (Sub Class) ===")
    print(classification_report(y_true, y_pred_internvl, target_names=target_names, zero_division=0))
    acc = accuracy_score(y_true, y_pred_internvl)
    print(f"accuracy                           {acc:.2f}")

    # classification report 생성(5)
    print("=== Milti LLM - Binary Classification Report_qwen (normal vs risk) ===")
    print(classification_report(binary_y_true, binary_y_pred_qwen, target_names=["normal", "risk"], zero_division=0))

    # classification report 생성(6)
    print("=== Milti LLM - Multi label Classification Report_qwen (Sub Class) ===")
    print(classification_report(y_true, y_pred_qwen, target_names=target_names, zero_division=0))
    acc = accuracy_score(y_true, y_pred_qwen)
    print(f"accuracy                           {acc:.2f}")

    # classification report 생성(7)
    print("=== Milti LLM - Binary Classification Report_multi_final (normal vs risk) ===")
    print(classification_report(binary_y_true, binary_y_pred_multi, target_names=["normal", "risk"], zero_division=0))

    # classification report 생성(8)
    print("=== Milti LLM - Multi label Classification Report_multi_final (Sub Class) ===")
    print(classification_report(y_true, y_pred_multi, target_names=target_names, zero_division=0))
    acc = accuracy_score(y_true, y_pred_multi)
    print(f"accuracy                           {acc:.2f}")

    # classification report 생성(7)
    print("=== Milti LLM - Binary Classification Report_multi_final2 (normal vs risk) ===")
    print(classification_report(binary_y_true, binary_y_pred_multi2, target_names=["normal", "risk"], zero_division=0))

    # classification report 생성(7)
    print("=== Milti LLM - Binary Classification Report_multi_final3 (normal vs risk) ===")
    print(classification_report(binary_y_true, binary_y_pred_multi3, target_names=["normal", "risk"], zero_division=0))

    # classification report 생성(7)
    print("=== Milti LLM - Binary Classification Report_multi_mix (normal vs risk) ===")
    print(classification_report(binary_y_true, binary_y_pred_multi4, target_names=["normal", "risk"], zero_division=0))