import os
import logging
import random
import logging
import jsonlines
import json
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from sat.helpers import print_rank0

def find_all_files(path, suffix=".jpg"):
    target_files = []
    for cur_dir, _, files in os.walk(path, followlinks=True):
        for f in files:
            if f.endswith(suffix):
                target_files.append(os.path.join(cur_dir, f))
    print_rank0(f'find {len(target_files)} files...')
    return target_files

class ItemDataset(Dataset):
    def __init__(self, image_processor, text_processor, args, data_dirs, cross_image_processor=None, **kwargs):
        super().__init__()
        self.data = self.load_data(data_dirs)
        self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor
    
    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        if self.cross_image_processor:
            img_dict.update({'cross': self.cross_image_processor(img)})
        return img_dict
    
    def process_text(self, answer, prompt):
        return self.text_processor(answer, prompt)
    
    # def load_data(self, data_dir):
    #     all_files = find_all_files(data_dir, suffix=".jpg")
    #     print_rank0(f"find {len(all_files)} samples in all...")
    #     return all_files
    # def load_data(self, data_dir):
    #     image_files = find_all_files(data_dir, suffix=".jpg")
    #     data_pairs = []
    #     for img_path in image_files:
    #         base_name = os.path.basename(img_path).split('.')[0]
    #         label_path = os.path.join(os.path.dirname(img_path), f"{base_name}.json")
    #         if os.path.exists(label_path):
    #             data_pairs.append((img_path, label_path))
    #         else:
    #             print_rank0(f"Warning: No label file found for {img_path}")
    #     print_rank0(f"Loaded {len(data_pairs)} image-label pairs")
    #     return data_pairs
    def load_data(self, data_dirs):
      data_pairs = []
      for dir_path in data_dirs:
          image_files = find_all_files(dir_path, suffix=".jpg")
          for img_path in image_files:
              base_name = os.path.basename(img_path).split('.')[0]
              label_path = os.path.join(os.path.dirname(img_path), f"{base_name}.json")
              if os.path.exists(label_path):
                  data_pairs.append((img_path, label_path))
              else:
                  print_rank0(f"Warning: No label file found for {img_path}")
      print_rank0(f"Loaded {len(data_pairs)} image-label pairs")
      return data_pairs

    
    def __len__(self):
        return len(self.data)
    def __len__(self):
        return len(self.data)

    # def __getitem__(self, index):
    #     # data = self.data[index]
    #     data, label_path = self.data[index] 
    #     # Dynamically construct the label path from the image path
    #     label_path = data.replace('.jpg', '.json') 
    #     # img
    #     try:
    #         img = Image.open(data).convert('RGB')
    #     except Exception as e:
    #         print_rank0(e, level=logging.WARNING)
    #         return {}
    #     img_dict = self.process_img(img)
    #     # text
    #     # label = data.split('/')[-1].split('.')[0]
    #     # uni_key = label
    #     # text_dict = self.process_text(label, "CAPTCHA:")
    #     # if text_dict is None:
    #     #     print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {data}", level=logging.WARNING)
    #     #     return {}
    #     # # other attr
    #     try:
    #         with open(label_path, 'r') as f:
    #             label_data = json.load(f)

    #         question = label_data.get("Question", "Default Question") 
    #         answer = label_data.get("Answer", {})
    #         answer_text = ", ".join(f"{key}: {value}" for key, value in answer.items())

    #         text_dict = self.process_text(answer_text, question)
    #         print_rank0(f"Q : {question}, A: {answer_text}")
    #     except Exception as e:
    #         print_rank0(f"Failed to process text for {label_path}: {e}")
    #         return {}
    #     uni_key = os.path.basename(data).split('.')[0]
    #     ret = {**img_dict, **text_dict, "question_id": uni_key}
    #     return ret
    def __getitem__(self, index):
        data, label_path = self.data[index] 
        label_path = data.replace('.jpg', '.json')  # Adjusted to the correct label path

        try:
            with open(label_path, 'r') as f:
                label_data = json.load(f)["QnA"]

            qa_texts = []
            for qa_pair in label_data:
                question = qa_pair.get("Question", "Default Question")
                answer = qa_pair.get("Answer", "Default Answer")

                text_dict = self.process_text(answer, question)
                if text_dict is None:
                    print_rank0(f"Process text failed for question: {question}")
                    continue  # Skip this QA pair if processing fails

                qa_texts.append(text_dict)
                print_rank0(f"Q : {question}, A: {answer}")

            # Combine all qa_texts into a single dictionary, along with any additional data you need
            uni_key = os.path.basename(data).split('.')[0]
            ret = {"qa_texts": qa_texts, "question_id": uni_key}

        except Exception as e:
            print_rank0(f"Failed to process text for {label_path}: {e}", level=logging.ERROR)
            return {}

        return ret  
