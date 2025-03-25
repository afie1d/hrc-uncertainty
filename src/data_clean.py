import os
import json

img_dir = '/home/fieldaj1/thesis/data/BinaryScenes/scene_img_abstract_v002_val2017'
question_file = '/home/fieldaj1/thesis/data/BinaryScenes/OpenEnded_abstract_v002_val2017_questions.json'
ans_file = '/home/fieldaj1/thesis/data/BinaryScenes/abstract_v002_val2017_annotations.json'
removed = 0

def get_question(img_id, question_file):
    with open(question_file, 'r') as f:
        data = json.load(f)
        questions = data.get('questions')
        for q in questions:
            if q.get("image_id") == img_id:
                return q.get("question")
        
        return None # couldn't find question for the given image id
    

def get_answer(img_id, ans_file):
    with open(ans_file, 'r') as f:
        data = json.load(f)
        annotations = data.get('annotations')
        for a in annotations:
            if a.get("image_id") == img_id:
                answers = a.get("answers")
                yes_count = 0; no_count = 0
                for answer in answers:
                    if answer.get("answer") == "yes":
                        yes_count += 1
                    else:
                        no_count += 1

                if yes_count > no_count:
                    ans = 'yes'
                    confidence = yes_count / (yes_count + no_count)
                else:
                    ans = 'no'
                    confidence = no_count / (yes_count + no_count)
                
                return ans
        
        return None
    
for img in os.listdir(img_dir):
    img_id = int(img.strip('.png')[-12:].strip('0'))
    q = get_question(img_id, question_file)
    a = get_answer(img_id, ans_file)
    if not(q and a):
        os.remove(os.path.join(img_dir, img))
        removed += 1

print("Number of images removed:", removed)
