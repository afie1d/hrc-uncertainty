import os
import json

img_dir = '/home/fieldaj1/thesis/data/VQA2.0/img'
path = '/home/fieldaj1/thesis/data/VQA2.0'
min_amb_votes = 4
removed = 0

for f_name in os.listdir(path):
    if not f_name.endswith('json'): continue
    json_file = os.path.join(path, f_name)

    with open(json_file) as f:
        data = json.load(f)

        for obj in data:
            ans_diff_labels = obj.get('ans_diff_labels')
            img_path = os.path.join(img_dir, obj.get('image'))
            if os.path.exists(img_path) and ans_diff_labels[4] < min_amb_votes:
                os.remove(img_path)
                print("Removed", img_path)
                removed += 1

print("Removed", removed, " total images")
