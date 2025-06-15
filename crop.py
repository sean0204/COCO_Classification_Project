import json, os
from PIL import Image
from tqdm import tqdm

# 設定
coco_json_path = r"D:\interview_project\raw_data\annotations/instances_val2017.json"
image_dir = r"D:\interview_project\raw_data\val2017"
output_base = r"D:\interview_project\picture"
#target_classes = ['toothbrush', 'stop sign', 'mouse', 'frisbee','laptop']
target_classes = ['baseball bat']

# target_classes = ['laptop']
# 建資料夾
os.makedirs(output_base, exist_ok=True)
for cls in target_classes:
    os.makedirs(os.path.join(output_base, cls.replace(" ", "_")), exist_ok=True)

# 讀 JSON
with open(coco_json_path, "r") as f:
    coco = json.load(f)

# 類別對照
cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}

# 建圖ID對檔名對照
img_id_to_info = {img["id"]: img for img in coco["images"]}

# 遍歷 annotations
counter = {cls: 0 for cls in target_classes}

for ann in tqdm(coco["annotations"]):
    cat_id = ann["category_id"]
    cat_name = cat_id_to_name[cat_id]

    if cat_name not in target_classes:
        continue

    # bbox 格式: [x, y, width, height]
    image_id = ann["image_id"]
    bbox = ann["bbox"]
    x, y, w, h = map(int, bbox)

    img_info = img_id_to_info[image_id]
    img_path = os.path.join(image_dir, img_info["file_name"])

    try:
        img = Image.open(img_path).convert("RGB")
        cropped = img.crop((x, y, x + w, y + h))

        save_name = f"{cat_name.replace(' ', '_')}_{counter[cat_name]}.jpg"
        save_path = os.path.join(output_base, cat_name.replace(" ", "_"), save_name)
        cropped.save(save_path)
        counter[cat_name] += 1
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
