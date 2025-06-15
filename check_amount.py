import json
from collections import defaultdict

# 假設 instances_val2017.json 檔案已經在同一個目錄下
# 如果不在，請修改這裡的路徑
file_path = r'D:\interview_project\raw_data\annotations\instances_train2017.json'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{file_path}'。請確保檔案路徑正確。")
    exit()

# 提取所有類別的資訊
categories_info = data['categories']
category_id_to_name = {cat['id']: cat['name'] for cat in categories_info}
category_name_to_id = {cat['name']: cat['id'] for cat in categories_info}

# 統計每個類別的物件實例數量
category_counts = defaultdict(int)
for ann in data['annotations']:
    category_id = ann['category_id']
    category_name = category_id_to_name.get(category_id, "未知類別")
    category_counts[category_name] += 1

# 顯示總類別數量
total_categories = len(categories_info)
print(f"MS COCO 驗證集總共有 {total_categories} 個物件類別。")
print("-" * 50)

# 您已選擇的類別
chosen_categories = ['person', 'car', 'cat', 'dog']
print(f"您已選擇的類別有：{chosen_categories}")
print("-" * 50)

# 顯示您已選擇類別的數量
print("您已選擇類別的物件實例數量：")
for name in chosen_categories:
    count = category_counts.get(name, 0)
    print(f"- {name}: {count} 個實例")
print("-" * 50)

# 排除已選擇的類別，並統計剩餘類別的實例數量
remaining_categories_counts = {}
for name, count in category_counts.items():
    if name not in chosen_categories:
        remaining_categories_counts[name] = count

# 顯示排除後的類別總數
print(f"排除您已選擇的 {len(chosen_categories)} 個類別後，剩餘 {len(remaining_categories_counts)} 個類別。")
print("-" * 50)

# 按照實例數量排序剩餘類別
sorted_remaining_categories = sorted(remaining_categories_counts.items(), key=lambda item: item[1])

print("剩餘類別及其物件實例數量 (從少到多排序)：")
for name, count in sorted_remaining_categories:
    print(f"- {name}: {count} 個實例")

print("-" * 50)

# 找出實例數量最少的幾個類別
if sorted_remaining_categories:
    print("\n實例數量最少的 5 個類別：")
    for i in range(min(5, len(sorted_remaining_categories))):
        name, count = sorted_remaining_categories[i]
        print(f"- {name}: {count} 個實例")
else:
    print("沒有剩餘類別可以分析。")

print("-" * 50)
print("提示：這些統計數據是基於物件實例的數量，而不是包含該類別的圖片數量。")
print("您可以使用這些信息來選擇其他類別進行挑戰，或者理解數據分佈。")
