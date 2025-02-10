import os
os.environ["WANDB_DISABLED"] = "true"
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from huggingface_hub import login
from sklearn.preprocessing import LabelEncoder
from diffusers import DiffusionPipeline

# Hugging Face 계정으로 로그인
login(token="")

# 기존 데이터셋 로드
dataset = load_dataset("rafaelpadilla/interior-cgi")

# 새 데이터셋 추가
additional_images = ["D:/test/sample_data/image1.jpg", "D:/test/sample_data/image2.jpg"]
additional_labels = ["bathroom", "warm_bed_room"]

# 기존 데이터셋의 'train' 분할에 추가 학습 데이터 병합
train_dataset = dataset['train']

# 새로운 이미지와 레이블을 데이터셋에 추가
new_images = []
new_labels = []

from PIL import Image
for img_path, label in zip(additional_images, additional_labels):
    img = Image.open(img_path).convert("RGB")
    new_images.append(img)
    new_labels.append(label)

# 추가 데이터셋 생성
new_data = Dataset.from_dict({
    "image": new_images,
    "label_name": new_labels
})

# 기존 데이터셋과 병합
train_dataset = concatenate_datasets([train_dataset, new_data])

# 기존 데이터셋에서 모든 레이블 이름 가져오기
all_labels = list(train_dataset['label_name'])

# LabelEncoder를 전체 레이블에 대해 학습
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# 기존 및 새로운 데이터의 label_name을 숫자로 인코딩
encoded_labels = label_encoder.transform(train_dataset['label_name'])
train_dataset = train_dataset.add_column("label", encoded_labels)

# 검증 데이터셋 생성 (train_dataset의 일부로부터 분할)
train_test_split = train_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# 모델 학습 후 저장
model_save_path = "./results"
pipe = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")
pipe.save_pretrained(model_save_path)

# Hugging Face Hub에 업로드
repo_id = "dju-2024-2-test/interior-test"  # 자신의 계정 이름과 모델 이름으로 수정
pipe.push_to_hub(repo_id=repo_id, commit_message="first commit")