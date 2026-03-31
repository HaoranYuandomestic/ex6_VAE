import os
import io
from datasets import load_dataset
from PIL import Image

# =========================
# 配置
# =========================
SAVE_DIR = "wikiart_subset"
TARGET_SIZE_GB = 2.0           # 想保存的大致总大小
JPEG_QUALITY = 90              # 保存质量，越低越省空间
MAX_IMAGES = 50000             # 安全上限，防止极端情况跑太久
START_INDEX = 0                # 断点续存时可改
HF_DATASET_NAME = "huggan/wikiart"

target_bytes = int(TARGET_SIZE_GB * 1024 * 1024 * 1024)
os.makedirs(SAVE_DIR, exist_ok=True)

def get_existing_size(folder):
    total = 0
    count = 0
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            total += os.path.getsize(path)
            count += 1
    return total, count

def main():
    # 统计已有大小，支持断点续存
    saved_bytes, saved_count = get_existing_size(SAVE_DIR)
    print(f"当前文件夹已有 {saved_count} 张图片，已占用 {saved_bytes / (1024**3):.2f} GB")

    if saved_bytes >= target_bytes:
        print("已经达到目标大小，无需继续下载。")
        return

    # 核心：streaming=True，避免整包下载
    dataset = load_dataset(
        HF_DATASET_NAME,
        split="train",
        streaming=True
    )

    # 可选：轻量打乱，提高多样性
    dataset = dataset.shuffle(seed=42, buffer_size=1000)

    skipped = 0
    saved_now = 0

    for idx, sample in enumerate(dataset):
        if idx < START_INDEX:
            continue

        if saved_count >= MAX_IMAGES:
            print("达到 MAX_IMAGES 上限，停止。")
            break

        if saved_bytes >= target_bytes:
            print("达到目标大小，停止。")
            break

        try:
            image = sample["image"]

            # 某些情况下 image 可能不是 RGB
            if not isinstance(image, Image.Image):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            else:
                image = image.convert("RGB")

            # 统一缩放，便于训练，也节省空间
            image = image.resize((256, 256))

            save_path = os.path.join(SAVE_DIR, f"wikiart_{saved_count:06d}.jpg")
            image.save(save_path, format="JPEG", quality=JPEG_QUALITY)

            file_size = os.path.getsize(save_path)
            saved_bytes += file_size
            saved_count += 1
            saved_now += 1

            if saved_count % 100 == 0:
                print(
                    f"已保存 {saved_count} 张，"
                    f"当前总大小 {saved_bytes / (1024**3):.2f} GB"
                )

        except Exception as e:
            skipped += 1
            print(f"跳过第 {idx} 条，原因: {e}")
            continue

    print("=" * 50)
    print(f"完成。新保存 {saved_now} 张")
    print(f"总图片数: {saved_count}")
    print(f"总大小: {saved_bytes / (1024**3):.2f} GB")
    print(f"跳过数量: {skipped}")

if __name__ == "__main__":
    main()