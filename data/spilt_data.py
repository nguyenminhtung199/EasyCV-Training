import os
import random
from math import floor
import shutil
import argparse

def split_dataset(input_dir, split_ratio=0.8):
    # Tạo thư mục validation ngang hàng với train
    val_dir = os.path.join(os.path.dirname(input_dir), 'validation')
    os.makedirs(val_dir, exist_ok=True)

    # Lấy danh sách tất cả các nhãn (các thư mục con trong train)
    labels = [label for label in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, label))]
    print(labels)
    for label in labels:
        train_label_dir = os.path.join(input_dir, label)
        val_label_dir = os.path.join(val_dir, label)

        # Tạo thư mục nhãn trong validation
        os.makedirs(val_label_dir, exist_ok=True)

        # Lấy danh sách tất cả các ảnh trong thư mục nhãn
        images = os.listdir(train_label_dir)
        random.shuffle(images)  # Trộn ngẫu nhiên danh sách ảnh

        # Tính số lượng ảnh cho tập train
        train_size = floor(len(images) * split_ratio)

        # Chia ảnh thành 2 tập: train và val
        val_images = images[train_size:]  # Lấy phần còn lại cho validation

        # Di chuyển ảnh vào thư mục validation
        for img in val_images:
            src = os.path.join(train_label_dir, img)
            dst = os.path.join(val_label_dir, img)
            shutil.move(src, dst)

    print(f"Hoàn thành việc tách dữ liệu thành train ({split_ratio*100}%) và validation ({(1-split_ratio)*100}%).")
    print(f"Validation folder created at: {val_dir}")

# Thiết lập argparse để nhận tham số từ dòng lệnh
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chia dữ liệu thành tập train và validation.")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Đường dẫn tới thư mục train chứa các nhãn (subfolders)."
    )
    parser.add_argument(
        "--split_ratio", 
        type=float, 
        default=0.8, 
        help="Tỉ lệ tách dữ liệu cho train (mặc định: 0.8)."
    )

    args = parser.parse_args()
    input_dir = os.path.join(args.input_dir, 'train')
    # Gọi hàm với các tham số đã nhận từ dòng lệnh
    split_dataset(input_dir, args.split_ratio)
