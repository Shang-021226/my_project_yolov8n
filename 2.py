import os
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
from PIL import Image  # 用于读取图片尺寸（处理XML缺少<size>的情况）

# -------------------------- 配置项（根据你的路径/类别修改） --------------------------
RAW_DATA_DIR = r"C:\Users\30227\Desktop\毕设\code\helmetdataset\helmetdataset"
YOLO_DATA_DIR = RAW_DATA_DIR
CLASSES = ["helmet", "no_helmet"]  # 替换为你的真实类别
# -----------------------------------------------------------------------------------

ANNOTATIONS_DIR = os.path.join(RAW_DATA_DIR, "Annotations")
JPEGIMAGES_DIR = os.path.join(RAW_DATA_DIR, "JPEGImages")
IMAGE_SETS_MAIN = os.path.join(RAW_DATA_DIR, "ImageSets", "Main")
YOLO_IMAGES_TRAIN = os.path.join(YOLO_DATA_DIR, "images", "train")
YOLO_IMAGES_VAL = os.path.join(YOLO_DATA_DIR, "images", "val")
YOLO_LABELS_TRAIN = os.path.join(YOLO_DATA_DIR, "labels", "train")
YOLO_LABELS_VAL = os.path.join(YOLO_DATA_DIR, "labels", "val")
DATA_YAML_PATH = os.path.join(YOLO_DATA_DIR, "data.yaml")


def create_dirs():
    """创建YOLO需要的文件夹"""
    dirs = [YOLO_IMAGES_TRAIN, YOLO_IMAGES_VAL, YOLO_LABELS_TRAIN, YOLO_LABELS_VAL]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_image_ids(file_name):
    """从ImageSets/Main读取train/test的图片ID"""
    file_path = os.path.join(IMAGE_SETS_MAIN, file_name)
    image_ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                image_ids.append(line)
    return image_ids


def xml_to_yolo(xml_path, img_width, img_height):
    """将VOC的XML标签转换为YOLO格式的txt内容"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in CLASSES:
            continue
        cls_id = CLASSES.index(cls_name)
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        # 转换为YOLO格式（归一化中心点+宽高）
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return '\n'.join(yolo_lines)


def process_dataset(image_ids, is_train=True):
    """处理指定的图片ID，复制图片+生成YOLO标签（自动补全图片尺寸）"""
    img_dst_dir = YOLO_IMAGES_TRAIN if is_train else YOLO_IMAGES_VAL
    label_dst_dir = YOLO_LABELS_TRAIN if is_train else YOLO_LABELS_VAL
    error_count = 0  # 统计错误文件数

    for img_id in tqdm(image_ids, desc="处理训练集" if is_train else "处理验证集"):
        # 1. 查找并复制图片（兼容多种格式）
        img_ext_list = ['jpg', 'png', 'jpeg']
        img_src_path = None
        for ext in img_ext_list:
            temp_path = os.path.join(JPEGIMAGES_DIR, f"{img_id}.{ext}")
            if os.path.exists(temp_path):
                img_src_path = temp_path
                break
        if not img_src_path:
            print(f"\n警告：图片 {img_id}（格式：{img_ext_list}）不存在，跳过")
            continue
        img_dst_path = os.path.join(img_dst_dir, os.path.basename(img_src_path))
        shutil.copyfile(img_src_path, img_dst_path)

        # 2. 处理标签（自动补全图片尺寸）
        xml_src_path = os.path.join(ANNOTATIONS_DIR, f"{img_id}.xml")
        if not os.path.exists(xml_src_path):
            print(f"\n警告：标签 {img_id}.xml 不存在，跳过")
            continue

        try:
            tree = ET.parse(xml_src_path)
            root = tree.getroot()
            size = root.find('size')
            # 情况1：XML有<size>节点
            if size:
                width_elem = size.find('width')
                height_elem = size.find('height')
                if width_elem and height_elem:
                    img_width = int(width_elem.text)
                    img_height = int(height_elem.text)
                else:
                    # 情况2：<size>节点存在但缺少宽高 → 读取图片本身尺寸
                    with Image.open(img_src_path) as img:
                        img_width, img_height = img.size
            else:
                # 情况3：XML无<size>节点 → 读取图片本身尺寸
                with Image.open(img_src_path) as img:
                    img_width, img_height = img.size
        except Exception as e:
            error_count += 1
            print(f"\n错误：{img_id}.xml 格式异常 → {str(e)}，跳过")
            continue

        # 生成YOLO标签
        yolo_content = xml_to_yolo(xml_src_path, img_width, img_height)
        label_dst_path = os.path.join(label_dst_dir, f"{img_id}.txt")
        with open(label_dst_path, 'w', encoding='utf-8') as f:
            f.write(yolo_content)

    print(f"\n本次处理完成，共跳过 {error_count} 个异常文件")


def generate_data_yaml():
    """生成YOLOv8需要的data.yaml配置文件"""
    yaml_content = f"""
train: {os.path.relpath(YOLO_IMAGES_TRAIN, YOLO_DATA_DIR)}
val: {os.path.relpath(YOLO_IMAGES_VAL, YOLO_DATA_DIR)}

nc: {len(CLASSES)}
names: {CLASSES}
""".strip()
    with open(DATA_YAML_PATH, 'w', encoding='utf-8') as f:
        f.write(yaml_content)


if __name__ == "__main__":
    print("=" * 50)
    print("开始整理YOLOv8数据集")
    print("=" * 50)
    # 1. 创建必要文件夹
    create_dirs()
    # 2. 读取训练/验证集ID
    train_ids = load_image_ids("train.txt")
    val_ids = load_image_ids("test.txt")
    print(f"读取到：训练集{len(train_ids)}张，验证集{len(val_ids)}张")
    # 3. 处理数据集
    process_dataset(train_ids, is_train=True)
    process_dataset(val_ids, is_train=False)
    # 4. 生成配置文件
    generate_data_yaml()
    print("=" * 50)
    print("✅ 数据集整理完成！")
    print(f"📁 图片路径：{YOLO_IMAGES_TRAIN}、{YOLO_IMAGES_VAL}")
    print(f"📁 标签路径：{YOLO_LABELS_TRAIN}、{YOLO_LABELS_VAL}")
    print(f"📄 配置文件：{DATA_YAML_PATH}")
    print("=" * 50)