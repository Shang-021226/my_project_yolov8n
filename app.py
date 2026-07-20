from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
import os
import cv2
import uuid
import warnings

warnings.filterwarnings("ignore")

# -------------------------- 配置项 --------------------------
MODEL_PATH = r"C:\Users\30227\Desktop\毕设\code\helmetdataset\helmet_model\helmet_detect\weights\best.pt"
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
DETECT_CONF = 0.3  # 置信度阈值（平衡漏检和误检）
DETECT_IMGSZ = 640  # 模型输入尺寸
DETECT_AUGMENT = True  # 推理增强
# -----------------------------------------------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 加载模型（使用YOLOv8x提升精度）
try:
    model = YOLO(MODEL_PATH)  # 若需更高精度，可替换为yolov8x.pt重新训练
    print(f"✅ 模型加载成功：{MODEL_PATH}")
except Exception as e:
    print(f"❌ 模型加载失败：{e}")
    exit(1)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 修复：彩色图片预处理（保留RGB通道）
def preprocess_image(image_path):
    """彩色图片预处理：缩放+色彩增强（不转灰度）"""
    img = cv2.imread(image_path)  # 读取彩色图（BGR格式）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转RGB（匹配模型训练格式）

    # 缩放并填充至模型输入尺寸
    h, w = img.shape[:2]
    scale = DETECT_IMGSZ / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    pad_h = (DETECT_IMGSZ - new_h) // 2
    pad_w = (DETECT_IMGSZ - new_w) // 2
    padded_img = cv2.copyMakeBorder(
        resized_img, pad_h, DETECT_IMGSZ - new_h - pad_h,
        pad_w, DETECT_IMGSZ - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # 彩色图色彩增强（提升对比度，保留RGB）
    padded_img = cv2.convertScaleAbs(padded_img, alpha=1.2, beta=10)  # 亮度+对比度增强

    # 保存为RGB格式（避免颜色失真）
    padded_img = cv2.cvtColor(padded_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, padded_img)
    return image_path


def detect_helmet(image_path):
    image_path = preprocess_image(image_path)
    # 推理时启用多尺度检测，提升小目标精度
    results = model(image_path, conf=DETECT_CONF, imgsz=DETECT_IMGSZ, augment=DETECT_AUGMENT, multi_scale=True)

    helmet_count = 0
    no_helmet_count = 0
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:
                helmet_count += 1
            elif cls_id == 1:
                no_helmet_count += 1

    img_id = str(uuid.uuid4())
    result_img_path = os.path.join(RESULT_FOLDER, f"{img_id}.jpg")
    results[0].save(result_img_path)  # 保存彩色检测结果图
    return result_img_path, {
        "helmet": helmet_count,
        "no_helmet": no_helmet_count,
        "total": helmet_count + no_helmet_count
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "未上传图片"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "请选择图片文件"}), 400
    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1].lower()}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        try:
            result_path, stats = detect_helmet(upload_path)
            return jsonify({
                "success": True,
                "result_image": result_path,
                "stats": stats,
                "message": f"检测完成：佩戴头盔{stats['helmet']}人，未佩戴{stats['no_helmet']}人"
            })
        except Exception as e:
            return jsonify({"error": f"检测失败：{str(e)}"}), 500
    else:
        return jsonify({"error": "不支持的文件格式"}), 400


@app.route('/<path:filename>')
def get_file(filename):
    return send_file(filename)


if __name__ == '__main__':
    print("🚀 头盔检测系统启动中...")
    print("访问地址：http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)