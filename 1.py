from ultralytics import YOLO
import os
import cv2
import warnings

# 屏蔽版本更新提示
warnings.filterwarnings("ignore", message=".*new version of ultralytics.*")

# -------------------------- 配置项 --------------------------
DATA_YAML = r"C:\Users\30227\Desktop\毕设\code\helmetdataset\helmetdataset\data.yaml"
MODEL_SAVE_ROOT = r"C:\Users\30227\Desktop\毕设\code\helmetdataset\helmet_model"
LOCAL_WEIGHT_PATH = r"C:\Users\30227\Desktop\毕设\code\yolov8n.pt"
MAX_EPOCHS = 50  # 固定训练50轮


# -----------------------------------------------------------------------------------


def check_local_weight():
    if not os.path.exists(LOCAL_WEIGHT_PATH):
        print(f"⚠️ 未找到本地权重，请下载：https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt")
        exit()
    print("✅ 本地权重已就绪")


def train_yolov8():
    check_local_weight()
    model = YOLO(LOCAL_WEIGHT_PATH)

    # 启动50轮训练
    train_results = model.train(
        data=DATA_YAML,
        epochs=MAX_EPOCHS,
        batch=4,
        imgsz=640,
        project=MODEL_SAVE_ROOT,
        name="helmet_detect",
        device=0,
        workers=2,
        optimizer="SGD",
        pretrained=True,
        verbose=True
    )

    # 提取训练指标（修复KeyError）
    val_map50 = train_results.results_dict.get("metrics/mAP50(B)", 0.0)
    val_box_loss = train_results.results_dict.get("val/box_loss", 1.0)

    # 过拟合检测
    if val_map50 < 0.5 and val_box_loss > 1.0:
        print("⚠️ 提示：模型可能存在过拟合风险")

    # 输出训练结果
    best_model_path = os.path.join(MODEL_SAVE_ROOT, "helmet_detect", "weights", "best.pt")
    print(f"\n✅ 50轮训练完成！")
    print(f"📌 最佳模型路径：{best_model_path}")
    print(f"📊 验证集mAP50：{val_map50:.4f}")
    return best_model_path


def infer_with_model(model_path):
    """模型推理测试"""
    model = YOLO(model_path)
    # 替换为你的测试图片路径
    test_img_path = r"C:\Users\30227\Desktop\毕设\code\helmetdataset\helmetdataset\JPEGImages\test.jpg"
    # 推理（置信度阈值0.5）
    results = model(test_img_path, conf=0.5)

    # 保存推理结果
    save_dir = os.path.join(MODEL_SAVE_ROOT, "infer_results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "detect_result.jpg")
    results[0].save(save_path)

    # 显示推理结果
    img = cv2.imread(save_path)
    cv2.imshow("Helmet Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"\n🔍 推理结果已保存至：{save_path}")


if __name__ == "__main__":
    # 训练完成后，可注释train_yolov8()，直接加载模型推理
    best_model = train_yolov8()
    infer_with_model(best_model)