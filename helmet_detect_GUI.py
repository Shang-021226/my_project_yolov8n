# 1. 模块导入与初始化配置
import os
import sys
import cv2
import numpy as np
import threading
import time
import matplotlib
import pygame
import uuid
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QListWidget,
    QListWidgetItem, QSplitter, QFrame, QStackedWidget, QDialog,
    QLineEdit, QFormLayout, QDialogButtonBox, QGridLayout
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QPalette, QColor
)
from PyQt5.QtCore import (
    QTimer, Qt, QThread, pyqtSignal,
    QPropertyAnimation, QEasingCurve
)

# -------------------------- 系统配置（核心修改）--------------------------
# 模型路径
MODEL_PATH = r"C:\Users\30227\Desktop\毕设\code\helmetdataset\helmet_model\helmet_detect\weights\best.pt"
# 语音告警文件路径
AUDIO_ALERT_PATH = r"C:\Users\30227\Desktop\毕设\code\warn.wav"
# 截图保存文件夹
SCREENSHOT_FOLDER = r"C:\Users\30227\Desktop\毕设\code\unhelmet_screenshots"
# 用户信息存储文件（自动生成）
USER_INFO_FILE = r"C:\Users\30227\Desktop\毕设\code\user_info.json"
# ---------------------------------------------------------------------------------

# -------------------------- 登录/注册窗口实现 --------------------------
class RegisterWindow(QDialog):
    """用户注册窗口"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("用户注册 - 头盔检测系统")
        self.setFixedSize(400, 300)
        self.init_ui()
        # 初始化用户信息文件
        self.init_user_file()

    def init_ui(self):
        """初始化注册界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(20)

        # 标题
        title_label = QLabel("新用户注册")
        title_label.setStyleSheet("font-size: 20pt; font-weight: bold; color: #4a86e8;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # 表单布局
        form_layout = QFormLayout()
        form_layout.setSpacing(15)
        font = QFont()
        font.setPointSize(12)

        # 用户名输入
        self.user_edit = QLineEdit()
        self.user_edit.setFont(font)
        self.user_edit.setPlaceholderText("请输入用户标识（账号）")
        form_layout.addRow(QLabel("用户标识："), self.user_edit)

        # 密码输入
        self.pwd_edit = QLineEdit()
        self.pwd_edit.setFont(font)
        self.pwd_edit.setEchoMode(QLineEdit.Password)
        self.pwd_edit.setPlaceholderText("请输入密码（6位及以上）")
        form_layout.addRow(QLabel("密    码："), self.pwd_edit)

        # 确认密码
        self.pwd_confirm_edit = QLineEdit()
        self.pwd_confirm_edit.setFont(font)
        self.pwd_confirm_edit.setEchoMode(QLineEdit.Password)
        self.pwd_confirm_edit.setPlaceholderText("请再次输入密码")
        form_layout.addRow(QLabel("确认密码："), self.pwd_confirm_edit)

        main_layout.addLayout(form_layout)

        # 注册按钮
        self.register_btn = QPushButton("完成注册")
        self.register_btn.setFixedSize(150, 40)
        self.register_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                font-size: 14px;
                border-radius: 20px;
                border: none;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QPushButton:pressed {
                background-color: #2a66c8;
            }
        """)
        self.register_btn.clicked.connect(self.do_register)
        main_layout.addWidget(self.register_btn, alignment=Qt.AlignCenter)

        # 样式
        self.setStyleSheet("background-color: #f8f9fa;")

    def init_user_file(self):
        """初始化用户信息JSON文件"""
        if not os.path.exists(os.path.dirname(USER_INFO_FILE)):
            os.makedirs(os.path.dirname(USER_INFO_FILE))
        if not os.path.exists(USER_INFO_FILE):
            with open(USER_INFO_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=4)

    def md5_encrypt(self, text):
        """MD5加密密码"""
        md5 = hashlib.md5()
        md5.update(text.encode('utf-8'))
        return md5.hexdigest()

    def do_register(self):
        """执行注册逻辑"""
        username = self.user_edit.text().strip()
        pwd = self.pwd_edit.text().strip()
        pwd_confirm = self.pwd_confirm_edit.text().strip()

        # 输入校验
        if not username or not pwd or not pwd_confirm:
            QMessageBox.warning(self, "输入错误", "请填写所有注册信息！", QMessageBox.Ok)
            return
        if len(pwd) < 6:
            QMessageBox.warning(self, "密码错误", "密码长度需6位及以上！", QMessageBox.Ok)
            return
        if pwd != pwd_confirm:
            QMessageBox.warning(self, "密码错误", "两次输入的密码不一致！", QMessageBox.Ok)
            return

        # 读取现有用户
        with open(USER_INFO_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
        if username in users:
            QMessageBox.warning(self, "注册失败", "该用户标识已存在！", QMessageBox.Ok)
            return

        # 注册成功：存储加密后的密码
        users[username] = self.md5_encrypt(pwd)
        with open(USER_INFO_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=4)

        QMessageBox.information(self, "注册成功", "用户注册成功，请返回登录！", QMessageBox.Ok)
        self.close()

class LoginWindow(QMainWindow):
    """系统登录窗口（程序启动首界面）"""
    login_success = pyqtSignal()  # 登录成功信号

    def __init__(self):
        super().__init__()
        self.setWindowTitle("系统登录 - 电动车头盔佩戴检测系统")
        self.setFixedSize(500, 400)
        self.init_ui()
        # 初始化用户信息文件
        self.init_user_file()

    def init_ui(self):
        """初始化登录界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(30)

        # 系统标题
        sys_label = QLabel("电动车头盔佩戴检测系统")
        sys_label.setStyleSheet("font-size: 22pt; font-weight: bold; color: #2d6cd9;")
        sys_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(sys_label)

        # 登录面板
        login_panel = QWidget()
        login_panel.setFixedSize(350, 200)
        login_panel.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 0 10px #e0e0e0;
            }
        """)
        panel_layout = QGridLayout(login_panel)
        panel_layout.setSpacing(20)
        panel_layout.setContentsMargins(40, 30, 40, 30)
        font = QFont()
        font.setPointSize(12)

        # 用户名
        user_label = QLabel("用户标识：")
        user_label.setFont(font)
        self.user_edit = QLineEdit()
        self.user_edit.setFont(font)
        self.user_edit.setPlaceholderText("请输入注册的用户标识")
        panel_layout.addWidget(user_label, 0, 0)
        panel_layout.addWidget(self.user_edit, 0, 1)

        # 密码
        pwd_label = QLabel("密    码：")
        pwd_label.setFont(font)
        self.pwd_edit = QLineEdit()
        self.pwd_edit.setFont(font)
        self.pwd_edit.setEchoMode(QLineEdit.Password)
        self.pwd_edit.setPlaceholderText("请输入密码")
        panel_layout.addWidget(pwd_label, 1, 0)
        panel_layout.addWidget(self.pwd_edit, 1, 1)

        main_layout.addWidget(login_panel)

        # 按钮布局
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)

        # 注册按钮
        self.register_btn = QPushButton("注册账号")
        self.register_btn.setFixedSize(120, 40)
        self.register_btn.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                color: #4a86e8;
                font-size: 14px;
                border: 1px solid #4a86e8;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #e8f4ff;
            }
        """)
        self.register_btn.clicked.connect(self.open_register)
        btn_layout.addWidget(self.register_btn)

        # 登录按钮
        self.login_btn = QPushButton("登 录")
        self.login_btn.setFixedSize(120, 40)
        self.login_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                font-size: 14px;
                border-radius: 20px;
                border: none;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QPushButton:pressed {
                background-color: #2a66c8;
            }
        """)
        self.login_btn.clicked.connect(self.do_login)
        btn_layout.addWidget(self.login_btn)

        # 用 addWidget 嵌套 QWidget 来实现居中，替代 addLayout 的 alignment 参数
        btn_container = QWidget()
        btn_container.setLayout(btn_layout)
        main_layout.addWidget(btn_container, alignment=Qt.AlignCenter)

        # 整体样式
        central_widget.setStyleSheet("background-color: #f8f9fa;")

    def init_user_file(self):
        """初始化用户信息文件"""
        if not os.path.exists(os.path.dirname(USER_INFO_FILE)):
            os.makedirs(os.path.dirname(USER_INFO_FILE))
        if not os.path.exists(USER_INFO_FILE):
            with open(USER_INFO_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=4)

    def md5_encrypt(self, text):
        """MD5加密"""
        md5 = hashlib.md5()
        md5.update(text.encode('utf-8'))
        return md5.hexdigest()

    def open_register(self):
        """打开注册窗口"""
        self.register_win = RegisterWindow(self)
        self.register_win.exec_()

    def do_login(self):
        """执行登录逻辑"""
        username = self.user_edit.text().strip()
        pwd = self.pwd_edit.text().strip()

        # 输入校验
        if not username or not pwd:
            QMessageBox.warning(self, "登录失败", "请输入用户标识和密码！", QMessageBox.Ok)
            return

        # 读取用户信息
        with open(USER_INFO_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)

        # 验证用户
        if username not in users:
            QMessageBox.warning(self, "登录失败", "用户标识不存在！", QMessageBox.Ok)
            return
        if users[username] != self.md5_encrypt(pwd):
            QMessageBox.warning(self, "登录失败", "密码错误！", QMessageBox.Ok)
            return

        # 登录成功
        QMessageBox.information(self, "登录成功", "欢迎使用头盔检测系统！", QMessageBox.Ok)
        self.login_success.emit()  # 发送登录成功信号
        self.close()

# 2. 自定义线程类
class DetectionThread(QThread):
    """后台检测线程类"""
    detection_complete = pyqtSignal(object)
    thread_error = pyqtSignal(str)
    def __init__(self, model, image_path):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self._is_running = True
    def run(self):
        try:
            print(f"线程开始检测: {self.image_path}")
            if not self._is_running:
                return
            results = self.model(self.image_path)
            print(f"线程检测完成，结果数量: {len(results)}")
            if self._is_running:
                self.detection_complete.emit(results)
        except Exception as e:
            error_msg = f"检测线程异常: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.thread_error.emit(error_msg)
            self.detection_complete.emit(None)
    def stop(self):
        print("请求停止检测线程")
        self._is_running = False
        self.quit()
        self.wait()
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()

# 3. 对话框组件
class AudioAlertSettings(QDialog):
    """语音告警设置对话框"""
    def __init__(self, parent=None, current_path=None):
        super().__init__(parent)
        self.setWindowTitle("语音告警设置")
        self.setMinimumWidth(500)
        layout = QFormLayout(self)
        self.audio_path_edit = QLineEdit()
        if current_path:
            self.audio_path_edit.setText(current_path)
        layout.addRow("语音文件路径:", self.audio_path_edit)
        browse_button = QPushButton("浏览...")
        browse_button.clicked.connect(self.browse_audio_file)
        layout.addRow(browse_button)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
    def browse_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择语音文件", "", "音频文件 (*.mp3 *.wav *.ogg)"
        )
        if file_path:
            self.audio_path_edit.setText(file_path)
    def get_audio_path(self):
        return self.audio_path_edit.text()

class ScreenshotGalleryDialog(QDialog):
    """截图查看对话框"""
    def __init__(self, parent=None, screenshot_folder=None):
        super().__init__(parent)
        self.setWindowTitle("截图查看器")
        self.setMinimumSize(800, 600)
        self.screenshot_folder = screenshot_folder
        layout = QHBoxLayout(self)
        self.file_list = QListWidget()
        self.file_list.setFixedWidth(200)
        self.file_list.itemClicked.connect(self.show_preview)
        layout.addWidget(self.file_list)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #f0f0f0;")
        layout.addWidget(self.preview_label)
        self.load_images()
    def load_images(self):
        self.file_list.clear()
        if not os.path.exists(self.screenshot_folder):
            return
        for filename in sorted(os.listdir(self.screenshot_folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                item = QListWidgetItem(filename)
                item.setData(Qt.UserRole, os.path.join(self.screenshot_folder, filename))
                self.file_list.addItem(item)
    def show_preview(self, item):
        file_path = item.data(Qt.UserRole)
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(self.preview_label.size(),
                               Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

# 4. 欢迎界面
class WelcomeScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(30)
        title_label = QLabel("电动车骑行人员头盔佩戴检测系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 24pt;
            font-weight: bold;
            color: #4a86e8;
            margin-bottom: 20px;
        """)
        layout.addWidget(title_label)
        self.start_button = QPushButton("开始使用")
        self.start_button.setMinimumSize(200, 60)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #3a76d8;
                transform: scale(1.05);
            }
            QPushButton:pressed {
                background-color: #2a66c8;
                transform: scale(0.95);
            }
        """)
        self.start_button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.start_button, alignment=Qt.AlignCenter)
        copyright_label = QLabel("YOLOv8n 头盔检测系统")
        copyright_label.setAlignment(Qt.AlignCenter)
        copyright_label.setStyleSheet("""
            font-size: 12px;
            color: #999;
            margin-top: 60px;
        """)
        layout.addWidget(copyright_label)
        self.setStyleSheet("background-color: #f8f9fa;")

# 饼图显示控件类
class PieChartWidget(QWidget):
    """实时统计饼图组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.figure = plt.figure(figsize=(4, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.update_chart(0, 0)
    def update_chart(self, with_helmet, without_helmet):
        self.figure.clear()
        if with_helmet + without_helmet == 0:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, '无数据', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
            self.canvas.draw()
            return
        labels = ['佩戴头盔', '未佩戴头盔']
        sizes = [with_helmet, without_helmet]
        colors = ['#00cc66', '#ff3333']
        explode = (0.1, 0)
        ax = self.figure.add_subplot(111)
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                          autopct='%1.1f%%', shadow=True, startangle=90)
        for text in texts + autotexts:
            text.set_fontsize(10)
        ax.axis('equal')
        self.figure.tight_layout()
        self.canvas.draw()

# 主界面类
class HelmetDetectionGUI(QMainWindow):
    """头盔检测系统主界面"""
    def __init__(self):
        super().__init__()
        self.screenshot_folder = SCREENSHOT_FOLDER
        if not os.path.exists(self.screenshot_folder):
            os.makedirs(self.screenshot_folder)
        self.hash_dict = {}
        self.screenshot_hashes = {}
        self.hash_lock = threading.Lock()
        self.load_existing_hashes()
        self.video_capture = None
        self.clear_screenshots_button = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.current_mode = None
        self.detecting = False
        self.current_stats = {}
        self.original_frame_size = None
        self.current_image_path = None
        self.detection_thread = None
        self.class_mapping = {}
        self.audio_alert_path = AUDIO_ALERT_PATH
        self.last_alert_time = 0
        self.alert_interval = 5000
        self.has_unhelmet_target = False
        self.screenshot_ids = set()
        self.screenshot_executor = ThreadPoolExecutor(max_workers=2)
        # 初始化pygame
        try:
            pygame.init()
            pygame.mixer.init()
            print("pygame.mixer 初始化成功，音频驱动:", pygame.mixer.get_init())
        except Exception as e:
            print(f"Pygame初始化失败: {str(e)}")
        self.setup_matplotlib_font()
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        self.welcome_screen = WelcomeScreen()
        self.welcome_screen.start_button.clicked.connect(self.switch_to_main_screen)
        self.stacked_widget.addWidget(self.welcome_screen)
        self.init_main_ui()
        self.stacked_widget.addWidget(self.main_widget)
        self.stacked_widget.setCurrentWidget(self.welcome_screen)
        self.init_model()
        self.setWindowTitle("YOLOv8n 电动车头盔佩戴检测系统")
        self.setGeometry(100, 100, 1200, 800)

    def save_screenshot_async(self, screenshot, screenshot_path):
        filename = os.path.basename(screenshot_path)
        new_hash = self.calculate_phash(screenshot)
        if new_hash is None:
            return
        is_duplicate = False
        with self.hash_lock:
            for existing_hash in self.screenshot_hashes.values():
                hamming_dist = sum(c1 != c2 for c1, c2 in zip(new_hash, existing_hash))
                if hamming_dist <= 25:
                    is_duplicate = True
                    break
        if not is_duplicate:
            def save_task():
                try:
                    if filename.lower().endswith('.png'):
                        cv2.imwrite(screenshot_path, cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    else:
                        cv2.imwrite(screenshot_path, cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, 100])
                    with self.hash_lock:
                        self.screenshot_hashes[filename] = new_hash
                    print(f"截图保存成功: {screenshot_path}")
                except Exception as e:
                    print(f"截图保存失败: {str(e)}")
            self.screenshot_executor.submit(save_task)
        else:
            print(f"跳过重复截图: {filename}")

    def load_existing_hashes(self):
        if not os.path.exists(self.screenshot_folder):
            return
        for filename in os.listdir(self.screenshot_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(self.screenshot_folder, filename)
                try:
                    image = cv2.imread(filepath)
                    if image is not None:
                        phash = self.calculate_phash(image)
                        with self.hash_lock:
                            self.screenshot_hashes[filename] = phash
                except Exception as e:
                    print(f"加载已有截图哈希失败 {filename}: {str(e)}")

    def calculate_phash(self, image):
        try:
            resized = cv2.resize(image, (32, 32))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            dct = cv2.dct(np.float32(gray))
            dct_roi = dct[0:8, 0:8]
            avg = np.mean(dct_roi)
            hash_str = ''.join(['1' if i > avg else '0' for i in dct_roi.flatten()])
            return hash_str
        except Exception as e:
            print(f"计算pHash失败: {str(e)}")
            return None

    def show_screenshots_gallery(self):
        if not os.path.exists(self.screenshot_folder) or not os.listdir(self.screenshot_folder):
            self.show_message("提示", "截图文件夹为空", "info")
            return
        dialog = ScreenshotGalleryDialog(self, self.screenshot_folder)
        dialog.exec_()

    def clear_screenshots(self):
        reply = self.show_message("确认清空", "确定要删除所有截图吗？此操作将不可恢复！", "question")
        if reply != QMessageBox.Yes:
            return
        deleted_files = 0
        try:
            if os.path.exists(self.screenshot_folder):
                for filename in os.listdir(self.screenshot_folder):
                    file_path = os.path.join(self.screenshot_folder, filename)
                    try:
                        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            os.remove(file_path)
                            deleted_files += 1
                    except Exception as e:
                        print(f"删除文件失败 {file_path}: {str(e)}")
            self.hash_dict.clear()
            self.screenshot_ids.clear()
            print(f"已删除 {deleted_files} 个截图文件")
            self.show_message("操作成功",
                              f"成功删除{deleted_files}个截图文件\n截图目录已清空",
                              "info")
            self.status_label.setText("状态: 截图已清空")
        except Exception as e:
            self.show_message("操作失败",
                              f"清空截图时发生错误: {str(e)}",
                              "critical")
            print(f"清空截图失败: {str(e)}")
        with self.hash_lock:
            self.screenshot_hashes.clear()

    def calculate_dhash(self, image):
        try:
            resized = cv2.resize(image, (9, 8))
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            hash_str = ''
            for i in range(8):
                for j in range(8):
                    if gray[i, j] > gray[i, j + 1]:
                        hash_str += '1'
                    else:
                        hash_str += '0'
            return hash_str
        except:
            return None

    def hamming_distance(self, hash1, hash2):
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def setup_matplotlib_font(self):
        try:
            import matplotlib.font_manager as fm
            available_fonts = {f.name: f for f in fm.fontManager.ttflist}
            chinese_fonts = [
                "SimHei", "WenQuanYi Zen Hei", "Heiti TC",
                "Microsoft YaHei", "SimSun"
            ]
            found_chinese_font = False
            for font_name in chinese_fonts:
                if font_name in available_fonts:
                    plt.rcParams["font.family"] = font_name
                    found_chinese_font = True
                    print(f"已设置matplotlib字体: {font_name}")
                    break
            if not found_chinese_font:
                default_font = fm.findfont(fm.FontProperties())
                default_font_name = fm.FontProperties(fname=default_font).get_name()
                plt.rcParams["font.family"] = default_font_name
                print(f"未找到中文字体，使用系统默认字体: {default_font_name}")
                plt.rcParams["font.sans-serif"] = chinese_fonts + plt.rcParams["font.sans-serif"]
        except Exception as e:
            print(f"设置matplotlib字体时出错: {str(e)}")
            print("将使用matplotlib默认字体设置")

    def switch_to_main_screen(self):
        self.animation = QPropertyAnimation(self.stacked_widget, b"currentIndex")
        self.animation.setDuration(500)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.start()
        self.update_menu_state()

    def init_main_ui(self):
        self.main_widget = QWidget()
        font = QFont()
        default_font = QFont()
        default_font_family = default_font.family()
        font.setFamily(default_font_family)
        font.setPointSize(10)
        self.setFont(font)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout = QVBoxLayout(self.main_widget)
        main_layout.addWidget(main_splitter)
        # 左侧菜单栏
        menu_widget = QWidget()
        menu_layout = QVBoxLayout(menu_widget)
        title_label = QLabel("电动车头盔检测系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 21px; font-weight: bold; padding: 10px; color: #333;")
        menu_layout.addWidget(title_label)
        # 语音告警设置
        audio_alert_button = QPushButton("设置语音告警")
        audio_alert_button.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                color: #333;
                border: 1px solid #ddd;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        audio_alert_button.clicked.connect(self.set_audio_alert)
        menu_layout.addWidget(audio_alert_button)
        # 清空截图
        clear_screenshots_button = QPushButton("清空截图")
        clear_screenshots_button.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                color: #333;
                border: 1px solid #ddd;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        clear_screenshots_button.clicked.connect(self.clear_screenshots)
        menu_layout.addWidget(clear_screenshots_button)
        self.clear_screenshots_button = clear_screenshots_button
        # 查看截图
        view_screenshots_button = QPushButton("查看截图")
        view_screenshots_button.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                color: #333;
                border: 1px solid #ddd;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        view_screenshots_button.clicked.connect(self.show_screenshots_gallery)
        menu_layout.addWidget(view_screenshots_button)
        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        menu_layout.addWidget(line)
        # 功能菜单列表
        self.menu_list = QListWidget()
        self.menu_list.setStyleSheet("""
            QListWidget {
                border: none;
                background-color: #f5f5f5;
                font-size: 20px;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #ddd;
            }
            QListWidget::item:selected {
                background-color: #4a86e8;
                color: white;
            }
        """)
        menu_items = ["加载图片", "加载视频", "启动/关闭摄像头", "开始检测", "停止检测"]
        for item in menu_items:
            list_item = QListWidgetItem(item)
            list_item.setFlags(list_item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.menu_list.addItem(list_item)
        self.menu_list.itemClicked.connect(self.menu_item_clicked)
        menu_layout.addWidget(self.menu_list)
        # 状态标签
        status_label = QLabel("状态: 就绪")
        self.status_label = status_label
        status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_label.setStyleSheet("padding: 8px; background-color: #e8f4ff; font-size: 11px;")
        menu_layout.addWidget(status_label)
        main_splitter.addWidget(menu_widget)
        # 右侧内容展示区
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        display_layout = QHBoxLayout()
        # 检测画面显示
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("请选择图片、视频或启用摄像头")
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("""
            border: 2px dashed #cccccc; 
            background-color: #f8f9fa;
            font-size: 14px;
            color: #666;
        """)
        display_layout.addWidget(self.image_label, 7)
        # 饼图统计
        self.chart_widget = PieChartWidget()
        self.chart_widget.setMinimumSize(300, 300)
        self.chart_widget.setMaximumWidth(350)
        display_layout.addWidget(self.chart_widget, 3)
        content_layout.addLayout(display_layout)
        # 检测统计标签
        self.stats_label = QLabel()
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("""
            padding: 12px; 
            background-color: #4a86e8; 
            color: white; 
            font-size: 14px; 
            font-weight: bold;
            border-radius: 4px;
            margin-top: 10px;
        """)
        content_layout.addWidget(self.stats_label)
        # 告警提示标签
        self.warning_label = QLabel()
        self.warning_label.setAlignment(Qt.AlignCenter)
        self.warning_label.setStyleSheet("""
                padding: 15px; 
                background-color: #ff4444; 
                color: white; 
                font-size: 18px; 
                font-weight: bold;
                border-radius: 8px;
                margin-top: 10px;
            """)
        self.warning_label.setText("⚠️ 检测到当前区域出现有未佩戴头盔者！")
        self.warning_label.hide()
        content_layout.addWidget(self.warning_label)
        main_splitter.addWidget(content_widget)
        main_splitter.setSizes([200, 1000])

    def init_model(self):
        if not os.path.exists(MODEL_PATH):
            self.show_message("错误", f"模型文件 {MODEL_PATH} 不存在，请先运行1.py训练模型！", "critical")
            for i in range(self.menu_list.count()):
                if self.menu_list.item(i).text() in ["开始检测", "停止检测"]:
                    self.menu_list.item(i).setFlags(self.menu_list.item(i).flags() & ~Qt.ItemIsEnabled)
            self.status_label.setText("状态: 模型未找到（请先训练）")
            return
        try:
            print(f"加载YOLOv8n模型: {MODEL_PATH}")
            from ultralytics import YOLO
            self.model = YOLO(MODEL_PATH)
            self.class_names = self.model.names
            self.create_class_mapping()
            print("YOLOv8n模型加载成功")
            self.status_label.setText("状态: YOLOv8n模型加载成功")
        except ImportError:
            self.show_message("错误", "未找到ultralytics包，请执行：pip install ultralytics", "critical")
            self.status_label.setText("状态: 缺少依赖包")
        except Exception as e:
            self.show_message("错误", f"加载模型失败: {str(e)}", "critical")
            print(f"模型加载失败: {str(e)}")
            for i in range(self.menu_list.count()):
                if self.menu_list.item(i).text() in ["开始检测", "停止检测"]:
                    self.menu_list.item(i).setFlags(self.menu_list.item(i).flags() & ~Qt.ItemIsEnabled)
            self.status_label.setText("状态: 模型加载失败")

    def create_class_mapping(self):
        self.class_mapping = {
            0: "With Helmet",
            1: "Without Helmet"
        }
        print(f"类别映射创建完成（适配你的数据集）: {self.class_mapping}")

    def set_audio_alert(self):
        dialog = AudioAlertSettings(self, self.audio_alert_path)
        if dialog.exec_():
            new_path = dialog.get_audio_path()
            if new_path and os.path.exists(new_path):
                self.audio_alert_path = new_path
                self.show_message("成功", f"语音告警文件已设置为: {os.path.basename(new_path)}", "info")
                self.status_label.setText(f"状态: 语音告警已设置 - {os.path.basename(new_path)}")
            else:
                self.show_message("错误", "无效的文件路径，请选择有效的音频文件", "warning")

    def play_audio_alert(self):
        print(f"触发语音告警，路径: {self.audio_alert_path}")
        if not self.audio_alert_path or not os.path.exists(self.audio_alert_path):
            print("错误：音频文件不存在，可在GUI中重新设置")
            return
        current_time = pygame.time.get_ticks()
        if current_time - self.last_alert_time < self.alert_interval:
            print(f"告警间隔未达到，剩余时间: {self.alert_interval - (current_time - self.last_alert_time)}ms")
            return
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(self.audio_alert_path)
            pygame.mixer.music.play()
            self.last_alert_time = current_time
            print("告警播放成功")
        except Exception as e:
            print(f"播放失败: {str(e)}")
            try:
                sound = pygame.mixer.Sound(self.audio_alert_path)
                sound.play()
                print("通过Sound对象播放成功")
            except Exception as e2:
                print(f"备选播放方案失败: {str(e2)}")

    def menu_item_clicked(self, item):
        action = item.text()
        print(f"菜单项点击: {action}")
        self.status_label.setText(f"状态: {action}")
        if action == "加载图片":
            self.load_image()
        elif action == "加载视频":
            self.load_video()
        elif action == "启动/关闭摄像头":
            self.toggle_camera()
        elif action == "开始检测":
            self.start_detection()
        elif action == "停止检测":
            self.stop_detection()
        self.update_menu_state()

    def load_image(self):
        self.stop_detection()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            print(f"加载图片: {file_path}")
            self.current_mode = "image"
            self.current_image_path = file_path
            self.display_image(file_path)
            self.status_label.setText(f"状态: 图片已加载 - {os.path.basename(file_path)}")

    def load_video(self):
        self.stop_detection()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            print(f"加载视频: {file_path}")
            self.current_mode = "video"
            self.video_capture = cv2.VideoCapture(file_path)
            if not self.video_capture.isOpened():
                self.show_message("错误", "无法打开视频文件", "critical")
                self.status_label.setText(f"状态: 无法打开视频 - {os.path.basename(file_path)}")
                return
            self.status_label.setText(f"状态: 视频已加载 - {os.path.basename(file_path)}")

    def toggle_camera(self):
        if self.current_mode == "camera" and self.timer.isActive():
            self.stop_detection()
            self.timer.stop()
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            self.current_mode = None
            self.image_label.setText("摄像头已关闭")
            self.status_label.setText("状态: 摄像头关闭")
        else:
            self.stop_detection()
            self.current_mode = "camera"
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                self.show_message("错误", "无法打开摄像头（检查摄像头权限）", "critical")
                self.status_label.setText("状态: 摄像头打开失败")
                return
            self.timer.start(30)
            self.status_label.setText("状态: 摄像头已启动（实时检测）")

    def start_detection(self):
        if not hasattr(self, 'model') or not self.model:
            self.show_message("错误", "模型未加载", "critical")
            self.status_label.setText("状态: 检测失败 - 模型未加载")
            return
        if self.current_mode is None:
            self.show_message("信息", "请先选择图片、视频或启动摄像头", "info")
            self.status_label.setText("状态: 检测失败 - 未选择内容")
            return
        self.detecting = True
        self.has_unhelmet_target = False
        self.screenshot_ids.clear()
        print("开始检测")
        self.status_label.setText("状态: 正在检测...")
        if self.current_mode == "image" and self.current_image_path:
            if self.detection_thread and self.detection_thread.isRunning():
                self.detection_thread.stop()
            print(f"开始图片检测: {self.current_image_path}")
            self.detection_thread = DetectionThread(self.model, self.current_image_path)
            self.detection_thread.detection_complete.connect(self.on_image_detection_complete)
            self.detection_thread.thread_error.connect(self.on_thread_error)
            self.detection_thread.finished.connect(self.on_detection_thread_finished)
            self.detection_thread.start()
        elif self.current_mode == "video":
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.timer.start(int(1000 / fps))
        self.update_menu_state()

    def stop_detection(self):
        if self.detecting:
            print("停止检测")
            self.detecting = False
            self.hash_dict = {}
            if self.current_mode == "video" and self.timer.isActive():
                self.timer.stop()
                print("视频已暂停")
        self.has_unhelmet_target = False
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
        self.update_menu_state()
        self.status_label.setText("状态: 检测已停止")
        self.warning_label.hide()

    def on_detection_thread_finished(self):
        print("检测线程已结束")
        if self.detecting:
            self.status_label.setText("状态: 检测已完成")

    def on_thread_error(self, error_msg):
        self.show_message("错误", error_msg, "critical")
        self.status_label.setText(f"状态: 错误 - {error_msg}")

    def on_image_detection_complete(self, results):
        print("图片检测完成回调")
        if results is not None and len(results) > 0:
            try:
                image = cv2.imread(self.current_image_path)
                if image is None:
                    self.show_message("错误", f"无法读取图片: {self.current_image_path}", "critical")
                    print(f"无法读取图片: {self.current_image_path}")
                    self.status_label.setText(f"状态: 读取图片失败 - {os.path.basename(self.current_image_path)}")
                    return
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image, without_helmet_count = self.draw_results(image, results[0])
                self.update_detection_stats(results[0])
                with_helmet = self.current_stats.get("with_helmet", 0)
                without_helmet = self.current_stats.get("without_helmet", 0)
                self.chart_widget.update_chart(with_helmet, without_helmet)
                if without_helmet_count > 0:
                    self.has_unhelmet_target = True
                    self.play_audio_alert()
                h, w, ch = image.shape
                bytes_per_line = ch * w
                q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = self.scale_to_fit(pixmap)
                self.image_label.setPixmap(scaled_pixmap)
                person_count = with_helmet + without_helmet
                self.status_label.setText(f"状态: 检测已完成 - 检测到{person_count}人，未佩戴{without_helmet}人")
            except Exception as e:
                print(f"显示检测结果时出错: {str(e)}")
                self.show_message("错误", f"处理检测结果时出错: {str(e)}", "critical")
                self.status_label.setText(f"状态: 处理结果错误 - {str(e)}")
        else:
            self.show_message("错误", "检测过程中发生错误或未检测到对象", "critical")
            self.status_label.setText("状态: 检测失败 - 未检测到对象")

    def update_frame(self):
        if not self.video_capture:
            return
        try:
            ret, frame = self.video_capture.read()
            if not ret:
                self.timer.stop()
                if self.current_mode == "video":
                    self.video_capture.release()
                    self.video_capture = None
                    self.current_mode = None
                    self.image_label.setText("视频播放已完成")
                    self.status_label.setText("状态: 视频播放已完成")
                return
            if self.detecting:
                try:
                    results = self.model(frame)[0]
                    frame, without_helmet_count = self.draw_results(frame, results)
                    self.update_detection_stats(results)
                    with_helmet = self.current_stats.get("with_helmet", 0)
                    without_helmet = self.current_stats.get("without_helmet", 0)
                    self.chart_widget.update_chart(with_helmet, without_helmet)
                    if without_helmet_count > 0:
                        self.has_unhelmet_target = True
                        self.play_audio_alert()
                    else:
                        self.has_unhelmet_target = False
                    self.status_label.setText(
                        f"状态: 正在检测 - 共{with_helmet+without_helmet}人，未佩戴{without_helmet}人"
                    )
                except Exception as e:
                    print(f"视频检测错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
            self.display_frame(frame)
        except Exception as e:
            print(f"视频帧处理异常: {str(e)}")
            self.stop_detection()

    def display_image(self, path):
        if not os.path.exists(path):
            self.show_message("错误", f"图片文件不存在: {path}", "critical")
            self.status_label.setText(f"状态: 图片未找到 - {os.path.basename(path)}")
            return
        print(f"读取图片: {path}")
        image = cv2.imread(path)
        if image is None:
            self.show_message("错误", f"无法读取图片: {path}", "critical")
            print(f"无法读取图片: {path}")
            self.status_label.setText(f"状态: 读取图片失败 - {os.path.basename(path)}")
            return
        max_size = 2048
        height, width = image.shape[:2]
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, new_height))
            print(f"图片已缩放: {width}x{height} -> {new_width}x{new_height}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.original_frame_size = (image.shape[1], image.shape[0])
        if self.detecting:
            try:
                print("执行图片检测...")
                results = self.model(image)[0]
                image, without_helmet_count = self.draw_results(image, results)
                self.update_detection_stats(results)
                with_helmet = self.current_stats.get("with_helmet", 0)
                without_helmet = self.current_stats.get("without_helmet", 0)
                self.chart_widget.update_chart(with_helmet, without_helmet)
                if without_helmet_count > 0:
                    self.has_unhelmet_target = True
                    self.play_audio_alert()
                print("图片检测已完成")
                person_count = with_helmet + without_helmet
                self.status_label.setText(f"状态: 图片已检测 - 共{person_count}人，未佩戴{without_helmet}人")
            except Exception as e:
                print(f"图片检测过程中出错: {e}")
                self.show_message("错误", f"检测过程中发生错误: {str(e)}", "critical")
                self.stop_detection()
                self.status_label.setText(f"状态: 图片检测错误 - {str(e)}")
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = self.scale_to_fit(pixmap)
        self.image_label.setPixmap(scaled_pixmap)

    def display_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.original_frame_size is None:
            self.original_frame_size = (frame.shape[1], frame.shape[0])
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = self.scale_to_fit(pixmap)
        self.image_label.setPixmap(scaled_pixmap)

    def scale_to_fit(self, pixmap):
        if pixmap.isNull():
            return pixmap
        label_size = self.image_label.size()
        scaled_pixmap = pixmap.scaled(
            label_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        return scaled_pixmap

    def draw_results(self, frame, result):
        current_hashes = {}
        with_helmet_count = 0
        without_helmet_count = 0
        current_time = time.time() * 1000
        def get_compound_hash(img, x_center, y_center):
            try:
                resized = cv2.resize(img, (32, 32))
                hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
                hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
                hist_h = cv2.normalize(hist_h, hist_h).flatten()
                gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                hash_str = []
                for i in range(31):
                    hash_str.append('1' if gray[i, i] > gray[i, i + 1] else '0')
                    hash_str.append('1' if gray[i, i] > gray[i + 1, i] else '0')
                grid_size = 100
                pos_x = int(x_center / grid_size)
                pos_y = int(y_center / grid_size)
                return (
                        ''.join(hash_str) +
                        f"{pos_x:04b}{pos_y:04b}" +
                        "".join([f"{int(v * 15):04b}" for v in hist_h[:2]])
                )
            except Exception as e:
                print(f"生成复合哈希失败: {str(e)}")
                return None
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            class_name = self.class_mapping.get(cls_id, f"Class_{cls_id}")
            if class_name == "Without Helmet":
                without_helmet_count += 1
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                h, w = frame.shape[:2]
                obj_height = y2 - y1
                margin = min(1200, int(obj_height * 4))
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                screenshot = frame[y1:y2, x1:x2]
                if screenshot.size == 0:
                    continue
                comp_hash = get_compound_hash(screenshot, x_center, y_center)
                if not comp_hash:
                    continue
                is_duplicate = False
                for existing_hash, (ex_x, ex_y, ex_time) in list(self.hash_dict.items()):
                    if self.hamming_distance(comp_hash, existing_hash) < 15 and \
                            abs(ex_x - x_center) < 50 and abs(ex_y - y_center) < 50 and \
                            current_time - ex_time < 5000:
                        is_duplicate = True
                        self.hash_dict[existing_hash] = (x_center, y_center, current_time)
                        break
                if not is_duplicate:
                    filename = f"{current_time}_{uuid.uuid4().hex[:6]}.jpg"
                    screenshot_path = os.path.join(self.screenshot_folder, filename)
                    self.save_screenshot_async(screenshot, screenshot_path)
                    self.hash_dict[comp_hash] = (x_center, y_center, current_time)
                    print(f"新截图: {filename}")
            elif class_name == "With Helmet":
                with_helmet_count += 1
        self.hash_dict = {
            k: v for k, v in self.hash_dict.items()
            if current_time - v[2] < 10000
        }
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            conf = round(float(box.conf), 2)
            class_name = self.class_mapping.get(cls_id, f"Class_{cls_id}")
            color = (0, 0, 255) if class_name == "Without Helmet" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.current_stats = {
            "with_helmet": with_helmet_count,
            "without_helmet": without_helmet_count
        }
        if without_helmet_count > 0:
            self.warning_label.show()
        else:
            self.warning_label.hide()
        return frame, without_helmet_count

    def update_detection_stats(self, result):
        if hasattr(self, 'current_stats'):
            with_helmet_count = self.current_stats.get("with_helmet", 0)
            without_helmet_count = self.current_stats.get("without_helmet", 0)
            total_persons = with_helmet_count + without_helmet_count
            compliance_rate = f"{with_helmet_count}/{total_persons}" if total_persons > 0 else "0/0"
            self.stats_label.setText(
                f"检测结果: 总检测{total_persons}人 | 佩戴头盔{with_helmet_count}人 | 未佩戴{without_helmet_count}人 | 佩戴率{compliance_rate}"
            )
        if self.current_stats.get("without_helmet", 0) > 0:
            self.warning_label.show()
        else:
            self.warning_label.hide()

    def update_menu_state(self):
        for i in range(self.menu_list.count()):
            item_text = self.menu_list.item(i).text()
            self.menu_list.item(i).setFlags(self.menu_list.item(i).flags() | Qt.ItemIsEnabled)
            if item_text in ["加载图片", "加载视频", "启动/关闭摄像头"]:
                if self.detecting:
                    self.menu_list.item(i).setFlags(self.menu_list.item(i).flags() & ~Qt.ItemIsEnabled)
            elif item_text == "开始检测":
                if self.current_mode is None or self.detecting:
                    self.menu_list.item(i).setFlags(self.menu_list.item(i).flags() & ~Qt.ItemIsEnabled)
            elif item_text == "停止检测":
                if not self.detecting:
                    self.menu_list.item(i).setFlags(self.menu_list.item(i).flags() & ~Qt.ItemIsEnabled)

    def show_message(self, title, message, msg_type="info"):
        print(f"{title}: {message}")
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        if msg_type == "info":
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setStandardButtons(QMessageBox.Ok)
        elif msg_type == "warning":
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setStandardButtons(QMessageBox.Ok)
        elif msg_type == "critical":
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setStandardButtons(QMessageBox.Ok)
        elif msg_type == "question":
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        return msg_box.exec_()

    def closeEvent(self, event):
        print("关闭窗口，释放资源...")
        self.stop_detection()
        if self.video_capture:
            self.video_capture.release()
        pygame.mixer.quit()
        event.accept()

# -------------------------- 程序主入口 --------------------------
if __name__ == "__main__":
    os.environ["QT_FONT_DPI"] = "96"
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    default_font = QFont()
    app.setFont(default_font)

    # 启动登录窗口
    login_win = LoginWindow()
    # 定义登录成功后的操作：启动主系统
    def start_main_system():
        main_win = HelmetDetectionGUI()
        main_win.show()
    login_win.login_success.connect(start_main_system)
    login_win.show()

    sys.exit(app.exec_())