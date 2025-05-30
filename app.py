import cv2
import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtWidgets import (
    QApplication, QInputDialog, QSizePolicy, QToolBar, 
    QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, 
    QFileDialog, QPushButton, QLabel, QVBoxLayout, QMessageBox,
    QWidget, QDialog, QVBoxLayout, QLabel, QSlider, QDialogButtonBox
)
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QPainter

class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)

        self.scale_factor = 1.01  # Zoom speed
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # Enable drag to move image

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def set_image(self, image):
        # Convert OpenCV image to QPixmap and display
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.image_item.setPixmap(pixmap)
        # self.image_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.setSceneRect(QRectF(pixmap.rect()))

    def wheelEvent(self, event: QWheelEvent):
        # Zoom in/out with mouse wheel
        if event.angleDelta().y() > 0:
            self.scale(self.scale_factor, self.scale_factor)
        else:
            self.scale(1 / self.scale_factor, 1 / self.scale_factor)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_image()

    def fit_image(self):
        # Fit image to view without losing quality (keeps original res)
        self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)

class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PixNity')
        self.setGeometry(100, 100, 800, 600)

        # Layout
        self.layout = QVBoxLayout()

        self.img_path = None
        self.cv_image = None

        self.file_toolbar = QToolBar()
        self.file_toolbar.setMovable(False)
        self.file_toolbar.setFloatable(False)
        self.file_toolbar.setOrientation(Qt.Orientation.Horizontal)
        self.file_toolbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.file_toolbar.setMinimumHeight(40)
        self.file_toolbar.setStyleSheet(
            """
                QToolBar {
                    spacing: 10px;
                    padding: 5px;
                }
            """
        )
        self.layout.addWidget(self.file_toolbar)

        # Button to load image
        self.load_button = QPushButton('Load')
        def load_image():
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
            if file_path:
                self.img_path = file_path
                self.set_img()
                self.viewer.fit_image()
        self.load_button.clicked.connect(load_image)
        self.file_toolbar.addWidget(self.load_button)

        # Button to save image
        self.save_button = QPushButton('Save')
        def save_image():
            if not hasattr(self, 'cv_image') or self.cv_image is None:
                QMessageBox.warning(self, "No Image", "No image loaded to save.")
                return
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image As", "", "Images (*.png *.jpg *.bmp)"
            )
            if save_path:
                image_to_save = cv2.cvtColor(self.cv_image, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(save_path, image_to_save)
                if not success:
                    QMessageBox.warning(self, "Save Error", "Failed to save the image.")
                else:
                    QMessageBox.information(self, "Saved", "Image saved successfully.")
        self.save_button.clicked.connect(save_image)
        self.file_toolbar.addWidget(self.save_button)

        # Label to show image
        self.viewer = ImageViewer()
        self.layout.addWidget(self.viewer)

        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setOrientation(Qt.Orientation.Horizontal)
        self.toolbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toolbar.setMinimumHeight(40)
        self.toolbar.setStyleSheet(
            """
                QToolBar {
                    spacing: 10px;
                    padding: 5px;
                }
            """
        )
        self.layout.addWidget(self.toolbar)

        # Button to sharpen image
        self.sharpen_button = QPushButton('Sharpen')
        def sharpen_button_fn():
            k = np.array([
                [  -1,  -1,  -1],
                [  -1,  17,  -1],
                [  -1,  -1,  -1]
            ], dtype=np.float32) / 9.0
            im_f = cv2.filter2D(self.cv_image, -1, k)
            self.cv_image = im_f
            self.viewer.set_image(self.cv_image)
        self.sharpen_button.clicked.connect(sharpen_button_fn)
        self.toolbar.addWidget(self.sharpen_button)

        # Button to blur image
        self.blur_button = QPushButton('Blur')
        def blur_button_fn():
            k = np.array([
                [   1,   1,   1],
                [   1,   1,   1],
                [   1,   1,   1]
            ], dtype=np.float32) / 9.0
            im_f = cv2.filter2D(self.cv_image, -1, k)
            self.cv_image = im_f
            self.viewer.set_image(self.cv_image)
        self.blur_button.clicked.connect(blur_button_fn)
        self.toolbar.addWidget(self.blur_button)

        # Button to outline image
        self.outline_button = QPushButton('Outline')
        def outline_button_fn():
            img = self.cv_image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
            gradient_magnitude = gradient_magnitude.astype(np.uint8)
            edge_mask = (gradient_magnitude > 50).astype(np.uint8) * 255  # 0 or 255 values
            edge_mask = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2RGB)
            self.cv_image = edge_mask
            self.viewer.set_image(self.cv_image)
        self.outline_button.clicked.connect(outline_button_fn)
        self.toolbar.addWidget(self.outline_button)

        # Button to invert image
        self.invert_button = QPushButton('Invert')
        def invert_button_fn():
            im_f = cv2.bitwise_not(self.cv_image)
            self.cv_image = im_f
            self.viewer.set_image(self.cv_image)
        self.invert_button.clicked.connect(invert_button_fn)
        self.toolbar.addWidget(self.invert_button)

        # Button to Black/White image
        self.bw_button = QPushButton('Black/White')
        def get_bw_value():
            dialog = QDialog(self)
            dialog.setWindowTitle("B&W Threshold")

            slider = QSlider(Qt.Orientation.Horizontal, dialog)
            slider.setRange(1, 255)
            slider.setValue(128)

            label = QLabel(f"Threshold: {slider.value()}", dialog)
            slider.valueChanged.connect(lambda v: label.setText(f"Threshold: {v}"))

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=dialog)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)

            layout = QVBoxLayout(dialog)
            layout.addWidget(label)
            layout.addWidget(slider)
            layout.addWidget(buttons)

            if dialog.exec():
                bw_button_fn(slider.value())
                print("Threshold set to:", slider.value())
        def bw_button_fn(n):
            img = self.cv_image
            threshold_b = n
            threshold_w = n
            if img.ndim == 2:
                mask_b = img < threshold_b
                mask_w = img > threshold_w
            elif img.ndim == 3:
                mask_b = (img < threshold_b).any(axis=2)
                mask_w = (img > threshold_w).any(axis=2)
            img[mask_b] = [  0,   0,   0]
            img[mask_w] = [255, 255, 255]
            self.cv_image = img
            self.viewer.set_image(self.cv_image)
        self.bw_button.clicked.connect(get_bw_value)
        self.toolbar.addWidget(self.bw_button)

        # Button to Colorify image
        self.colorify_button = QPushButton('Colorify')
        def get_colorify_value():
            value, ok = QInputDialog.getInt(self, "Colorify Value Input", "Enter number of splits:", min=1, max=256)
            if ok:
                colorify_button_fn(value)
                print("User entered:", value)
        def colorify_button_fn(n):
            img = self.cv_image
            def quantize_channel(channel, n):
                out = np.zeros_like(channel)
                for i in range(0, n):
                    high = 256//n * (i+1)
                    low = 256//n * i
                    out[(channel >= low) & (channel < high)] = (high+low)//2
                return out
            r = quantize_channel(img[:, :, 0], n)
            g = quantize_channel(img[:, :, 1], n)
            b = quantize_channel(img[:, :, 2], n)
            quantized_img = np.stack((r, g, b), axis=2)
            self.cv_image = quantized_img
            self.viewer.set_image(self.cv_image)
        self.colorify_button.clicked.connect(get_colorify_value)
        self.toolbar.addWidget(self.colorify_button)
        
        # Button to reset image
        self.r_button = QPushButton('Reset')
        self.r_button.clicked.connect(self.set_img)
        self.toolbar.addWidget(self.r_button)

        self.setLayout(self.layout)

    def set_img(self):
        """Reset the image to the original."""
        self.cv_image = cv2.imread(self.img_path)
        if self.cv_image is None:
            return
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.viewer.set_image(self.cv_image)

app = QApplication([])

window = ImageApp()
window.show()

app.exec()