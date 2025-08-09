"""PixNity - A simple image viewer and editor using PyQt6 and OpenCV."""

from typing import (
    Callable,
    Optional,
)

import cv2
import numpy as np
from PyQt6.QtCore import (
    QRectF,
    Qt,
)
from PyQt6.QtGui import (
    QImage,
    QPainter,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


class ImageViewer(QGraphicsView):
    """A custom QGraphicsView to display and manipulate images."""

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene()
        self.setScene(self._scene)

        self.image_item = QGraphicsPixmapItem()
        self._scene.addItem(self.image_item)

        self.scale_factor = 1.01  # Zoom speed
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # Enable drag to move image

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def display_image(self, image):
        """Convert OpenCV image to QPixmap and display"""
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.image_item.setPixmap(pixmap)
        # self.image_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.setSceneRect(QRectF(pixmap.rect()))

    def wheel_event(self, event: QWheelEvent):
        """Zoom in/out with mouse wheel"""
        if event.angleDelta().y() > 0:
            self.scale(self.scale_factor, self.scale_factor)
        else:
            self.scale(1 / self.scale_factor, 1 / self.scale_factor)

    def resize_event(self, event):
        """Handle resize events to fit image"""
        super().resizeEvent(event)
        self.fit_image()

    def fit_image(self):
        """Fit image to view without losing quality (keeps original res)"""
        self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)


class ImageApp(QWidget):
    """Main application window for PixNity."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PixNity")
        self.setGeometry(100, 100, 800, 600)

        self.img_path: str = ""
        self.cv_image: Optional[np.ndarray] = None
        self.viewer: ImageViewer = ImageViewer()

        # Layout
        self.main_layout: QVBoxLayout = QVBoxLayout()

        # Toolbar for file operations
        self.file_toolbar = self._create_toolbar()
        self.main_layout.addWidget(self.file_toolbar)
        self._populate_toolbar(
            self.file_toolbar,
            [
                ("Load", self.load_image_button_fn),
                ("Save", self.save_image_button_fn),
            ]
        )

        # Label to show image
        self.main_layout.addWidget(self.viewer)

        # Toolbar for image manipulation
        self.toolbar: QToolBar = self._create_toolbar()
        self.main_layout.addWidget(self.toolbar)

        self._populate_toolbar(
            self.toolbar,
            [
                ("Sharpen", self.sharpen_button_fn),
                ("Blur", self.blur_button_fn),
                ("Outline", self.outline_button_fn),
                ("Invert", self.invert_button_fn),
                ("Black/White", self.bw_button_fn),
                ("Colorify", self.colorify_button_fn),
                ("Reset", self.set_image),
            ]
        )

        self.setLayout(self.main_layout)


    def load_image_button_fn(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.img_path = file_path
            self.set_image()
            self.viewer.fit_image()


    def save_image_button_fn(self):
        """Save the current image to a file."""
        if not hasattr(self, "cv_image") or self.cv_image is None:
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


    def sharpen_button_fn(self):
        """Sharpen the image using a kernel."""
        if self.cv_image is None:
            QMessageBox.warning(self, "No Image", "No image loaded to sharpen.")
            return
        k = np.array([[-1, -1, -1], [-1, 17, -1], [-1, -1, -1]], dtype=np.float32) / 9.0
        im_f = cv2.filter2D(self.cv_image, -1, k)
        self.cv_image = im_f
        self.viewer.display_image(self.cv_image)


    def blur_button_fn(self):
        """Blur the image using a simple averaging kernel."""
        if self.cv_image is None:
            QMessageBox.warning(self, "No Image", "No image loaded to blur.")
            return
        k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9.0
        im_f = cv2.filter2D(self.cv_image, -1, k)
        self.cv_image = im_f
        self.viewer.display_image(self.cv_image)


    def outline_button_fn(self):
        """Outline the image using Sobel edge detection."""
        if self.cv_image is None:
            QMessageBox.warning(self, "No Image", "No image loaded to outline.")
            return
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
        self.viewer.display_image(self.cv_image)


    def invert_button_fn(self):
        """Invert the colors of the image."""
        if self.cv_image is None:
            QMessageBox.warning(self, "No Image", "No image loaded to invert.")
            return
        im_f = cv2.bitwise_not(self.cv_image)
        self.cv_image = im_f
        self.viewer.display_image(self.cv_image)


    def bw_button_fn(self):
        """Prompt user for B&W threshold value."""
        dialog = QDialog(self)
        dialog.setWindowTitle("B&W Threshold")

        slider = QSlider(Qt.Orientation.Horizontal, dialog)
        slider.setRange(1, 255)
        slider.setValue(128)

        label = QLabel(f"Threshold: {slider.value()}", dialog)
        slider.valueChanged.connect(lambda v: label.setText(f"Threshold: {v}"))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout = QVBoxLayout(dialog)
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(buttons)

        if dialog.exec():
            self.convert_to_bw(slider.value())
            print("Threshold set to:", slider.value())


    def convert_to_bw(self, threshold: int):
        """Convert image to black and white based on threshold."""
        if self.cv_image is None:
            QMessageBox.warning(self, "No Image", "No image loaded to convert to B&W.")
            return
        img = self.cv_image
        threshold_b = threshold
        threshold_w = threshold
        if img.ndim == 2:
            mask_b = img < threshold_b
            mask_w = img > threshold_w
            img[mask_b] = 0
            img[mask_w] = 255
        elif img.ndim == 3:
            mask_b = np.atleast_1d((img < threshold_b).any(axis=2))
            mask_w = np.atleast_1d((img > threshold_w).any(axis=2))
            img[mask_b] = [0, 0, 0]
            img[mask_w] = [255, 255, 255]
        else:
            QMessageBox.warning(self, "Invalid Image", "Image format not supported for B&W conversion.")
            return
        self.cv_image = img
        self.viewer.display_image(self.cv_image)


    def colorify_button_fn(self):
        """Prompt user for number of splits to colorify the image."""
        value, ok = QInputDialog.getInt(
            self, "Colorify Value Input", "Enter number of splits:", min=1, max=256
        )
        if ok:
            self.colorify_image(value)
            print("User entered:", value)


    def colorify_image(self, n):
        """Colorify the image by quantizing RGB channels."""
        if self.cv_image is None:
            QMessageBox.warning(self, "No Image", "No image loaded to colorify.")
            return
        img = self.cv_image

        def _quantize_channel(channel, n):
            """Quantize a single channel into n levels."""
            out = np.zeros_like(channel)
            for i in range(0, n):
                high = 256 // n * (i + 1)
                low = 256 // n * i
                out[(channel >= low) & (channel < high)] = (high + low) // 2
            return out

        r = _quantize_channel(img[:, :, 0], n)
        g = _quantize_channel(img[:, :, 1], n)
        b = _quantize_channel(img[:, :, 2], n)
        quantized_img = np.stack((r, g, b), axis=2)
        self.cv_image = quantized_img
        self.viewer.display_image(self.cv_image)


    def _create_toolbar(self) -> QToolBar:
        """Create a toolbar for the application."""
        toolbar: QToolBar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setOrientation(Qt.Orientation.Horizontal)
        toolbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        toolbar.setMinimumHeight(40)
        toolbar.setStyleSheet(
            """
                QToolBar {
                    spacing: 10px;
                    padding: 5px;
                }
            """
        )
        return toolbar


    def _populate_toolbar(self, toolbar: QToolBar, buttons: list[tuple[str, Callable]]):
        """
        Creates and adds buttons to a toolbar from a list of configurations.

        Args:
            toolbar: The QToolBar to add buttons to.
            buttons: A list of tuples, where each tuple is (button_text, slot_function).
        """
        for text, slot in buttons:
            button = QPushButton(text)
            button.clicked.connect(slot)
            toolbar.addWidget(button)


    def set_image(self):
        """Reset the image to the original."""
        if not self.img_path:
            QMessageBox.warning(self, "No Image", "No image loaded to reset.")
            return
        self.cv_image = cv2.imread(self.img_path)
        if self.cv_image is None:
            return
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.viewer.display_image(self.cv_image)

if __name__ == "__main__":  # pragma: no cover
    app: QApplication = QApplication([])

    window: ImageApp = ImageApp()
    window.show()

    app.exec()
