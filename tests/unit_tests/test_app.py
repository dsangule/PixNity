"""Unit tests for the PixNity application."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from PyQt6.QtCore import (
    QPoint,
    QRectF,
    QSize,
    Qt,
)
from PyQt6.QtGui import (
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QPushButton,
    QSizePolicy,
    QToolBar,
)
from pytest_mock import MockerFixture

from PixNity.app import (
    ImageApp,
    ImageViewer,
)


@pytest.fixture
def viewer(qtbot) -> ImageViewer:
    """A pytest fixture to create a clean ImageViewer instance for each test."""
    test_viewer = ImageViewer()
    qtbot.addWidget(test_viewer)
    return test_viewer


@pytest.fixture
def app(qtbot) -> ImageApp:
    """A pytest fixture to create an instance of our app for testing."""
    test_app = ImageApp()
    qtbot.addWidget(test_app)
    return test_app


def test_init_sets_properties(viewer: ImageViewer):
    """Tests that the viewer is initialized with the correct settings."""
    assert isinstance(viewer.scene(), QGraphicsScene)
    assert viewer.dragMode() == QGraphicsView.DragMode.ScrollHandDrag
    assert viewer.transformationAnchor() == QGraphicsView.ViewportAnchor.AnchorUnderMouse
    assert viewer.resizeAnchor() == QGraphicsView.ViewportAnchor.AnchorUnderMouse


def test_display_image_updates_pixmap_and_scene(viewer: ImageViewer):
    """Tests that a numpy image is correctly converted and displayed."""
    fake_image = np.zeros((50, 100, 3), dtype=np.uint8)

    viewer.display_image(fake_image)

    pixmap = viewer.image_item.pixmap()
    assert not pixmap.isNull()
    assert pixmap.width() == 100
    assert pixmap.height() == 50
    assert viewer.sceneRect() == QRectF(0, 0, 100, 50)


def test_wheel_event_zooms_in(viewer: ImageViewer, mocker: MockerFixture):
    """Tests that a positive wheel scroll calls scale() to zoom in."""
    mock_scale: MagicMock = mocker.patch.object(viewer, 'scale')
    mock_event = MagicMock(spec=QWheelEvent)
    mock_event.angleDelta.return_value = QPoint(0, 120)

    viewer.wheel_event(mock_event)

    mock_scale.assert_called_once_with(viewer.scale_factor, viewer.scale_factor)


def test_wheel_event_zooms_out(viewer: ImageViewer, mocker: MockerFixture):
    """Tests that a negative wheel scroll calls scale() to zoom out."""
    mock_scale: MagicMock = mocker.patch.object(viewer, 'scale')
    mock_event = MagicMock(spec=QWheelEvent)
    mock_event.angleDelta.return_value = QPoint(0, -120)

    viewer.wheel_event(mock_event)

    zoom_out_factor = pytest.approx(1 / viewer.scale_factor)
    mock_scale.assert_called_once_with(zoom_out_factor, zoom_out_factor)


def test_resize_event_calls_fit_image(viewer: ImageViewer, mocker: MockerFixture):
    """Tests that resizing the widget triggers fit_image()."""
    mock_fit: MagicMock = mocker.patch.object(viewer, 'fit_image')
    mocker.patch.object(QGraphicsView, 'resizeEvent')
    mock_event = MagicMock(spec=QResizeEvent)
    mock_event.size.return_value = QSize(200, 200)

    viewer.resize_event(mock_event)

    mock_fit.assert_called_once()


def test_fit_image_calls_fitinview(viewer: ImageViewer, mocker: MockerFixture):
    """Tests that fit_image() calls the underlying fitInView method correctly."""
    viewer.image_item.setPixmap(QPixmap(10, 10))
    mock_fit_in_view: MagicMock = mocker.patch.object(viewer, 'fitInView')

    viewer.fit_image()

    mock_fit_in_view.assert_called_once_with(
        viewer.image_item, Qt.AspectRatioMode.KeepAspectRatio
    )


def test_load_image_button_fn_when_user_selects_file(app: ImageApp, mocker: MockerFixture):
    """Tests that the image is loaded and displayed when the user selects a file."""
    fake_path = "/fake/path/to/image.png"
    mocker.patch("PixNity.app.QFileDialog.getOpenFileName", return_value=(fake_path, None))
    mock_set_image = mocker.patch.object(app, 'set_image', autospec=True)
    mock_fit_image = mocker.patch.object(app.viewer, 'fit_image', autospec=True)

    app.load_image_button_fn()

    assert app.img_path == fake_path
    mock_set_image.assert_called_once()
    mock_fit_image.assert_called_once()


def test_load_image_button_fn_when_user_cancels(app: ImageApp, mocker: MockerFixture):
    """Tests that no action is taken when the user cancels the file dialog."""
    mocker.patch("PixNity.app.QFileDialog.getOpenFileName", return_value=("", None))
    mock_set_image = mocker.patch.object(app, 'set_image', autospec=True)
    mock_fit_image = mocker.patch.object(app.viewer, 'fit_image', autospec=True)

    app.img_path = "initial/path"
    app.load_image_button_fn()

    assert app.img_path == "initial/path"
    mock_set_image.assert_not_called()
    mock_fit_image.assert_not_called()


def test_save_image_shows_warning_if_no_image(app: ImageApp, mocker: MockerFixture):
    """
    Tests that a warning is shown if save is clicked with no image loaded.
    """
    # 1. Setup: The app fixture starts with cv_image=None, which is perfect for this test.
    mock_warning = mocker.patch("PixNity.app.QMessageBox.warning")
    mock_file_dialog = mocker.patch("PixNity.app.QFileDialog.getSaveFileName")

    # 2. Action: Call the method.
    app.save_image_button_fn()

    # 3. Assert: Check that the warning was shown and the file dialog was never opened.
    mock_warning.assert_called_once_with(app, "No Image", "No image loaded to save.")
    mock_file_dialog.assert_not_called()


def test_save_image_does_nothing_if_user_cancels_dialog(app: ImageApp, mocker: MockerFixture):
    """Tests that no action is taken if the user cancels the save dialog."""
    app.cv_image = np.zeros((10, 10, 3), dtype=np.uint8)

    mocker.patch("PixNity.app.QFileDialog.getSaveFileName", return_value=("", None))
    mock_imwrite = mocker.patch("cv2.imwrite")

    app.save_image_button_fn()

    mock_imwrite.assert_not_called()


def test_save_image_succeeds_and_shows_info_message(app: ImageApp, mocker: MockerFixture):
    """Tests that the image is saved and a success message is shown."""
    fake_image = np.zeros((10, 10, 3), dtype=np.uint8)
    app.cv_image = fake_image
    save_path = "/fake/path/image.png"

    mocker.patch("PixNity.app.QFileDialog.getSaveFileName", return_value=(save_path, None))
    mock_cvtColor = mocker.patch("cv2.cvtColor", return_value=fake_image)
    mock_imwrite = mocker.patch("cv2.imwrite", return_value=True) # Simulate success
    mock_info = mocker.patch("PixNity.app.QMessageBox.information")

    app.save_image_button_fn()

    mock_cvtColor.assert_called_once()
    mock_imwrite.assert_called_once_with(save_path, fake_image)
    mock_info.assert_called_once_with(app, "Saved", "Image saved successfully.")


def test_save_image_shows_warning_on_write_failure(app: ImageApp, mocker: MockerFixture):
    """Tests that a warning is shown if saving the image fails."""
    app.cv_image = np.zeros((10, 10, 3), dtype=np.uint8)
    save_path = "/fake/protected/image.png"

    mocker.patch("PixNity.app.QFileDialog.getSaveFileName", return_value=(save_path, None))
    mocker.patch("cv2.cvtColor")
    mocker.patch("cv2.imwrite", return_value=False) # Simulate failure
    mock_warning = mocker.patch("PixNity.app.QMessageBox.warning")
    mock_info = mocker.patch("PixNity.app.QMessageBox.information")

    app.save_image_button_fn()

    mock_warning.assert_called_once_with(app, "Save Error", "Failed to save the image.")
    mock_info.assert_not_called()


def test_sharpen_button_shows_warning_if_no_image(app: ImageApp, mocker: MockerFixture):
    """Tests that a warning is shown if sharpen is clicked with no image loaded."""
    mock_warning = mocker.patch("PixNity.app.QMessageBox.warning")
    mock_filter = mocker.patch("cv2.filter2D")

    app.sharpen_button_fn()

    mock_warning.assert_called_once_with(app, "No Image", "No image loaded to sharpen.")
    mock_filter.assert_not_called()


def test_sharpen_button_applies_filter_and_updates_viewer(app: ImageApp, mocker: MockerFixture):
    """Tests that the sharpen filter is applied and the image is updated."""
    input_image = np.array([[[10, 20, 30]]], dtype=np.uint8)
    output_image = np.array([[[15, 25, 35]]], dtype=np.uint8)
    app.cv_image = input_image

    mock_filter = mocker.patch("cv2.filter2D", return_value=output_image)
    mock_display = mocker.patch.object(app.viewer, 'display_image', autospec=True)

    app.sharpen_button_fn()

    mock_filter.assert_called_once()
    expected_kernel = np.array([[-1, -1, -1], [-1, 17, -1], [-1, -1, -1]], dtype=np.float32) / 9.0
    actual_kernel = mock_filter.call_args[0][2]
    np.testing.assert_array_almost_equal(actual_kernel, expected_kernel)
    np.testing.assert_array_equal(app.cv_image, output_image)
    mock_display.assert_called_once_with(output_image)


def test_blur_button_shows_warning_if_no_image(app: ImageApp, mocker: MockerFixture):
    """Tests that a warning is shown if blur is clicked with no image loaded."""
    mock_warning: MagicMock = mocker.patch("PixNity.app.QMessageBox.warning")
    mock_filter = mocker.patch("cv2.filter2D")

    app.blur_button_fn()

    mock_warning.assert_called_once_with(app, "No Image", "No image loaded to blur.")
    mock_filter.assert_not_called()


def test_blur_button_applies_filter_and_updates_viewer(app: ImageApp, mocker: MockerFixture):
    """Tests that the blur filter is applied and the image is updated."""
    input_image = np.array([[[100, 100, 100]]], dtype=np.uint8)
    output_image = np.array([[[50, 50, 50]]], dtype=np.uint8)
    app.cv_image = input_image

    mock_filter: MagicMock = mocker.patch("cv2.filter2D", return_value=output_image)
    mock_display: MagicMock = mocker.patch.object(app.viewer, 'display_image', autospec=True)

    app.blur_button_fn()

    mock_filter.assert_called_once()
    expected_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9.0
    actual_kernel = mock_filter.call_args[0][2]
    np.testing.assert_array_almost_equal(actual_kernel, expected_kernel)
    np.testing.assert_array_equal(app.cv_image, output_image)
    mock_display.assert_called_once_with(output_image)


def test_outline_button_shows_warning_if_no_image(app: ImageApp, mocker: MockerFixture):
    """Tests that a warning is shown if outline is clicked with no image loaded."""
    mock_warning: MagicMock = mocker.patch("PixNity.app.QMessageBox.warning")
    mock_cvtColor = mocker.patch("cv2.cvtColor")

    app.outline_button_fn()

    mock_warning.assert_called_once_with(app, "No Image", "No image loaded to outline.")
    mock_cvtColor.assert_not_called()


def test_outline_button_applies_sobel_filter_and_updates_viewer(app: ImageApp, mocker: MockerFixture):
    """Tests the full Sobel edge detection pipeline and viewer update."""
    input_image = np.zeros((10, 10, 3), dtype=np.uint8)
    gray_image = np.zeros((10, 10), dtype=np.uint8)
    sobel_image = np.zeros((10, 10), dtype=np.float64)
    final_image = np.zeros((10, 10, 3), dtype=np.uint8)
    app.cv_image = input_image

    mock_sobel: MagicMock = mocker.patch("cv2.Sobel", return_value=sobel_image)
    mock_cvtColor: MagicMock = mocker.patch(
        "cv2.cvtColor", side_effect=[gray_image, final_image]
    )
    mock_display: MagicMock = mocker.patch.object(app.viewer, 'display_image', autospec=True)

    app.outline_button_fn()

    assert mock_cvtColor.call_count == 2
    assert mock_sobel.call_count == 2
    np.testing.assert_array_equal(app.cv_image, final_image)
    mock_display.assert_called_once_with(final_image)


def test_invert_button_shows_warning_if_no_image(app: ImageApp, mocker: MockerFixture):
    """Tests that a warning is shown if invert is clicked with no image loaded."""
    mock_warning: MagicMock = mocker.patch("PixNity.app.QMessageBox.warning")
    mock_bitwise_not = mocker.patch("cv2.bitwise_not")

    app.invert_button_fn()

    mock_warning.assert_called_once_with(app, "No Image", "No image loaded to invert.")
    mock_bitwise_not.assert_not_called()


def test_invert_button_inverts_image_and_updates_viewer(app: ImageApp, mocker: MockerFixture):
    """Tests that the image is inverted and the viewer is updated."""
    input_image = np.array([[[0, 100, 255]]], dtype=np.uint8)
    output_image = np.array([[[255, 155, 0]]], dtype=np.uint8)
    app.cv_image = input_image

    mock_bitwise_not: MagicMock = mocker.patch("cv2.bitwise_not", return_value=output_image)
    mock_display: MagicMock = mocker.patch.object(app.viewer, 'display_image', autospec=True)

    app.invert_button_fn()

    mock_bitwise_not.assert_called_once_with(input_image)
    np.testing.assert_array_equal(app.cv_image, output_image)
    mock_display.assert_called_once_with(output_image)


def test_convert_to_bw_shows_warning_if_no_image(app: ImageApp, mocker: MockerFixture):
    """Tests that a warning is shown if there is no image to convert."""
    mock_warning: MagicMock = mocker.patch("PixNity.app.QMessageBox.warning")
    app.cv_image = None

    app.convert_to_bw(threshold=128)

    mock_warning.assert_called_once_with(
        app, "No Image", "No image loaded to convert to B&W."
    )

def test_convert_to_bw_applies_threshold_to_3d_image(app: ImageApp, mocker: MockerFixture):
    """Tests that the B&W threshold is correctly applied to a color image."""
    input_image = np.array([
        [[50, 50, 50], [200, 200, 200]]
    ], dtype=np.uint8)
    app.cv_image = input_image.copy()
    expected_image = np.array([
        [[0, 0, 0], [255, 255, 255]]
    ], dtype=np.uint8)

    mock_display: MagicMock = mocker.patch.object(app.viewer, 'display_image', autospec=True)

    app.convert_to_bw(threshold=128)

    mock_display.assert_called_once()
    np.testing.assert_array_equal(app.cv_image, expected_image)


def test_convert_to_bw_applies_threshold_to_2d_image(app: ImageApp, mocker: MockerFixture):
    """Tests that the B&W threshold is correctly applied to a 2D (grayscale) image."""
    input_image = np.array([[50, 200]], dtype=np.uint8)
    app.cv_image = input_image.copy()
    expected_image = np.array([[0, 255]], dtype=np.uint8)

    mock_display: MagicMock = mocker.patch.object(app.viewer, 'display_image', autospec=True)

    app.convert_to_bw(threshold=128)

    mock_display.assert_called_once()
    np.testing.assert_array_equal(app.cv_image, expected_image)


def test_convert_to_bw_shows_warning_for_invalid_image_dimensions(app: ImageApp, mocker: MockerFixture):
    """Tests that a warning is shown for an image with unsupported dimensions."""
    app.cv_image = np.array([1, 2, 3, 4], dtype=np.uint8)

    mock_warning: MagicMock = mocker.patch("PixNity.app.QMessageBox.warning")
    mock_display: MagicMock = mocker.patch.object(app.viewer, 'display_image', autospec=True)

    app.convert_to_bw(threshold=128)

    mock_warning.assert_called_once()
    mock_display.assert_not_called()


def test_bw_button_fn_calls_convert_to_bw_on_ok(app: ImageApp, mocker: MockerFixture):
    """Tests that convert_to_bw is called when the user clicks 'OK'."""
    mock_dialog_class = mocker.patch("PixNity.app.QDialog")
    mock_slider_class = mocker.patch("PixNity.app.QSlider")
    mocker.patch("PixNity.app.QLabel")
    mocker.patch("PixNity.app.QDialogButtonBox")
    mocker.patch("PixNity.app.QVBoxLayout")
    mock_dialog_class.return_value.exec.return_value = True
    mock_slider_class.return_value.value.return_value = 150
    mock_convert_fn: MagicMock = mocker.patch.object(app, 'convert_to_bw')

    app.bw_button_fn()

    mock_convert_fn.assert_called_once_with(150)

def test_bw_button_fn_does_nothing_on_cancel(app: ImageApp, mocker: MockerFixture):
    """Tests that nothing happens when the user clicks 'Cancel'."""
    mock_dialog_class = mocker.patch("PixNity.app.QDialog")
    mocker.patch("PixNity.app.QSlider")
    mocker.patch("PixNity.app.QLabel")
    mocker.patch("PixNity.app.QDialogButtonBox")
    mocker.patch("PixNity.app.QVBoxLayout")
    mock_dialog_class.return_value.exec.return_value = False

    mock_convert_fn: MagicMock = mocker.patch.object(app, 'convert_to_bw')

    app.bw_button_fn()

    mock_convert_fn.assert_not_called()


def test_colorify_image_shows_warning_if_no_image(app: ImageApp, mocker: MockerFixture):
    """Tests that a warning is shown if there is no image to colorify."""
    mock_warning: MagicMock = mocker.patch("PixNity.app.QMessageBox.warning")
    app.cv_image = None

    app.colorify_image(n=8)

    mock_warning.assert_called_once_with(app, "No Image", "No image loaded to colorify.")


def test_colorify_image_quantizes_image_correctly(app: ImageApp, mocker: MockerFixture):
    """Tests that the image quantization is applied correctly."""
    input_image = np.array([[[10, 80, 200]]], dtype=np.uint8)
    app.cv_image = input_image
    expected_image = np.array([[[32, 96, 224]]], dtype=np.uint8)

    mock_display: MagicMock = mocker.patch.object(app.viewer, 'display_image', autospec=True)

    app.colorify_image(n=4)

    mock_display.assert_called_once()
    np.testing.assert_array_equal(app.cv_image, expected_image)


def test_colorify_button_fn_calls_processing_function_on_ok(app: ImageApp, mocker: MockerFixture):
    """Tests that colorify_image is called when the user clicks 'OK'."""
    mocker.patch("PixNity.app.QInputDialog.getInt", return_value=(8, True))
    mock_colorify_fn: MagicMock = mocker.patch.object(app, 'colorify_image')

    app.colorify_button_fn()

    mock_colorify_fn.assert_called_once_with(8)

def test_colorify_button_fn_does_nothing_on_cancel(app: ImageApp, mocker: MockerFixture):
    """Tests that nothing happens when the user clicks 'Cancel'."""
    mocker.patch("PixNity.app.QInputDialog.getInt", return_value=(0, False))
    mock_colorify_fn: MagicMock = mocker.patch.object(app, 'colorify_image')

    app.colorify_button_fn()

    mock_colorify_fn.assert_not_called()


def test_create_toolbar_returns_correctly_configured_instance(app: ImageApp):
    """Tests that _create_toolbar returns a properly configured QToolBar."""
    toolbar = app._create_toolbar()

    assert isinstance(toolbar, QToolBar)
    assert not toolbar.isMovable()
    assert not toolbar.isFloatable()
    assert toolbar.orientation() == Qt.Orientation.Horizontal
    assert toolbar.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert toolbar.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed
    assert toolbar.minimumHeight() == 40
    assert "spacing: 10px" in toolbar.styleSheet()


def test_populate_toolbar_creates_and_connects_buttons(app: ImageApp, qtbot):
    """Tests that _populate_toolbar creates buttons and connects them to slots."""
    toolbar = QToolBar()
    mock_slot_one = MagicMock()
    mock_slot_two = MagicMock()

    app._populate_toolbar(
        toolbar,
        [
            ("Load", mock_slot_one),
            ("Save", mock_slot_two),
        ]
    )

    buttons = toolbar.findChildren(QPushButton)
    assert len(buttons) == 2

    load_button = buttons[0]
    assert load_button.text() == "Load"

    # Simulate a click and check if our mock function was called
    qtbot.mouseClick(load_button, Qt.MouseButton.LeftButton)
    mock_slot_one.assert_called_once()
    mock_slot_two.assert_not_called()

    save_button = buttons[1]
    assert save_button.text() == "Save"

    qtbot.mouseClick(save_button, Qt.MouseButton.LeftButton)
    mock_slot_two.assert_called_once()
    assert mock_slot_one.call_count == 1


def test_set_image_shows_warning_when_path_is_missing(app: ImageApp, mocker: MockerFixture):
    """Checks that a warning is shown when no image path is set."""
    mock_warning = mocker.patch("PixNity.app.QMessageBox.warning")

    app.set_image()

    mock_warning.assert_called_once()
    # mock_warning.assert_called_with(
    #     app, "No Image", "No image loaded to reset."
    # )


def test_set_image_does_nothing_when_path_is_bad(app: ImageApp, mocker: MockerFixture):
    """Checks that no action is taken when the image path is invalid."""
    mock_display = mocker.patch.object(app.viewer, 'display_image', autospec=True)
    mocker.patch("cv2.imread", return_value=None)

    app.img_path = "path/to/a/bad/image.png"
    app.set_image()

    mock_display.assert_not_called()


def test_set_image_succeeds_with_valid_path(app: ImageApp, mocker: MockerFixture):
    """Checks that the image is displayed correctly when a valid path is set."""
    fake_image = np.zeros((10, 10, 3), dtype=np.uint8)

    mocker.patch("cv2.imread", return_value=fake_image)
    mock_cvt = mocker.patch("cv2.cvtColor", return_value=fake_image)
    mock_display = mocker.patch.object(app.viewer, 'display_image', autospec=True)

    app.img_path = "path/to/a/good/image.png"
    app.set_image()

    mock_cvt.assert_called_once()
    mock_display.assert_called_once()
    np.testing.assert_array_equal(mock_display.call_args[0][0], fake_image)
