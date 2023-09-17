"""Use PySide2 to create a GUI to visualize CT scan data, where we use pydicom to read the CT scan data."""
import os
import traceback
import pydicom
import nibabel
import numpy as np
import cv2
import h5py
import pandas as pd

from PySide2.QtWidgets import QApplication, QMainWindow, QFrame, QTreeView, QSlider, QLabel, QHBoxLayout, QVBoxLayout,\
    QMessageBox, QComboBox, QWidget
from PySide2.QtCore import Qt
from PySide2.QtGui import QStandardItemModel, QStandardItem, QPainter, QColor
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import multiprocessing
import scan_preprocessing
import image_sampler
import image_sampler_async

def convert_segmentation_to_color(segmentation_image: np.ndarray) -> np.ndarray:
    """Convert a segmentation image to color, in hsv format"""
    seg_float_32 = segmentation_image.astype(np.float32)
    hue = (255.0 * seg_float_32 / 5.0).astype(np.uint8)
    saturation = (seg_float_32 > 0).astype(np.uint8) * 255
    value = (seg_float_32 > 0).astype(np.uint8) * 255
    hsv_image = np.stack([hue, saturation, value], axis=-1)
    return hsv_image

def getHSVColor(segmentation_class: int) -> tuple[int, int, int]:
    shifted_class = segmentation_class + 1
    hue = (255.0 * shifted_class / 5.0)
    saturation = 255
    value = 255
    return (int(hue), int(saturation), int(value))

class OrgansSliceInfo:
    def __init__(self):
        self.loaded = False

        self.organs_available = np.zeros(shape=(4,), dtype=bool)
        self.organ_left = np.zeros(shape=(4,), dtype=np.int32)
        self.organ_right = np.zeros(shape=(4,), dtype=np.int32)

    def load_from_csv(self, filepath: str):
        df = pd.read_csv(filepath)
        self.organs_available[:] = df["found"].values
        self.organ_left[:] = df["left"].values
        self.organ_right[:] = df["right"].values

    def set_from_segmentation(self, segmentation_arr: np.ndarray):
        self.organs_available[:] = False
        self.organ_left[:] = 0
        self.organ_right[:] = 0

        for organ_index in range(4):
            organ_volume = segmentation_arr[..., organ_index]
            organ_slice = np.any(organ_volume, axis=(-1, -2))
            organ_slice_indices = np.argwhere(organ_slice).flatten()
            if np.any(organ_slice):
                self.organs_available[organ_index] = True
                self.organ_left[organ_index] = organ_slice_indices[0]
                self.organ_right[organ_index] = organ_slice_indices[-1]

        self.loaded = True

class SliceOrganRenderer(QWidget):
    """A widget that renders the slicewise segmentation classes."""
    def __init__(self, slices_length: int, current_index: int,
                 parent=None, gt_organ_slice_info: OrgansSliceInfo=None, pred_organ_slice_info: OrgansSliceInfo=None):
        super().__init__(parent)
        self.slices_length = slices_length
        self.current_index = current_index

        self.gt_organ_slice_info = gt_organ_slice_info
        self.pred_organ_slice_info = pred_organ_slice_info

    def getXPos(self, index):
        return index * self.width() / self.slices_length

    def drawVerticalLine(self, painter: QPainter, x_pos):
        painter.drawLine(x_pos, 0, x_pos, self.height())

    def drawText(self, painter: QPainter, text, x, y):
        font_metrics = painter.fontMetrics()
        text_width = font_metrics.width(text)
        text_height = font_metrics.height()
        painter.drawText(x - text_width / 2, y + text_height / 2, text)

    def drawSliceInfo(self, painter: QPainter, organ_slice_info: OrgansSliceInfo, min_y, max_y,
                      text=" (gt)"):
        y_poses = np.linspace(min_y, max_y, 5)
        y_spacing = y_poses[1] - y_poses[0]

        # find a QFont such that the height is equal to y_spacing
        font = painter.font()
        font.setPointSizeF(y_spacing / 2)
        painter.setFont(font)
        
        organs = ["liver", "spleen", "kidney", "bowel"]
        for organ_index in range(4):
            if organ_slice_info.organs_available[organ_index]:
                left_x_pos = self.getXPos(organ_slice_info.organ_left[organ_index])
                right_x_pos = self.getXPos(organ_slice_info.organ_right[organ_index])

                # set color
                hsv_color = getHSVColor(organ_index)
                painter.setPen(Qt.NoPen)
                # convert HSV to RGB and set color
                color = QColor()
                color.setHsv(hsv_color[0], hsv_color[1], hsv_color[2])
                painter.setBrush(color)

                # draw rectangle
                painter.drawRect(left_x_pos, y_poses[organ_index], right_x_pos - left_x_pos,
                                 y_poses[organ_index + 1] - y_poses[organ_index])

                # draw text
                self.drawText(painter, organs[organ_index] + text,
                              (left_x_pos + right_x_pos) / 2, y_poses[organ_index] + y_spacing / 2)


    def paintEvent(self, event):
        """Paint the slice organ image"""
        painter = QPainter(self)

        # Draw a white background
        painter.setBrush(Qt.white)
        painter.drawRect(0, 0, self.width(), self.height())

        if (self.gt_organ_slice_info is not None) and (self.pred_organ_slice_info is not None):
            self.drawSliceInfo(painter, self.gt_organ_slice_info, 0, self.height() / 2)
            self.drawSliceInfo(painter, self.pred_organ_slice_info, self.height() / 2, self.height(), text=" (pred)")
        elif self.gt_organ_slice_info is not None:
            self.drawSliceInfo(painter, self.gt_organ_slice_info, 0, self.height())

        # Draw current index
        painter.setPen(Qt.black)
        self.drawVerticalLine(painter, self.getXPos(self.current_index))

    def setIndex(self, index):
        self.current_index = index
        self.update()


class RawCTViewer(QMainWindow):

    series_ct: dict[str, list[str]] # Contains a list of CT scan series for each patient.
    ct_3D_image: np.ndarray # Contains the 3D CT scan data. Shape is (z, y, x).
    z_positions: np.ndarray # Contains the z positions of the slices. Shape is (z,).

    fig: plt.Figure
    image_canvas: FigureCanvas

    def __init__(self):
        super().__init__()
        self.ct_folder = os.path.join("data", "train_images")
        self.ct_npy_folder = "data_npy"
        self.ct_hdf5_folder = "data_hdf5"
        self.ct_hdf5_cropped_folder = "data_hdf5_cropped"

        self.segmentations_folder = os.path.join("data", "segmentations")
        self.segmentations_cropped_folder = "data_segmentation_hdf_cropped"
        self.generated_segmentations_cropped_folder = "total_segmentator_hdf_cropped"

        self.setup_ui()
        self.setup_folders()
        self.setup_connections()

        self.ct_3D_image = None
        self.z_positions = None
        self.segmentation_image = None
        self.min_series = None
        self.max_series = None

    def setup_ui(self):
        """Should be a 1920x1080 window, with title Raw CT Viewer. This GUI should have a left panel and a main panel.
           The left panel has maximum width of 300 pixels, and should contain a tree view representing the folder structure
           of the data. The main panel displays the CT scan data. The main panel should have a slider at the bottom to
           control the slice number. Above the slider, there should be a label displaying the current slice number.
           Above the label, there should be an area to display the 2D slice of the CT scan, corrsponding to the current
              slice number."""
        # Create the main window
        self.setWindowTitle("Raw CT Viewer")
        self.resize(1920, 1080)
        self.main_layout = QHBoxLayout()

        # Left panel
        self.left_panel = QFrame()
        self.left_panel.setFixedWidth(300)
        # Main panel
        self.main_panel = QFrame()

        # Add the left panel and main panel to the main layout
        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.main_panel)

        # Set the main layout
        self.main_widget = QFrame()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        # Add the tree view to the left panel
        self.tree_view = QTreeView()
        self.left_panel_layout = QHBoxLayout()
        self.left_panel_layout.addWidget(self.tree_view)
        self.left_panel.setLayout(self.left_panel_layout)
        self.tree_view.setEditTriggers(QTreeView.NoEditTriggers)

        # Create a dropdown to toggle between dicom, rescaled_dicom, and npy
        self.image_options = QComboBox()
        self.image_options.addItems(["dicom", "rescaled_dicom", "npy", "hdf5", "hdf5_cropped", "hdf5_sampler", "hdf5_sampler_async"])
        # Create a matplotlib canvas to display the image
        self.fig = plt.figure()
        self.image_canvas = FigureCanvas(self.fig)
        # Create the renderer for the slice information
        self.slice_info_renderer = SliceOrganRenderer(10, 0)
        self.slice_info_renderer.setMaximumHeight(200)

        # Create a label for the slice number
        self.slice_number_label = QLabel()
        self.slice_number_label.setText("Slice Number: 0")
        self.slice_number_label.setAlignment(Qt.AlignCenter)
        self.slice_number_label.setFixedHeight(self.slice_number_label.fontMetrics().boundingRect(
            self.slice_number_label.text()).height())
        # Create a slider to control the slice number, horizontal
        self.slice_number_slider = QSlider()
        self.slice_number_slider.setOrientation(Qt.Horizontal)
        self.slice_number_slider.setRange(0, 100)
        self.slice_number_slider.setValue(0)
        self.slice_number_slider.setFixedHeight(self.slice_number_slider.sizeHint().height())
        # Add the slider and label to the main panel
        self.main_panel_layout = QVBoxLayout()
        self.main_panel_layout.addWidget(self.image_options)
        self.main_panel_layout.addWidget(self.image_canvas)
        self.main_panel_layout.addWidget(self.slice_info_renderer)
        self.main_panel_layout.addWidget(self.slice_number_label)
        self.main_panel_layout.addWidget(self.slice_number_slider)

        # Set the layout of the main panel
        self.main_panel.setLayout(self.main_panel_layout)

        # Create an async image sampler
        self.async_loader = image_sampler_async.ImageLoaderWorker("async_loader")

    def setup_folders(self):
        self.patient_ids = os.listdir(self.ct_folder)

        # for each folder in ct folder, list the folders in the folder
        self.series_ct = {}
        for patient_id in self.patient_ids:
            patient_folder = os.path.join(self.ct_folder, patient_id)
            series_folders = os.listdir(patient_folder)
            self.series_ct[patient_id] = series_folders

        # use series_ct to populate the tree view
        self.tree_view_model = QStandardItemModel()
        self.tree_view_model.setHorizontalHeaderLabels(["Patient ID", "Series ID"])
        self.tree_view.setModel(self.tree_view_model)

        self.patient_ids = [int(patient_id) for patient_id in self.patient_ids]
        self.patient_ids.sort()
        self.patient_ids = [str(patient_id) for patient_id in self.patient_ids]

        for patient_id in self.patient_ids:
            patient_id_item = QStandardItem(patient_id)
            self.tree_view_model.appendRow(patient_id_item)
            for series_id in self.series_ct[patient_id]:
                series_id_item = QStandardItem(series_id)
                patient_id_item.appendRow(series_id_item)

        self.tree_view.expandAll()

    def setup_connections(self):
        self.slice_number_slider.valueChanged.connect(self.slice_number_updated)
        self.tree_view.doubleClicked.connect(self.tree_view_double_clicked)

    def slice_number_updated(self, value: int):
        if self.z_positions is None:
            z_pos = 0
        else:
            z_pos = self.z_positions[value - self.min_series]
        self.slice_number_label.setText("Slice Number: {}   z-position: {}".format(value, z_pos))
        self.update_image()

    def tree_view_double_clicked(self, index):
        # if its series ID, load the CT scan data
        item = self.tree_view_model.itemFromIndex(index)
        if item.parent() is not None:
            patient_id = item.parent().text()
            series_id = item.text()
            self.load_ct_scan(patient_id, series_id)

    def load_ct_scan(self, patient_id: str, series_id: str):
        series_folder = os.path.join(self.ct_folder, patient_id, series_id)
        series_folder_npy = os.path.join(self.ct_npy_folder, patient_id, series_id)
        series_folder_hdf5 = os.path.join(self.ct_hdf5_folder, patient_id, series_id)
        series_folder_hdf5_cropped = os.path.join(self.ct_hdf5_cropped_folder, patient_id, series_id)
        ct_scan_files = [int(dcm[:-4]) for dcm in os.listdir(series_folder)]
        ct_scan_files.sort()

        min_slice = ct_scan_files[0]
        max_slice = ct_scan_files[-1]

        self.min_series = min_slice
        self.max_series = max_slice

        self.ct_3D_image = None
        self.segmentation_image = None
        self.slice_info_renderer.gt_organ_slice_info = None
        self.slice_info_renderer.pred_organ_slice_info = None
        # Load the data
        if self.image_options.currentText() == "dicom":
            print("Loading DICOM data...")
            # load from dicom file
            self.ct_3D_image = None
            self.z_positions = np.zeros((max_slice - min_slice + 1,), dtype=np.float32)
            for slice_number in range(min_slice, max_slice + 1):
                dcm_file = os.path.join(series_folder, "{}.dcm".format(slice_number))
                dcm_data = pydicom.dcmread(dcm_file)
                slice_array = scan_preprocessing.to_float_array(dcm_data)
                if self.ct_3D_image is None:
                    self.ct_3D_image = np.zeros(
                        (max_slice - min_slice + 1, slice_array.shape[0], slice_array.shape[1]),
                        dtype=np.uint8)
                self.ct_3D_image[slice_number - min_slice, :, :] = (slice_array * 255).astype(np.uint8)
                self.z_positions[slice_number - min_slice] = dcm_data[(0x20, 0x32)].value[-1]

            segmentation_file = os.path.join(self.segmentations_folder, str(series_id) + ".nii")
            if os.path.isfile(segmentation_file):
                segmentation_image = np.array(nibabel.load(segmentation_file).get_fdata()).astype(np.uint8).transpose(2, 0, 1)
                segmentation_image = segmentation_image[::-1, ...]
                segmentation_image = np.rot90(segmentation_image, axes=(1, 2), k=1)
                self.segmentation_image = convert_segmentation_to_color(segmentation_image)
        elif self.image_options.currentText() == "rescaled_dicom":
            print("Loading rescaled DICOM data...")
            # load from dicom file
            self.ct_3D_image = None
            self.z_positions = np.zeros((max_slice - min_slice + 1,), dtype=np.float32)
            shape = None
            for slice_number in range(min_slice, max_slice + 1):
                dcm_file = os.path.join(series_folder, "{}.dcm".format(slice_number))
                dcm_data = pydicom.dcmread(dcm_file)
                slice_array = scan_preprocessing.to_float_array(dcm_data)
                if shape is None:
                    scales = np.array(dcm_data.PixelSpacing)
                    shape = slice_array.shape
                    new_shape = (int(shape[0] * scales[0]), int(shape[1] * scales[1]))
                    slice_array = cv2.resize(slice_array, (new_shape[1], new_shape[0]))
                    shape = slice_array.shape
                else:
                    slice_array = cv2.resize(slice_array, (shape[1], shape[0]))
                if self.ct_3D_image is None:
                    self.ct_3D_image = np.zeros(
                        (max_slice - min_slice + 1, slice_array.shape[0], slice_array.shape[1]),
                        dtype=np.uint8)
                self.ct_3D_image[slice_number - min_slice, :, :] = (slice_array * 255).astype(np.uint8)
                self.z_positions[slice_number - min_slice] = dcm_data[(0x20, 0x32)].value[-1]
        elif self.image_options.currentText() == "npy":
            print("Loading NPY data...")
            if os.path.isfile(os.path.join(series_folder_npy, "ct_3D_image.npy")):
                # load from npy file
                self.ct_3D_image = (np.load(os.path.join(series_folder_npy, "ct_3D_image.npy")) * 255).astype(np.uint8)
                self.z_positions = np.load(os.path.join(series_folder_npy, "z_positions.npy"))
            else:
                # show dialog "NPY data not found. If you haven't generated the NPY files, please run convert_to_npy.py. If you have converted the NPY to HDF5, please view them with HDF5 option."
                QMessageBox.information(self, "NPY data not found",
                                        "NPY data not found. If you haven't generated the NPY files, please run convert_to_npy.py. If you have converted the NPY to HDF5, please view them with HDF5 option.")
                return
        elif self.image_options.currentText() == "hdf5":
            print("Loading HDF5 data...")
            # load from HDF5 file
            if not os.path.isfile(os.path.join(series_folder_hdf5, "ct_3D_image.hdf5")):
                # show dialog "HDF5 data not found. If you haven't generated the HDF5 files, please run convert_to_hdf5.py."
                QMessageBox.information(self, "HDF5 data not found",
                                        "HDF5 data not found. If you haven't generated the HDF5 files, please run convert_to_hdf5.py.")
                return
            else:
                with h5py.File(os.path.join(series_folder_hdf5, "ct_3D_image.hdf5"), "r") as f:
                    self.ct_3D_image = f["ct_3D_image"][()]
                self.z_positions = np.load(os.path.join(series_folder_hdf5, "z_positions.npy"))
        elif self.image_options.currentText() == "hdf5_cropped":
            print("Loading HDF5 data...")
            # load from HDF5 file
            if not os.path.isfile(os.path.join(series_folder_hdf5_cropped, "ct_3D_image.hdf5")):
                # show dialog "HDF5 data not found. If you haven't generated the HDF5 files, please run convert_to_hdf5.py."
                QMessageBox.information(self, "HDF5 data not found",
                                        "HDF5 data not found. If you haven't generated the HDF5 files, please run convert_to_hdf5.py.")
                return
            else:
                with h5py.File(os.path.join(series_folder_hdf5_cropped, "ct_3D_image.hdf5"), "r") as f:
                    self.ct_3D_image = f["ct_3D_image"][()]
                self.z_positions = np.load(os.path.join(series_folder_hdf5_cropped, "z_positions.npy"))
                if os.path.isfile(os.path.join(self.segmentations_cropped_folder, series_id + ".hdf5")):
                    with h5py.File(os.path.join(self.segmentations_cropped_folder, series_id + ".hdf5"), "r") as f:
                        segmentation_3D_image = f["segmentation_arr"][()]
                    self.slice_info_renderer.gt_organ_slice_info = OrgansSliceInfo()
                    self.slice_info_renderer.gt_organ_slice_info.set_from_segmentation(segmentation_3D_image)
                    self.segmentation_image = convert_segmentation_to_color(np.any(segmentation_3D_image, axis=-1) * (np.argmax(segmentation_3D_image, axis=-1) + 1))
                    assert segmentation_3D_image.shape[:3] == self.ct_3D_image.shape
                elif os.path.isfile(os.path.join(self.generated_segmentations_cropped_folder, series_id + ".hdf5")):
                    with h5py.File(os.path.join(self.generated_segmentations_cropped_folder, series_id + ".hdf5"), "r") as f:
                        segmentation_3D_image = f["segmentation_arr"][()]
                    self.slice_info_renderer.gt_organ_slice_info = OrgansSliceInfo()
                    self.slice_info_renderer.gt_organ_slice_info.set_from_segmentation(segmentation_3D_image)
                    self.segmentation_image = convert_segmentation_to_color(np.any(segmentation_3D_image, axis=-1) * (np.argmax(segmentation_3D_image, axis=-1) + 1))
                    assert segmentation_3D_image.shape[:3] == self.ct_3D_image.shape
        elif self.image_options.currentText() == "hdf5_sampler":
            print("Loading HDF5 sampler data...")
            self.ct_3D_image = image_sampler.load_image(patient_id, series_id, slices_random=True, augmentation=True)
            self.z_positions = np.zeros((15,), dtype=np.float32)
            self.min_series, self.max_series = 0, 14
            min_slice, max_slice = 0, 14
        elif self.image_options.currentText() == "hdf5_sampler_async":
            print("Loading HDF5 sampler async...")
            self.async_loader.request_load_image({"patient_id": patient_id, "series_id": series_id, "slices_random": True, "augmentation": True})
            self.ct_3D_image = self.async_loader.get_requested_image()
            self.z_positions = np.zeros((15,), dtype=np.float32)
            self.min_series, self.max_series = 0, 14
            min_slice, max_slice = 0, 14


        # Update the slider and the renderer
        self.slice_number_slider.setRange(min_slice, max_slice)
        self.slice_number_slider.setValue(min_slice)
        self.slice_info_renderer.slices_length = max_slice - min_slice + 1
        self.slice_info_renderer.current_index = 0

        # Update the image
        self.update_image()

        # Show dialog with message "CT scan loaded, Series {}, Patient {}".format(series_id, patient_id)
        QMessageBox.information(self, "CT scan loaded", "CT scan loaded, Series {}, Patient {}".format(series_id, patient_id))

    def update_image(self):
        if self.ct_3D_image is not None:
            # Get the slice number
            slice_number = self.slice_number_slider.value()
            # Get the image
            image = self.ct_3D_image[slice_number - self.min_series]

            # Plot it on the image canvas, which is a matplotlib widget. update self.fig. make sure to explicitly set size of the plot to equal to the size of the image
            self.fig.clear()
            if self.segmentation_image is None:
                self.fig.set_size_inches(image.shape[1] / 100, image.shape[0] / 100)
                self.fig.add_subplot(1, 1, 1).imshow(image, cmap="gray")
            else:
                ax_ct = self.fig.add_subplot(1, 4, 1)
                ax_ct.imshow(image, cmap="gray")

                seg_img = self.segmentation_image[slice_number - self.min_series, ...]
                seg_img = cv2.cvtColor(seg_img, cv2.COLOR_HSV2RGB)
                ax_seg = self.fig.add_subplot(1, 4, 2)
                ax_seg.imshow(seg_img)

                ax_overlay = self.fig.add_subplot(1, 4, 3)
                ax_overlay.imshow(image, cmap="gray")
                ax_overlay.imshow(seg_img, alpha=0.5)

                ax_colors = self.fig.add_subplot(1, 4, 4)
                color_image = np.zeros((self.segmentation_image.shape[0], self.segmentation_image.shape[1]), dtype=np.uint8)
                color_image[:1 * self.segmentation_image.shape[0] // 5, :] = 1
                color_image[1 * self.segmentation_image.shape[0] // 5:2 * self.segmentation_image.shape[0] // 5, :] = 2
                color_image[2 * self.segmentation_image.shape[0] // 5:3 * self.segmentation_image.shape[0] // 5, :] = 3
                color_image[3 * self.segmentation_image.shape[0] // 5:4 * self.segmentation_image.shape[0] // 5, :] = 4
                color_image[4 * self.segmentation_image.shape[0] // 5:5 * self.segmentation_image.shape[0] // 5, :] = 5
                color_image = convert_segmentation_to_color(color_image)
                color_image = cv2.cvtColor(color_image, cv2.COLOR_HSV2RGB)
                ax_colors.imshow(color_image)

                self.fig.set_size_inches(image.shape[1] / 100 * 4, image.shape[0] / 100)


            self.image_canvas.draw()

    def closeEvent(self, event):
        self.async_loader.terminate()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    app = QApplication([])
    window = RawCTViewer()
    window.show()
    app.exec_()
