"""Use PySide2 to create a GUI to visualize CT scan data, where we use pydicom to read the CT scan data."""
import os
import traceback
import pydicom
import nibabel
import numpy as np
import cv2
import h5py

from PySide2.QtWidgets import QApplication, QMainWindow, QFrame, QTreeView, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QMessageBox, QComboBox, QFileDialog
from PySide2.QtCore import Qt, QThread, Signal
from PySide2.QtGui import QStandardItemModel, QStandardItem, QPixmap, QImage
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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

class DualSegmentationViewer(QMainWindow):

    series_ct: dict[str, list[str]] # Contains a list of CT scan series for each patient.
    z_positions: np.ndarray # Contains the z positions of the slices. Shape is (z,).

    fig: plt.Figure
    image_canvas: FigureCanvas

    def __init__(self, segmentation_image1: np.ndarray, segmentation_image2: np.ndarray):
        super().__init__()
        self.segmentation_image1 = segmentation_image1
        self.segmentation_image2 = segmentation_image2

        self.setup_ui()
        self.setup_connections()

        self.z_positions = np.arange(segmentation_image1.shape[0])
        self.min_series = 0
        self.max_series = segmentation_image1.shape[0] - 1
        self.slice_number_slider.setRange(self.min_series, self.max_series)

    def setup_ui(self):
        """
        Set up the UI.
        """
        # Create the main window
        self.setWindowTitle("Raw CT Viewer")
        self.resize(1920, 1080)

        # Set the main layout
        self.main_widget = QFrame()
        self.setCentralWidget(self.main_widget)

        # Create a matplotlib canvas to display the image
        self.fig = plt.figure()
        self.image_canvas = FigureCanvas(self.fig)
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
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_canvas)
        self.main_layout.addWidget(self.slice_number_label)
        self.main_layout.addWidget(self.slice_number_slider)

        self.main_widget.setLayout(self.main_layout)

    def setup_connections(self):
        self.slice_number_slider.valueChanged.connect(self.slice_number_updated)

    def slice_number_updated(self, value: int):
        if self.z_positions is None:
            z_pos = 0
        else:
            z_pos = self.z_positions[value - self.min_series]
        self.slice_number_label.setText("Slice Number: {}   z-position: {}".format(value, z_pos))
        self.update_image()

    def update_image(self):
        # Get the slice number
        slice_number = self.slice_number_slider.value()
        # Get the image
        seg1 = self.segmentation_image1[slice_number - self.min_series, ...]
        seg2 = self.segmentation_image2[slice_number - self.min_series, ...]
        seg1 = cv2.cvtColor(seg1, cv2.COLOR_HSV2RGB)
        seg2 = cv2.cvtColor(seg2, cv2.COLOR_HSV2RGB)

        self.fig.clear()
        ax_ct1 = self.fig.add_subplot(1, 2, 1)
        ax_ct1.imshow(seg1)

        ax_ct2 = self.fig.add_subplot(1, 2, 2)
        ax_ct2.imshow(seg2)

        self.fig.set_size_inches(seg1.shape[1] / 100 * 2, seg1.shape[0] / 100)
        self.image_canvas.draw()

if __name__ == "__main__":
    app = QApplication([])

    # prompt the user to select the nii file
    nii_file1, _ = QFileDialog.getOpenFileName(None, "Select the first nii file", "", "Nifti Files (*.nii)")
    if not nii_file1:
        exit(0)
    nii_file2, _ = QFileDialog.getOpenFileName(None, "Select the second nii file", "", "Nifti Files (*.nii)")
    if not nii_file2:
        exit(0)

    # load the nii file
    nii_image1 = np.array(nibabel.load(nii_file1).get_fdata()).transpose(2, 0, 1).astype(np.uint8)
    nii_image2 = np.array(nibabel.load(nii_file2).get_fdata()).transpose(2, 0, 1).astype(np.uint8)
    if not nii_image1.shape == nii_image2.shape:
        # open prompt "The two nii files have different shapes. Exiting."
        QMessageBox.critical(None, "Error", "The two nii files have different shapes. Exiting.")
        exit(0)

    nii_image1 = nii_image1[::-1, ...]
    nii_image2 = nii_image2[::-1, ...]
    nii_image1 = np.rot90(nii_image1, axes=(1, 2), k=1)
    nii_image2 = np.rot90(nii_image2, axes=(1, 2), k=1)

    # convert image2 labels
    nii_image2 = (nii_image2 == 5).astype(np.uint8) + (nii_image2 == 1).astype(np.uint8) * 2\
                    + (nii_image2 == 3).astype(np.uint8) * 3 + (nii_image2 == 2).astype(np.uint8) * 4 + ((nii_image2 >= 55) & (nii_image2 <= 57)).astype(np.uint8) * 5

    # convert to color
    seg1 = convert_segmentation_to_color(nii_image1)
    seg2 = convert_segmentation_to_color(nii_image2)

    window = DualSegmentationViewer(seg1, seg2)
    window.show()
    app.exec_()
