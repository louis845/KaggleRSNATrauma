"""Use PySide2 to create a GUI to visualize CT scan data, where we use pydicom to read the CT scan data."""
import os
import numpy as np
import pandas as pd

import h5py
import pandas as pd

from PySide2.QtWidgets import QApplication, QMainWindow, QFrame, QTreeView, QSlider, QLabel, QHBoxLayout, QVBoxLayout, \
    QMessageBox, QComboBox, QWidget, QFileDialog, QInputDialog
from PySide2.QtCore import Qt
from PySide2.QtGui import QStandardItemModel, QStandardItem, QPainter, QColor, QBrush, QLinearGradient, QGradient, QPen
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class SliceRenderer(QWidget):
    """A widget that renders the slicewise segmentation classes."""

    def __init__(self, slices_length: int, current_index: int, parent, left, right):
        super().__init__(parent)
        self.slices_length = slices_length
        self.current_index = current_index

        self.left = left
        self.right = right

    def getXPos(self, index):
        return index * self.width() / self.slices_length

    def drawVerticalLine(self, painter: QPainter, x_pos):
        painter.drawLine(x_pos, 0, x_pos, self.height())

    def drawSliceInfo(self, painter: QPainter):
        left_x_pos = self.getXPos(self.left)
        right_x_pos = self.getXPos(self.right)

        painter.setBrush(Qt.black)

        # draw rectangle
        painter.drawRect(left_x_pos, int(self.height() / 3), right_x_pos - left_x_pos, int(self.height() / 3))

    def paintEvent(self, event):
        """Paint the slice organ image"""
        painter = QPainter(self)

        # Draw a white background
        painter.setBrush(Qt.white)
        painter.drawRect(0, 0, self.width(), self.height())

        # Draw slice info
        self.drawSliceInfo(painter)

        # Draw current index
        painter.setPen(Qt.black)
        self.drawVerticalLine(painter, self.getXPos(self.current_index))

    def setIndex(self, index):
        self.current_index = index
        self.update()


class ProcessedStage1CTViewer(QMainWindow):
    series_ct: dict[str, list[str]]  # Contains a list of CT scan series for each patient.
    ct_3D_image: np.ndarray  # Contains the 3D CT scan data. Shape is (z, y, x).
    z_positions: np.ndarray  # Contains the z positions of the slices. Shape is (z,).

    fig: plt.Figure
    image_canvas: FigureCanvas

    def __init__(self, ct_3D_image, z_positions, segmentation_image, organ_left, organ_right):
        super().__init__()

        assert ct_3D_image.shape[0] == z_positions.shape[0]
        assert ct_3D_image.shape[-1] == segmentation_image.shape[-1]
        assert ct_3D_image.shape[-2] == segmentation_image.shape[-2]

        self.ct_3D_image = ct_3D_image
        self.z_positions = z_positions
        self.segmentation_image = segmentation_image
        self.organ_left = organ_left
        self.organ_right = organ_right
        self.rot_options_length = segmentation_image.shape[1]
        self.rot_options = 0

        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        # Create the main window
        self.setWindowTitle("Processed stage1 CT Viewer")
        self.resize(1920, 1080)
        self.main_layout = QHBoxLayout()
        # Main panel
        self.main_panel = QFrame()

        # Add the left panel and main panel to the main layout
        self.main_layout.addWidget(self.main_panel)

        # Set the main layout
        self.main_widget = QFrame()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        # Create a matplotlib canvas to display the image
        self.fig = plt.figure()
        self.image_canvas = FigureCanvas(self.fig)
        # Create the renderer for the slice information
        self.slice_info_renderer = SliceRenderer(self.ct_3D_image.shape[0], 0, self.main_panel, left=self.organ_left, right=self.organ_right)
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
        self.slice_number_slider.setRange(0, self.ct_3D_image.shape[0] - 1)
        self.slice_number_slider.setValue(0)
        self.slice_number_slider.setFixedHeight(self.slice_number_slider.sizeHint().height())

        # Create a dropdown to select the rot options
        self.rot_options_dropdown = QComboBox()
        self.rot_options_dropdown.addItems([str(k) for k in range(self.rot_options_length)])

        # Add the slider and label to the main panel
        self.main_panel_layout = QVBoxLayout()
        self.main_panel_layout.addWidget(self.image_canvas)
        self.main_panel_layout.addWidget(self.rot_options_dropdown)
        self.main_panel_layout.addWidget(self.slice_info_renderer)
        self.main_panel_layout.addWidget(self.slice_number_label)
        self.main_panel_layout.addWidget(self.slice_number_slider)

        # Set the layout of the main panel
        self.main_panel.setLayout(self.main_panel_layout)

    def setup_connections(self):
        self.slice_number_slider.valueChanged.connect(self.slice_number_updated)
        self.rot_options_dropdown.currentIndexChanged.connect(self.rot_options_changed)

    def slice_number_updated(self, value: int):
        if self.z_positions is None:
            z_pos = 0
        else:
            z_pos = self.z_positions[value]
        self.slice_info_renderer.setIndex(value)
        self.slice_number_label.setText("Slice Number: {}   z-position: {}".format(value, z_pos))
        self.update_image()

    def rot_options_changed(self, value: int):
        self.rot_options = int(value)
        self.update_image()

    def update_image(self):
        if self.ct_3D_image is not None:
            # Get the slice number
            slice_number = self.slice_number_slider.value()
            # Get the image
            image = self.ct_3D_image[slice_number]
            seg_img = self.segmentation_image[slice_number, self.rot_options, ...].astype(dtype=np.uint8) * 255
            seg_img = np.stack([np.zeros_like(seg_img), seg_img, seg_img], axis=-1)

            # Plot it on the image canvas, which is a matplotlib widget. update self.fig. make sure to explicitly set size of the plot to equal to the size of the image
            self.fig.clear()
            if self.segmentation_image is None:
                self.fig.set_size_inches(image.shape[1] / 100, image.shape[0] / 100)
                self.fig.add_subplot(1, 1, 1).imshow(image, cmap="gray")
            else:
                ax_ct = self.fig.add_subplot(1, 3, 1)
                ax_ct.imshow(image, cmap="gray")

                ax_seg = self.fig.add_subplot(1, 3, 2)
                ax_seg.imshow(seg_img)

                ax_overlay = self.fig.add_subplot(1, 3, 3)
                ax_overlay.imshow(image, cmap="gray")
                ax_overlay.imshow(seg_img, alpha=0.5)

            self.image_canvas.draw()

    def closeEvent(self, event):
        pass


if __name__ == "__main__":
    app = QApplication([])

    # show a prompt box with two options "Load external" or "Load internal"
    response = QMessageBox.question(None, "Load CT", "Load external or internal CT?", QMessageBox.Yes | QMessageBox.No)

    if response == QMessageBox.Yes:
        # show a file chooser to prompt the user to select an existing .npy file
        file_path, _ = QFileDialog.getOpenFileName(None, "Select the z-positions of processed stage1 CT file", "", "Numpy files (*.npy)")
        if file_path == "":
            exit(0)
        parent_folder = os.path.dirname(file_path)
        subfiles = os.listdir(parent_folder)
        if not "ct_3D_image.hdf5" in subfiles:
            QMessageBox.critical(None, "Error", "Cannot find ct_3D_image.hdf5 in the same folder as the z-positions file")
            exit(0)
        if not "organ_segmentations.hdf5" in subfiles:
            QMessageBox.critical(None, "Error", "Cannot find organ_segmentations.hdf5 in the same folder as the z-positions file")
            exit(0)

        ct_3D_image_file = h5py.File(os.path.join(parent_folder, "ct_3D_image.hdf5"), "r")
        ct_3D_image = ct_3D_image_file["ct_3D_image"][...]
        ct_3D_image_file.close()

        organ_segmentations_file = h5py.File(os.path.join(parent_folder, "organ_segmentations.hdf5"), "r")
        organ_segmentations = organ_segmentations_file["organ_segmentations"][...]
        organ_segmentations_file.close()

        z_positions = np.load(file_path)

        series_id = int(os.path.basename(parent_folder))
        dataset_path = os.path.dirname(os.path.dirname(os.path.dirname(parent_folder)))
        dataset_name = os.path.basename(dataset_path)
        dataset_name = dataset_name.split("_")
        organ_name = dataset_name[-1]
        dataset_name = "_".join(dataset_name[:-1])
        print("dataset_name: ", dataset_name)
        print("organ_name: ", organ_name)

        organ_file = os.path.join("EXTRACTED_STAGE1_RESULTS", "stage1_organ_segmentator", dataset_name, str(series_id) + ".csv")
        organ_info = pd.read_csv(organ_file, index_col=0)

        left, right = organ_info.loc[organ_name, "left"], organ_info.loc[organ_name, "right"]
    else:
        file_path, _ = QFileDialog.getOpenFileName(None, "Select the processed stage1 CT file", "", "HDFF files (*.hdf5)")
        if file_path == "":
            exit(0)
        ct_3D_image_file = h5py.File(file_path, "r")
        ct_3D_image = ct_3D_image_file["ct_3D_image"][...]
        ct_3D_image_file.close()

        file_path, _ = QFileDialog.getOpenFileName(None, "Select the processed segmentation file", "", "HDFF files (*.hdf5)")
        if file_path == "":
            exit(0)
        organ_segmentations_file = h5py.File(file_path, "r")
        organ_segmentations = organ_segmentations_file["organ_segmentations"][...]
        organ_segmentations_file.close()

        assert ct_3D_image.shape == organ_segmentations.shape
        ct_3D_image = np.squeeze(ct_3D_image, axis=1)
        organ_segmentations = np.squeeze(organ_segmentations, axis=1)

        # Open dialog to ask the user to select a number from 0 to ct_3D_image.shape[0] - 1 inclusive
        batch_number, ok = QInputDialog.getInt(None, "Select batch number", "Enter batch number", 0, 0, ct_3D_image.shape[0] - 1, 1)
        if not ok:
            exit(0)
        ct_3D_image = ct_3D_image[batch_number, ...]
        organ_segmentations = organ_segmentations[batch_number, ...]

        z_positions = np.arange(organ_segmentations.shape[0])

        left, right = 0, organ_segmentations.shape[0] - 1

    window = ProcessedStage1CTViewer(ct_3D_image, z_positions, organ_segmentations, left, right)
    window.show()
    app.exec_()

