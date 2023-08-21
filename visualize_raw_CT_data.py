"""Use PySide2 to create a GUI to visualize CT scan data, where we use pydicom to read the CT scan data."""
import os
import traceback
import pydicom
import numpy as np
import cv2
import h5py

from PySide2.QtWidgets import QApplication, QMainWindow, QFrame, QTreeView, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QMessageBox, QComboBox
from PySide2.QtCore import Qt, QThread, Signal
from PySide2.QtGui import QStandardItemModel, QStandardItem, QPixmap, QImage
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import multiprocessing
import scan_preprocessing
import image_sampler
import image_sampler_async

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

        self.setup_ui()
        self.setup_folders()
        self.setup_connections()

        self.ct_3D_image = None
        self.z_positions = None
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
        self.image_options.addItems(["dicom", "rescaled_dicom", "npy", "hdf5", "hdf5_sampler", "hdf5_sampler_async"])
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
        self.main_panel_layout = QVBoxLayout()
        self.main_panel_layout.addWidget(self.image_options)
        self.main_panel_layout.addWidget(self.image_canvas)
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
        ct_scan_files = [int(dcm[:-4]) for dcm in os.listdir(series_folder)]
        ct_scan_files.sort()

        min_slice = ct_scan_files[0]
        max_slice = ct_scan_files[-1]

        self.min_series = min_slice
        self.max_series = max_slice

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


        # Update the slider
        self.slice_number_slider.setRange(min_slice, max_slice)
        self.slice_number_slider.setValue(min_slice)

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
            self.fig.set_size_inches(image.shape[1] / 100, image.shape[0] / 100)
            self.fig.add_subplot(111).imshow(image, cmap="gray")
            self.image_canvas.draw()

    def closeEvent(self, event):
        self.async_loader.terminate()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    app = QApplication([])
    window = RawCTViewer()
    window.show()
    app.exec_()
