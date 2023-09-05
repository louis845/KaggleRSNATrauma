import numpy as np
import cv2
import torch
import torchvision.transforms.functional


def apply_random_shear(displacement_field, xory, image_width, image_height, magnitude_low, magnitude_high):
    x_low = image_width // 3
    x_high = image_width * 2 // 3
    y_low = image_height // 3
    y_high = image_height * 2 // 3

    x = np.random.randint(low=x_low, high=x_high + 1)
    y = np.random.randint(low=y_low, high=y_high + 1)
    sigma = np.random.uniform(low=100.0, high=200.0)
    magnitude = np.random.uniform(low=magnitude_low, high=magnitude_high) * np.random.choice([-1, 1])

    kernel_width = image_width
    kernel_height = image_height
    kernel = np.expand_dims(cv2.getGaussianKernel(ksize=kernel_height * 2 + 1, sigma=sigma), axis=-1)\
             * cv2.getGaussianKernel(ksize=kernel_width * 2 + 1, sigma=sigma) * magnitude

    expand_left = min(kernel_width, x)
    expand_right = min(kernel_width + 1, image_width - x)
    expand_top = min(kernel_height, y)
    expand_bottom = min(kernel_height + 1, image_height - y)

    if xory == "x":
        displacement_field[0, y - expand_top:y + expand_bottom, x - expand_left:x + expand_right, 0:1] += \
            kernel[kernel_height - expand_top:kernel_height + expand_bottom, kernel_width - expand_left:kernel_width + expand_right, :]
    else:
        displacement_field[0, y - expand_top:y + expand_bottom, x - expand_left:x + expand_right, 1:2] += \
            kernel[kernel_height - expand_top:kernel_height + expand_bottom, kernel_width - expand_left:kernel_width + expand_right, :]

def generate_displacement_field(image_width, image_height, num_kernels=7) -> np.ndarray:
    displacement_field = np.zeros(shape=(1, image_height, image_width, 2), dtype=np.float32)

    type = np.random.choice(3)
    if type == 0:
        magnitude_low = 0.0
        magnitude_high = 1000.0
    elif type == 1:
        magnitude_low = 1000.0
        magnitude_high = 4000.0
    elif type == 2:
        magnitude_low = 4000.0
        magnitude_high = 7000.0

    for k in range(num_kernels):
        apply_random_shear(displacement_field, xory="x", image_width=image_width, image_height=image_height, magnitude_low=magnitude_low, magnitude_high=magnitude_high)
        apply_random_shear(displacement_field, xory="y", image_width=image_width, image_height=image_height, magnitude_low=magnitude_low, magnitude_high=magnitude_high)

    return displacement_field

def apply_displacement_field(image: torch.Tensor, displacement_field: torch.Tensor):
    """
    Apply a displacement field to an image.
    :param image: The image to apply the displacement field to. The image should have shape (..., H, W)
    :param displacement_field: The displacement field to apply. The displacement field should have shape (1, H, W, 2)
    :return: The image with the displacement field applied.
    """
    assert image.shape[-2] == displacement_field.shape[1], "The image and displacement field must have the same height."
    assert image.shape[-1] == displacement_field.shape[2], "The image and displacement field must have the same width."
    img_shape = image.shape

    image = torchvision.transforms.functional.elastic_transform(image.view(-1, 1, img_shape[-2], img_shape[-1]),
                displacement_field, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    return image.view(img_shape)

def apply_random_shear3D(displacement_field, xory, image_depth, image_width, image_height,
                         kernel_depth_span, magnitude_low, magnitude_high):
    x_low = image_width // 3
    x_high = image_width * 2 // 3
    y_low = image_height // 3
    y_high = image_height * 2 // 3

    depth_radius = (image_depth - 1) // 2
    span_radius = (len(kernel_depth_span) - 1) // 2
    d_low = depth_radius - span_radius
    d_high = depth_radius + span_radius

    x = np.random.randint(low=x_low, high=x_high + 1)
    y = np.random.randint(low=y_low, high=y_high + 1)
    d = np.random.randint(low=d_low, high=d_high + 1)
    sigma = np.random.uniform(low=100.0, high=200.0)
    magnitude = np.random.uniform(low=magnitude_low, high=magnitude_high) * np.random.choice([-1, 1])

    kernel_width = image_width
    kernel_height = image_height
    kernel = np.expand_dims(cv2.getGaussianKernel(ksize=kernel_height * 2 + 1, sigma=sigma), axis=-1)\
             * cv2.getGaussianKernel(ksize=kernel_width * 2 + 1, sigma=sigma) * magnitude

    expand_left = min(kernel_width, x)
    expand_right = min(kernel_width + 1, image_width - x)
    expand_top = min(kernel_height, y)
    expand_bottom = min(kernel_height + 1, image_height - y)

    if xory == "x":
        displacement_field[d - span_radius:d + span_radius + 1, y - expand_top:y + expand_bottom, x - expand_left:x + expand_right, 0:1] += \
            (kernel[kernel_height - expand_top:kernel_height + expand_bottom, kernel_width - expand_left:kernel_width + expand_right, :]
                * np.expand_dims(kernel_depth_span, axis=(1, 2, 3)))
    else:
        displacement_field[d - span_radius:d + span_radius + 1, y - expand_top:y + expand_bottom, x - expand_left:x + expand_right, 1:2] += \
            (kernel[kernel_height - expand_top:kernel_height + expand_bottom, kernel_width - expand_left:kernel_width + expand_right, :]
                * np.expand_dims(kernel_depth_span, axis=(1, 2, 3)))

def generate_displacement_field3D(image_width, image_height, image_depth, kernel_depth_span: list[float], num_kernels=7) -> np.ndarray:
    assert len(kernel_depth_span) <= image_depth, "The kernel depth span must be smaller than the image depth."
    assert image_depth % 2 == 1, "The image depth must be odd."
    assert len(kernel_depth_span) % 2 == 1, "The kernel depth span must be odd."

    displacement_field = np.zeros(shape=(image_depth, image_height, image_width, 2), dtype=np.float32)
    kernel_depth_span = np.array(kernel_depth_span, dtype=np.float32)

    type = np.random.choice(3)
    if type == 0:
        magnitude_low = 0.0
        magnitude_high = 1000.0
    elif type == 1:
        magnitude_low = 1000.0
        magnitude_high = 4000.0
    elif type == 2:
        magnitude_low = 4000.0
        magnitude_high = 7000.0

    for k in range(num_kernels):
        apply_random_shear3D(displacement_field, xory="x", image_depth=image_depth, image_width=image_width, image_height=image_height,
                             kernel_depth_span=kernel_depth_span, magnitude_low=magnitude_low, magnitude_high=magnitude_high)
        apply_random_shear3D(displacement_field, xory="y", image_depth=image_depth, image_width=image_width, image_height=image_height,
                             kernel_depth_span=kernel_depth_span, magnitude_low=magnitude_low, magnitude_high=magnitude_high)

    return displacement_field

def apply_displacement_field3D(image: torch.Tensor, displacement_field: torch.Tensor):
    """
    Apply a displacement field to an image.
    :param image: The image to apply the displacement field to. The image should have shape (..., D, H, W)
    :param displacement_field: The displacement field to apply. The displacement field should have shape (D, H, W, 2)
    :return: The image with the displacement field applied.
    """
    assert image.shape[-2] == displacement_field.shape[1], "The image and displacement field must have the same height."
    assert image.shape[-1] == displacement_field.shape[2], "The image and displacement field must have the same width."
    assert image.shape[-3] == displacement_field.shape[0], "The image and displacement field must have the same depth."

    for d in image.shape[-3]:
        slice = image[..., d, :, :]
        slice_shape = slice.shape
        image[..., d, :, :].copy_(torchvision.transforms.functional.elastic_transform(slice.view(-1, 1, slice_shape[-2], slice_shape[-1]),
                displacement_field[d, ...].unsqueeze(0), interpolation=torchvision.transforms.InterpolationMode.NEAREST).view(slice_shape), non_blocking=True)
    return image


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication, QMainWindow, QFrame, QSlider, QLabel, \
        QVBoxLayout, QFileDialog, QPushButton
    from PySide2.QtCore import Qt, QThread, Signal
    from PySide2.QtGui import QStandardItemModel, QStandardItem, QPixmap, QImage
    import h5py
    import matplotlib.pyplot as plt
    import torch
    import torchvision.transforms.functional
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


    class ElasticAugmentationViewer(QMainWindow):

        series_ct: dict[str, list[str]]  # Contains a list of CT scan series for each patient.
        z_positions: np.ndarray  # Contains the z positions of the slices. Shape is (z,).

        fig: plt.Figure
        image_canvas: FigureCanvas

        def __init__(self, ct_image: np.ndarray):
            super().__init__()
            self.ct_image = ct_image
            self.transformed_ct_image = None

            self.setup_ui()
            self.setup_connections()

            self.z_positions = np.arange(ct_image.shape[0])
            self.min_series = 0
            self.max_series = ct_image.shape[0] - 1
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
            # Add a button to regenerate the elastic augmentation
            self.regenerate_button = QPushButton()
            self.regenerate_button.setText("Regenerate Elastic Augmentation")
            self.regenerate_button.setFixedHeight(self.regenerate_button.sizeHint().height())

            # Add the slider and label to the main panel
            self.main_layout = QVBoxLayout()
            self.main_layout.addWidget(self.image_canvas)
            self.main_layout.addWidget(self.slice_number_label)
            self.main_layout.addWidget(self.slice_number_slider)
            self.main_layout.addWidget(self.regenerate_button)

            self.main_widget.setLayout(self.main_layout)

        def setup_connections(self):
            self.slice_number_slider.valueChanged.connect(self.slice_number_updated)
            self.regenerate_button.clicked.connect(self.regenerate_clicked)

        def regenerate_clicked(self):
            del self.transformed_ct_image
            self.transformed_ct_image = None

            dp_field = generate_displacement_field(self.ct_image.shape[2], self.ct_image.shape[1])
            dp_field = torch.from_numpy(dp_field)
            ct_torch = torch.from_numpy(self.ct_image).unsqueeze(1)
            ct_torch = torchvision.transforms.functional.elastic_transform(ct_torch, dp_field,
                                        interpolation=torchvision.transforms.InterpolationMode.NEAREST).squeeze(1)
            self.transformed_ct_image = ct_torch.numpy()

            self.update_image()


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
            slice1 = self.ct_image[slice_number - self.min_series, ...]
            if self.transformed_ct_image is None:
                slice2 = self.ct_image[slice_number - self.min_series, ...]
            else:
                slice2 = self.transformed_ct_image[slice_number - self.min_series, ...]

            self.fig.clear()
            ax_ct1 = self.fig.add_subplot(1, 2, 1)
            ax_ct1.imshow(slice1, cmap="gray")

            ax_ct2 = self.fig.add_subplot(1, 2, 2)
            ax_ct2.imshow(slice2, cmap="gray")

            self.fig.set_size_inches(slice1.shape[1] / 100 * 2, slice2.shape[0] / 100)
            self.image_canvas.draw()

    app = QApplication([])

    # prompt the user to select the hdf file
    ct_3D_image_file, _ = QFileDialog.getOpenFileName(None, "Select the first hdf5 file", "", "HDF Files (*.hdf5)")
    if not ct_3D_image_file:
        exit(0)

    # load the hdf file
    with h5py.File(ct_3D_image_file, "r") as f:
        ct_3D_image = np.array(f["ct_3D_image"])

    window = ElasticAugmentationViewer(ct_3D_image)
    window.show()
    app.exec_()