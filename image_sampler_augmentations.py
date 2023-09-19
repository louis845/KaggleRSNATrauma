import math

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

    image = torchvision.transforms.functional.elastic_transform(image.reshape(-1, 1, img_shape[-2], img_shape[-1]),
                displacement_field, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    return image.view(img_shape)

def apply_random_shear3D(displacement_field, xory, image_depth, image_width, image_height,
                         kernel_depth_span, magnitude_low, magnitude_high, device):
    x_low = image_width // 3
    x_high = image_width * 2 // 3
    y_low = image_height // 3
    y_high = image_height * 2 // 3

    depth_radius = (image_depth - 1) // 2
    span_radius = (kernel_depth_span.shape[0] - 1) // 2
    d_low = depth_radius - span_radius
    d_high = depth_radius + span_radius

    x = np.random.randint(low=x_low, high=x_high + 1)
    y = np.random.randint(low=y_low, high=y_high + 1)
    d = np.random.randint(low=d_low, high=d_high + 1)
    sigma = np.random.uniform(low=100.0, high=200.0)
    magnitude = np.random.uniform(low=magnitude_low, high=magnitude_high) * np.random.choice([-1, 1])

    kernel_width = image_width
    kernel_height = image_height
    kernel = torch.tensor(cv2.getGaussianKernel(ksize=kernel_height * 2 + 1, sigma=sigma), dtype=torch.float32, device=device).unsqueeze(-1) \
                * torch.tensor(cv2.getGaussianKernel(ksize=kernel_width * 2 + 1, sigma=sigma), dtype=torch.float32, device=device) * magnitude

    expand_left = min(kernel_width, x)
    expand_right = min(kernel_width + 1, image_width - x)
    expand_top = min(kernel_height, y)
    expand_bottom = min(kernel_height + 1, image_height - y)

    if xory == "x":
        displacement_field[d - span_radius:d + span_radius + 1, y - expand_top:y + expand_bottom, x - expand_left:x + expand_right, 0:1] += \
            (kernel[kernel_height - expand_top:kernel_height + expand_bottom, kernel_width - expand_left:kernel_width + expand_right, :]
                * kernel_depth_span.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
    else:
        displacement_field[d - span_radius:d + span_radius + 1, y - expand_top:y + expand_bottom, x - expand_left:x + expand_right, 1:2] += \
            (kernel[kernel_height - expand_top:kernel_height + expand_bottom, kernel_width - expand_left:kernel_width + expand_right, :]
                * kernel_depth_span.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

def generate_displacement_field3D(image_width, image_height, image_depth, kernel_depth_span: list[float], num_kernels=7,
                                  device="cpu") -> torch.Tensor:
    assert len(kernel_depth_span) <= image_depth, "The kernel depth span must be smaller than the image depth."
    assert image_depth % 2 == 1, "The image depth must be odd."
    assert len(kernel_depth_span) % 2 == 1, "The kernel depth span must be odd."

    displacement_field = torch.zeros(size=(image_depth, image_height, image_width, 2), dtype=torch.float32, device=device)
    kernel_depth_span = torch.tensor(kernel_depth_span, dtype=torch.float32, device=device)

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
                             kernel_depth_span=kernel_depth_span, magnitude_low=magnitude_low, magnitude_high=magnitude_high, device=device)
        apply_random_shear3D(displacement_field, xory="y", image_depth=image_depth, image_width=image_width, image_height=image_height,
                             kernel_depth_span=kernel_depth_span, magnitude_low=magnitude_low, magnitude_high=magnitude_high, device=device)

    return displacement_field


def apply_displacement_field3D(image: torch.Tensor, displacement_field: torch.Tensor):
    """Applies a depth dependent 2D displacement field to a 3D image.
    The image should be of shape (..., D, H, W), and the displacement field (D, H, W, 2)"""
    assert image.shape[-3] == displacement_field.shape[0], "The image depth must match the displacement field depth."
    assert image.shape[-2] == displacement_field.shape[1], "The image height must match the displacement field height."
    assert image.shape[-1] == displacement_field.shape[2], "The image width must match the displacement field width."

    shape = image.shape
    image = image.reshape(-1, shape[-3], shape[-2], shape[-1])

    # code from torchvision transforms elastic_transform
    size = list(image.shape[-2:])
    hw_space = [torch.linspace((-s + 1) / s, (s - 1) / s, s, device=image.device) for s in size]
    grid_y, grid_x = torch.meshgrid(hw_space, indexing="ij")
    grid = torch.stack([grid_x, grid_y], -1).unsqueeze(0)
    grid = grid + displacement_field

    image = torch.nn.functional.grid_sample(image.permute(1, 0, 2, 3), grid, mode="nearest", padding_mode="zeros",
                                            align_corners=False).permute(1, 0, 2, 3)

    return image.view(shape)

def apply_displacement_field3D_simple(image: torch.Tensor, displacement_field: torch.Tensor):
    """
    Same as above, but more restrictions on the image. The image should be of shape (D, C, H, W), and the displacement field (D, H, W, 2).
    """
    assert image.shape[0] == displacement_field.shape[0], "The image depth must match the displacement field depth."
    assert image.shape[-2] == displacement_field.shape[1], "The image height must match the displacement field height."
    assert image.shape[-1] == displacement_field.shape[2], "The image width must match the displacement field width."

    # code from torchvision transforms elastic_transform
    size = list(image.shape[-2:])
    hw_space = [torch.linspace((-s + 1) / s, (s - 1) / s, s, device=image.device) for s in size]
    grid_y, grid_x = torch.meshgrid(hw_space, indexing="ij")
    grid = torch.stack([grid_x, grid_y], -1).unsqueeze(0)
    grid = grid + displacement_field

    image = torch.nn.functional.grid_sample(image, grid, mode="nearest", padding_mode="zeros",
                                            align_corners=False)
    return image

def _get_inverse_affine_matrix(angle: float, scale: float, inverted: bool = True) -> list[float]:
    """
    Copied from PyTorch source code. https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional.py
    """
    center, translate, shear = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0] # don't need these
    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    if inverted:
        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = [d, -b, 0.0, -c, a, 0.0]
        matrix = [x / scale for x in matrix]
        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx
        matrix[5] += cy
    else:
        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [x * scale for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    return matrix

def _gen_affine_grid(theta: torch.Tensor, w: int, h: int, ow: int, oh: int) -> torch.Tensor:
    # Copied from https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py
    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
    x_grid = torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * w, 0.5 * h], dtype=theta.dtype, device=theta.device)
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    return output_grid.view(1, oh, ow, 2)

def rotate(img: torch.Tensor, angles: list[float]):
    """
    Rotates the 3D image volumes by the given angles, where the angles are given per image in the batch.
    """
    w, h = img.shape[-1], img.shape[-2]
    batch_size = img.shape[0]
    assert len(angles) == batch_size, "The number of angles must match the batch size."
    assert len(img.shape) == 5, "The input must be a 5D tensor, (N, C, D, H, W)"

    angle_matrices = [_get_inverse_affine_matrix(-angle, 1.0) for angle in angles]
    transformation_grids = torch.tensor()
    for k in range(len(angle_matrices)):
        mat = angle_matrices[k]
        mat_torch = torch.tensor(mat, dtype=torch.float32, device=img.device).reshape(1, 2, 3)
        grid = _gen_affine_grid(mat_torch, w=w, h=h, ow=w, oh=h)
        transformation_grids.append(grid)



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