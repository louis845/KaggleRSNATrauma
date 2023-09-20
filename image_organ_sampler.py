import h5py
import os
import numpy as np
import torch
import torchvision.transforms.functional
import pandas as pd

import config
import image_ROI_sampler
import image_sampler_augmentations
import manager_stage1_results

def load_series_image_and_organloc(patient_id: str, series_id: str, slice_indices,
                      organ_id: int, target_w: int, target_h: int,
                      segmentation_dataset_folder: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the image of a series from the hdf5 file
    """

    ## Computes the required location of the image to locate the organ
    with h5py.File(os.path.join(segmentation_dataset_folder, series_id + ".hdf5"), "r") as f:
        organ_location = f["organ_location"][organ_id, ...] > 0 # shape (H, W)
    heights = np.any(organ_location, axis=-1)
    widths = np.any(organ_location, axis=-2)
    heights, widths = np.argwhere(heights), np.argwhere(widths)
    # Compute the organ bounds
    minH, maxH = heights.min(), heights.max() + 1
    minW, maxW = widths.min(), widths.max() + 1
    midH, midW = (minH + maxH) / 2.0, (minW + maxW) / 2.0
    # Expand to required bounds
    required_minH = int(midH - target_h / 2.0)
    required_minW = int(midW - target_w / 2.0)
    required_maxH = required_minH + target_h
    required_maxW = required_minW + target_w

    ## Load the image
    collapsed_nearest_indices, repeats = image_ROI_sampler.consecutive_repeats(slice_indices)
    with h5py.File(os.path.join("data_hdf5_cropped", patient_id, series_id, "ct_3D_image.hdf5"), "r") as f:
        ct_3D_image = f["ct_3D_image"] # shape (D, H, W)
        # If the required bounds are outside the image, we pad the image. First compute the pad
        pad_left = max(0, -required_minW)
        pad_right = max(0, required_maxW - ct_3D_image.shape[2])
        pad_top = max(0, -required_minH)
        pad_bottom = max(0, required_maxH - ct_3D_image.shape[1])
        required_minH = max(0, required_minH)
        required_minW = max(0, required_minW)
        required_maxH = min(ct_3D_image.shape[1], required_maxH)
        required_maxW = min(ct_3D_image.shape[2], required_maxW)
        # Load and pad the image
        image = ct_3D_image[collapsed_nearest_indices, required_minH:required_maxH, required_minW:required_maxW]
    image = np.pad(image, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
    image = np.repeat(image, repeats, axis=0) # if there is collapsing, repeat it

    ## Do the same to the organ location
    organ_location = organ_location[required_minH:required_maxH, required_minW:required_maxW]
    organ_location = np.pad(organ_location, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
    return image, organ_location
    
def load_series_image_and_organloc_from_minmax(patient_id: str, series_id: str,
                      organ_id: int, organ_sampling_depth: int,
                      min_slice: int, max_slice: int,
                      target_w: int, target_h: int,
                      segmentation_dataset_folder: str,
                      elastic_augmentation: bool):
    # Generate the slice indices (for the depth) to sample from
    z_poses = np.load(os.path.join("data_hdf5_cropped", str(patient_id), str(series_id), "z_positions.npy"))
    is_flipped = np.mean(np.diff(z_poses)) > 0.0
    z_min, z_max = z_poses[min_slice], z_poses[max_slice]
    if elastic_augmentation:  # expand a bit randomly if elastic augmentation is used
        z_length = z_max - z_min
        if min_slice != 0:
            z_min -= np.random.uniform(low=0.0, high=z_length * 0.05)
        if max_slice != len(z_poses) - 1:
            z_max += np.random.uniform(low=0.0, high=z_length * 0.05)
    expected_zposes = np.linspace(z_min, z_max, organ_sampling_depth)
    if is_flipped:
        nearest_slice_indices = image_ROI_sampler.find_closest(z_poses, expected_zposes)
    else:
        nearest_slice_indices = (z_poses.shape[0] - 1 - image_ROI_sampler.find_closest(
            z_poses[::-1], expected_zposes))[::-1]

    # Apply more depthwise elastic augmentation
    if elastic_augmentation:
        dist = (np.min(np.diff(nearest_slice_indices)) // 4) - 1
        if dist > 1:
            nearest_slice_indices = nearest_slice_indices + np.random.randint(-dist, dist + 1,
                                                                              size=organ_sampling_depth)
    nearest_slice_indices = np.clip(nearest_slice_indices, 0, len(z_poses) - 1)

    # Flip the nearest slice indices if not flipped
    if not is_flipped:
        nearest_slice_indices = nearest_slice_indices[::-1]

    # Load the image
    image, organ_location = load_series_image_and_organloc(str(patient_id), str(series_id),
                                                           nearest_slice_indices,
                                                           organ_id, target_w, target_h,
                                                           segmentation_dataset_folder)
    if is_flipped:
        image = image[::-1, ...].copy()
    return image, organ_location

def load_image(patient_ids: list,
               series_ids: list,
               organ_id: int, organ_height: int, organ_width: int, # organ id, expected organ height and width
               stage1_information: manager_stage1_results.Stage1ResultsManager,
               organ_sampling_depth = 9,
               translate_rotate_augmentation=False,
               elastic_augmentation=False) -> torch.Tensor:
    assert len(patient_ids) == len(series_ids), "patient_ids and series_ids must have the same length"
    assert organ_width % 32 == 0, "Organ width must be divisible by 32"
    assert organ_height % 32 == 0, "Organ height must be divisible by 32"
    batch_size = len(patient_ids)


    ## Compute the required height and width
    max_angle = 15 * np.pi / 180
    cur_angle = np.arctan2(organ_height, organ_width)
    diag = np.hypot(organ_height + 64.0, organ_width + 64.0)
    req_rot_w = int(np.ceil(diag * max(np.sin(cur_angle + max_angle), np.sin(cur_angle - max_angle))))
    req_rot_h = int(np.ceil(diag * max(np.cos(cur_angle + max_angle), np.cos(cur_angle - max_angle))))

    ## Load the images
    image_batch = torch.zeros((batch_size, 1, organ_sampling_depth, req_rot_h, req_rot_w), dtype=torch.float32,
                              device=config.device)
    organ_loc_batch = torch.zeros((batch_size, 1, req_rot_h, req_rot_w), dtype=torch.float32,
                                device=config.device)
    for k in range(batch_size):
        organ_slice_min, organ_slice_max = stage1_information.get_organ_slicelocs(int(series_ids[k]), organ_id)
        image, organ_location = load_series_image_and_organloc_from_minmax(str(patient_ids[k]), str(series_ids[k]),
                                                                           organ_id, organ_sampling_depth,
                                                                           organ_slice_min, organ_slice_max,
                                                                           req_rot_w, req_rot_h,
                                                                           stage1_information.segmentation_dataset_folder,
                                                                           elastic_augmentation)
        image_batch[k, 0, ...].copy_(torch.from_numpy(image), non_blocking=True)
        organ_loc_batch[k, 0, ...].copy_(torch.from_numpy(organ_location), non_blocking=True)

    with torch.no_grad():
        ## Apply elastic deformation to height width
        if elastic_augmentation:
            # 3d elastic deformation (varying 2d elastic deformation over depth), and also varying over slices
            displacement_field = torch.stack([image_sampler_augmentations.generate_displacement_field3D(req_rot_w, req_rot_h, organ_sampling_depth, # (batch_size, organ_sampling_depth, H, W, 2)
                                                                                            kernel_depth_span=[0.3, 0.7, 1, 0.7, 0.3], device=config.device) for k in range(batch_size)], dim=0)
            assert displacement_field.shape == (batch_size, organ_sampling_depth, req_rot_h, req_rot_w, 2)
            image_batch = image_sampler_augmentations.apply_displacement_field3D_simple(image_batch.reshape(batch_size * organ_sampling_depth, 1, req_rot_h, req_rot_w),
                                                                                  displacement_field.view(batch_size * organ_sampling_depth, req_rot_h, req_rot_w, 2))\
                            .view(batch_size, 1, organ_sampling_depth, req_rot_h, req_rot_w)
            organ_loc_batch = torch.any(image_sampler_augmentations.apply_displacement_field3D_simple(organ_loc_batch.expand(batch_size, organ_sampling_depth, req_rot_h, req_rot_w)
                                                                                            .reshape(batch_size * organ_sampling_depth, 1, req_rot_h, req_rot_w),
                                                                                  displacement_field.view(batch_size * organ_sampling_depth, req_rot_h, req_rot_w, 2))
                                            .view(batch_size, 1, organ_sampling_depth, req_rot_h, req_rot_w) > 0.5, dim=2).float()

        ## Apply rotation augmentation to height width
        if translate_rotate_augmentation:
            # Generate the rotation angles
            rotation_angles = np.random.uniform(low=-max_angle, high=max_angle, size=batch_size)
            rotation_angles = rotation_angles * 180.0 / np.pi # convert to degrees

            # Rotate the image
            image_batch = image_sampler_augmentations.rotate(image_batch, list(rotation_angles))
            organ_loc_batch = image_sampler_augmentations.rotate(organ_loc_batch.view(batch_size, 1, 1, req_rot_h, req_rot_w), list(rotation_angles))\
                                .view(batch_size, req_rot_h, req_rot_w) > 0.5

        ## Crop the image to the desired size, and apply translation augmentation if necessary
        final_image_batch = torch.zeros((batch_size, 1, organ_sampling_depth, organ_height, organ_width), dtype=torch.float32,
                                                device=config.device)
        for k in range(batch_size):
            # compute organ bounds
            heights = torch.any(organ_loc_batch[k, ...], dim=-1)
            widths = torch.any(organ_loc_batch[k, ...], dim=-2)
            heights = np.argwhere(heights.cpu().numpy())
            widths = np.argwhere(widths.cpu().numpy())
            if len(heights) == 0 or len(widths) == 0:
                mid_x = req_rot_h // 2
                mid_y = req_rot_w // 2
            else:
                mid_x = (np.min(widths) + np.max(widths)) // 2
                mid_y = (np.min(heights) + np.max(heights)) // 2
            x_min, x_max = mid_x - organ_width // 2, mid_x + organ_width // 2
            y_min, y_max = mid_y - organ_height // 2, mid_y + organ_height // 2

            # correct for out of bounds
            if x_min < 0:
                x_max -= x_min
                x_min = 0
            elif x_max > req_rot_w:
                x_min -= (x_max - req_rot_w)
                x_max = req_rot_w
            if y_min < 0:
                y_max -= y_min
                y_min = 0
            elif y_max > req_rot_h:
                y_min -= (y_max - req_rot_h)
                y_max = req_rot_h

            # apply translation augmentation
            if translate_rotate_augmentation:
                left_available, right_available = x_min, req_rot_w - x_max
                top_available, bottom_available = y_min, req_rot_h - y_max
                x_translation = np.random.randint(-min(left_available, 48), min(right_available, 48) + 1)
                y_translation = np.random.randint(-min(top_available, 48), min(bottom_available, 48) + 1)
                x_min, x_max = x_min + x_translation, x_max + x_translation
                y_min, y_max = y_min + y_translation, y_max + y_translation

            # crop the image
            final_image_batch[k, 0, ...].copy_(image_batch[k, 0, :, y_min:y_max, x_min:x_max], non_blocking=True)

    return final_image_batch


if __name__ == "__main__":
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

    from PySide2.QtWidgets import QApplication, QMainWindow, QFrame, QSlider, QLabel, \
        QVBoxLayout, QFileDialog, QPushButton, QComboBox
    from PySide2.QtCore import Qt, QThread, Signal
    from PySide2.QtGui import QStandardItemModel, QStandardItem, QPixmap, QImage
    import h5py
    import matplotlib.pyplot as plt
    import torch
    import torchvision.transforms.functional
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    import manager_folds

    class OrganSamplerViewer(QMainWindow):

        ct_batch: np.ndarray

        fig: plt.Figure
        image_canvas: FigureCanvas

        def __init__(self, ct_batch: np.ndarray):
            super().__init__()
            assert ct_batch.shape[2] % 2 == 0
            self.ct_batch = ct_batch
            self.N, self.C, self.D, self.H, self.W = self.ct_batch.shape

            self.setup_ui()
            self.setup_connections()

            self.slice_number_slider.setRange(0, (self.D // 2) - 1)

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
            # Create a dropdown menu to control the batch index
            self.batch_index_dropdown = QComboBox()
            self.batch_index_dropdown.setFixedHeight(self.batch_index_dropdown.sizeHint().height())
            self.batch_index_dropdown.setFixedWidth(self.batch_index_dropdown.sizeHint().width() * 2)
            self.batch_index_dropdown.addItems([str(i) for i in range(self.N)])

            # Add the slider and label to the main panel
            self.main_layout = QVBoxLayout()
            self.main_layout.addWidget(self.image_canvas)
            self.main_layout.addWidget(self.slice_number_label)
            self.main_layout.addWidget(self.slice_number_slider)
            self.main_layout.addWidget(self.batch_index_dropdown)

            self.main_widget.setLayout(self.main_layout)

        def setup_connections(self):
            self.slice_number_slider.valueChanged.connect(self.slice_number_updated)
            self.batch_index_dropdown.currentIndexChanged.connect(self.batch_index_updated)

        def slice_number_updated(self, value: int):
            self.slice_number_label.setText("Slice Number: {}".format(value))
            self.update_image()

        def batch_index_updated(self, value: int):
            self.update_image()

        def update_image(self):
            # Get the slice number
            slice_number = self.slice_number_slider.value()
            # Get the batch index
            batch_index = self.batch_index_dropdown.currentIndex()
            # Get the image
            slice1 = self.ct_batch[batch_index, 0, slice_number, ...]
            slice2 = self.ct_batch[batch_index, 0, slice_number + (self.ct_batch.shape[2] // 2), ...]

            self.fig.clear()
            ax_ct1 = self.fig.add_subplot(1, 2, 1)
            ax_ct1.imshow(slice1, cmap="gray")
            ax_ct2 = self.fig.add_subplot(1, 2, 2)
            ax_ct2.imshow(slice2, cmap="gray")

            self.fig.set_size_inches(slice1.shape[1] / 100 * 2, slice1.shape[0] / 100)
            self.image_canvas.draw()


    app = QApplication([])
    dataset_path = QFileDialog.getOpenFileName(None, "Select Dataset", "folds", "JSON (*.json)")[0]
    if dataset_path == "":
        exit(0)

    sizes = [  # sizes in (H, W)
        (352, 448),  # 0: liver
        (320, 416),  # 1: spleen
        (224, 352)  # 2: kidney
    ]

    organ_id = 2
    organ_size = sizes[organ_id]
    subdata = os.path.basename(dataset_path)[:-5]

    # Load the dataset
    dataset = manager_folds.load_dataset(subdata)
    dataset = dataset
    stage1_results = manager_stage1_results.Stage1ResultsManager(subdata)

    dataset = stage1_results.restrict_patient_ids_to_good_series(dataset)
    dataset = stage1_results.restrict_patient_ids_to_organs(dataset, organ_id)

    # generate series
    patient_ids = dataset[:10]
    series1, series2 = stage1_results.get_dual_series(patient_ids, organ_id=organ_id)

    # load image
    volume_depth = 9
    disable_rotpos_augmentation = False
    disable_elastic_augmentation = False
    image_batch1 = load_image(patient_ids,
                                  series1,
                                  organ_id, organ_size[0], organ_size[1],
                                  stage1_results,
                                  volume_depth,
                                  translate_rotate_augmentation=not disable_rotpos_augmentation,
                                  elastic_augmentation=not disable_elastic_augmentation)
    image_batch2 = load_image(patient_ids,
                                  series2,
                                  organ_id, organ_size[0], organ_size[1],
                                  stage1_results,
                                  volume_depth,
                                  translate_rotate_augmentation=not disable_rotpos_augmentation,
                                  elastic_augmentation=not disable_elastic_augmentation)

    image_batch = torch.cat([image_batch1, image_batch2], dim=2)

    window = OrganSamplerViewer(image_batch.cpu().numpy())
    window.show()
    app.exec_()