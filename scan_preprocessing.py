import pydicom
import numpy as np
import scan_preprocessing_use_sdl

def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype
        pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift

    intercept = float(dcm.RescaleIntercept)
    slope = float(dcm.RescaleSlope)
    center = int(dcm.WindowCenter)
    width = int(dcm.WindowWidth)
    low = center - width / 2
    high = center + width / 2

    pixel_array = (pixel_array * slope) + intercept
    pixel_array = np.clip(pixel_array, low, high)

    return pixel_array

def to_float_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    image = standardize_pixel_array(dcm).astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        image = 1 - image

    return image

if scan_preprocessing_use_sdl.USE_SDL:
    import dicomsdl

    def standardize_pixel_array_dsdl(dcm: dicomsdl._dicomsdl.DataSet) -> np.ndarray:
        pixel_array = dcm.pixelData(storedvalue=True)
        if dcm.PixelRepresentation == 1:
            bit_shift = dcm.BitsAllocated - dcm.BitsStored
            dtype = pixel_array.dtype
            pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift

        intercept = float(dcm.RescaleIntercept)
        slope = float(dcm.RescaleSlope)
        center = int(dcm.WindowCenter)
        width = int(dcm.WindowWidth)
        low = center - width / 2
        high = center + width / 2

        pixel_array = (pixel_array * slope) + intercept
        pixel_array = np.clip(pixel_array, low, high)

        return pixel_array

    def to_float_array_dsdl(dcm: dicomsdl._dicomsdl.DataSet) -> np.ndarray:
        image = standardize_pixel_array_dsdl(dcm).astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)

        if dcm.PhotometricInterpretation == "MONOCHROME1":
            image = 1 - image

        return image
