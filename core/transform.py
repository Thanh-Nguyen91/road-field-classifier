import albumentations as A

###================================================================================

def valid_transform(input_size=224):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=input_size, p=1),
            A.PadIfNeeded(
                min_height=input_size,
                min_width=input_size,
                border_mode=0,
                value=[0, 0, 0],
                p=1,
            ),
            A.Normalize(),
        ],
    )

###================================================================================

def train_transform(input_size=224, reg_factor=0.5):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=input_size, p=1),
            A.PadIfNeeded(
                min_height=input_size,
                min_width=input_size,
                border_mode=0,
                value=[0, 0, 0],
                p=1,
            ),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=60, p=reg_factor
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.4, contrast_limit=0.4, p=reg_factor
            ),
            A.HueSaturationValue(p=reg_factor),
            A.CoarseDropout(
                min_holes=1,
                max_holes=30,
                min_height=5,
                max_height=20,
                min_width=5,
                max_width=20,
                p=reg_factor,
            ),
            A.Normalize(),
        ],
    )


###================================================================================
