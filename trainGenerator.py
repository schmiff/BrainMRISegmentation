from keras.preprocessing.image import ImageDataGenerator


# Function taken from https://github.com/zhixuhao/unet/blob/master/data.py
def trainGenerator(data_frame,
                   batch_size,
                   augmentation_dict,
                   image_color_mode="rgb",
                   mask_color_mode="grayscale",
                   image_save_prefix="image",
                   mask_save_prefix="mask",
                   save_to_dir=None,
                   target_size=(256, 256),
                   seed=1):
    """
        generate image and mask at the same time use the same seed for
        image_datagen and mask_datagen to ensure the transformation for image
        and mask is the same if you want to visualize the results of generator,
        set save_to_dir = "your path"
        """
    image_datagen = ImageDataGenerator(**augmentation_dict)
    mask_datagen = ImageDataGenerator(**augmentation_dict)

    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col='image_filenames',
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
    )

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col='mask',
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
    )

    train_gen = zip(image_generator, mask_generator)

    # Final return Tuple after image Normalization and Diagnostics
    for (img, mask) in train_gen:
        img, mask = normalize_and_diagnose(img, mask)
        yield img, mask


def normalize_and_diagnose(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return img, mask
