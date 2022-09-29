from glob import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class DataGenerator:
    def __init__(self, mask_path: str):
        self.mask_files = glob(mask_path)
        self.image_path = self.load_images_and_masks()
        self.df = pd.DataFrame(data={'image_filenames': self.image_path, 'mask': self.mask_files})
        df_train, self.df_test = train_test_split(self.df, test_size=0.1)
        self.df_train, self.df_val = train_test_split(self.df, test_size=0.2)
        print(f'Df generation and train test split - done!\n'
              f'Train size: {self.df_train.shape}\n'
              f'Test size: {self.df_test.shape}\n'
              f'Validation size: {self.df_val.shape}'
              )

    def load_images_and_masks(self):
        image_filenames_train = []
        for i in self.mask_files:
            # generate file list with existing masks
            image_filenames_train.append(i.replace('_mask', ''))

        return image_filenames_train

    def plot_examples(self, rows: int, columns: int):
        fig = plt.figure(figsize=(12,12))
        # build grid
        for i in range(1, rows*columns + 1):
            fig.add_subplot(rows, columns, i)
            img_path = self.image_path[i]
            example_mask = self.mask_files[i]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(example_mask)
            plt.imshow(image)
            plt.imshow(mask, alpha=0.4)
        plt.show()


