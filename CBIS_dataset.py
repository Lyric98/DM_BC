import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch

class CBISDataset(torch.utils.data.Dataset):
    def __init__(self, json, transform=None, save=False, save_dir=None):
        assert not save or save_dir, "save_dir must be provided if save is True"
        self.json_path = json
        self.transform = transform

        # Get the list of image paths
        self.image_paths = pd.read_json(json, orient='records')[0].tolist()

        # Save directories
        self.save = save
        self.save_dir = save_dir

        if self.save:
            # Ensure the save directory exists
            os.makedirs(self.save_dir, exist_ok=True)
            self.save_dir_L = os.path.join(self.save_dir, 'full_image_L')
            self.save_dir_RGB = os.path.join(self.save_dir, 'full_image_RGB')
            os.makedirs(self.save_dir_L, exist_ok=True)
            os.makedirs(self.save_dir_RGB, exist_ok=True)

    def process_single_image(self, img_path):
        """Function to process a single image with a shorter folder name."""
        try:
            # Extract only the last segment of the folder name after the last dot
            folder_name = os.path.basename(os.path.dirname(img_path)).split('.')[-1]
            image_name = os.path.basename(img_path).replace('.jpg', '.png')
            
            with Image.open(img_path) as img:
                # Convert the image to RGB
                rgb_img = img.convert('RGB')

                # Create unique filenames with shortened folder name
                save_path_L = os.path.join(self.save_dir_L, f"{folder_name}_{image_name}")
                save_path_RGB = os.path.join(self.save_dir_RGB, f"{folder_name}_{image_name}")

                img.save(save_path_L, 'PNG')
                rgb_img.save(save_path_RGB, 'PNG')

                return f"Processed: {folder_name}/{image_name}"
        except Exception as e:
            return f"Error processing {img_path}: {e}"

    def process_images_multiprocess(self):
        """Method to process all images using multiple CPUs."""
        if not self.save:
            print("Image saving is disabled.")
            return

        with ProcessPoolExecutor() as executor:
            # Submit all tasks to the executor
            futures = [executor.submit(self.process_single_image, img_path) for img_path in self.image_paths]

            # Use tqdm to track the progress as tasks complete
            for future in tqdm(as_completed(futures), total=len(self.image_paths), desc="Processing Images"):
                result = future.result()
                print(result)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        
        if self.transform:
            img = self.transform(img)

        return img


def get_cropped_data(datacsv_path):
    # Load the CSV file
    # df = pd.read_csv('dataset/CBIS-DDSM/csv/dicom_info.csv')
    df = pd.read_csv(datacsv_path)

    # Filter rows where SeriesDescription is "cropped images"
    filtered_df = df[df['SeriesDescription'] == 'cropped images']

    # Extract the image paths
    image_paths = filtered_df['image_path']

    # add dataset/ to the image paths in the beginning
    image_paths = image_paths.apply(lambda x: 'dataset/' + x)

    # Output the image paths
    print(image_paths)

    # Save to a json
    image_paths.to_json('cropped_image_paths.json', orient='records')


def get_full_mammogram_images(datacsv_path):
    # Load the CSV file
    # df = pd.read_csv('dataset/CBIS-DDSM/csv/dicom_info.csv')
    df = pd.read_csv(datacsv_path)

    # Filter rows where SeriesDescription is "cropped images"
    filtered_df = df[df['SeriesDescription'] == 'full mammogram images']

    # Extract the image paths
    image_paths = filtered_df['image_path']

    # add dataset/ to the image paths in the beginning
    image_paths = image_paths.apply(lambda x: 'dataset/' + x)

    # Output the image paths
    print(image_paths)

    # Save to a json
    image_paths.to_json('full_mammogram_image_paths.json', orient='records')


def get_ROI_mask_images(datacsv_path):
    # Load the CSV file
    # df = pd.read_csv('dataset/CBIS-DDSM/csv/dicom_info.csv')
    df = pd.read_csv(datacsv_path)

    # Filter rows where SeriesDescription is "cropped images"
    filtered_df = df[df['SeriesDescription'] == 'ROI mask images']

    # Extract the image paths
    image_paths = filtered_df['image_path']

    # add dataset/ to the image paths in the beginning
    image_paths = image_paths.apply(lambda x: 'dataset/' + x)

    # Output the image paths
    print(image_paths)

    # save to a json file
    image_paths.to_json('ROI_mask_image_paths.json', orient='records')


def read_data(csv_path):
    # Read the text file
    # image_p = pd.read_csv('cropped_image_paths.csv', header=None)
    # image_p = pd.read_csv('full_mammogram_image_paths.csv', header=None)
    # image_p = pd.read_csv('ROI_mask_image_paths.csv', header=None)
    image_p = pd.read_csv(csv_path, header=None)

    # Extract the image paths
    image_paths = image_p.iloc[:, 0]

    exist_images = []
    # if the file exists, print the image paths and save to a list, also print the amount of images
    for p in image_paths:
        # p = p.replace('CBIS-DDSM', 'dataset/CBIS-DDSM')
        if os.path.exists(p):
            print(p)

            exist_images.append(p)
    
    print('Amount of images: ', len(exist_images))



if __name__ == '__main__':
    # csv_path = "dataset/full_mammogram_image_paths.csv"
    json_path = "dataset/full_mammogram_image_paths.json"
    # dataset = CBISDataset(json_path, save=True, save_dir='dataset/full_image_1024_1024')
    dataset = CBISDataset(json_path, save=True, save_dir='dataset/CBIS_full')
    dataset.process_images_multiprocess()


    # datacsv_path = "dataset/CBIS-DDSM/csv/dicom_info.csv"
    # get_cropped_data(datacsv_path)
    # get_full_mammogram_images(datacsv_path)
    # get_ROI_mask_images(datacsv_path)
    # read_data(csv_path)
