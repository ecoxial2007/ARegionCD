import json
import os
from PIL import Image
from torch.utils.data import Dataset

class RegionVQADataset(Dataset):
    def __init__(self, annotation_file='', vis_root='', transform=None):
        """
        Initialize the dataset.

        Parameters:
            annotation_file (str): Path to the annotation file containing image IDs and captions.
            vis_root (str): Root directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a PIL image.
        """
        # Load the annotation data from a JSON file.
        with open(annotation_file, 'r') as file:
            self.annotation = json.load(file)

        self.vis_root = vis_root
        self.img_ids = {ann['image_ids'][0]: idx for idx, ann in enumerate(self.annotation)}
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.annotation)

    def __getitem__(self, index):
        """
        Retrieve a sample from the dataset at the specified index.
        """
        ann = self.annotation[index]
        image_id = ann['image_ids'][0]

        if 'MIMIC' in self.vis_root:
            # --img_root ../../00_Data/MIMIC-CXR/images_512 \
            image_id = image_id.split('_')[0] + '.png'
            image_path = os.path.join(self.vis_root, image_id)
            # no image_mask_path

            image = Image.open(image_path).convert('RGB')
            image_mask = Image.open(image_path).convert('L') # 随便给的，防止报错 mimic用不到

        elif 'OBScan' in self.vis_root:
            image_id0 = image_id + '.jpg'
            # --img_root '../../00_Data/MedRegion/OBScan-Region' \ (/images or masks )
            image_path = os.path.join(self.vis_root, "images", image_id0)
            mask_filename = image_id + "_mask.png"
            image_mask_path = os.path.join(self.vis_root, "masks", mask_filename)

            image = Image.open(image_path).convert('RGB')
            image_mask = Image.open(image_mask_path).convert('L')
        
        elif 'SLAKE' in self.vis_root:
            # --img_root ../../00_Data/MedRegion/SLAKE-Region \ (/images or new_masks )
            image_path = os.path.join(self.vis_root, "images", image_id)
            base_name = os.path.splitext(image_id)[0]
            mask_filename = base_name + "_mask.png"
            image_mask_path = os.path.join(self.vis_root, "new_masks", mask_filename)

            image = Image.open(image_path).convert('RGB')
            image_mask = Image.open(image_mask_path).convert('L')

        else:
            pass
        
        
        if self.transform:
            image = self.transform(image)

        question = ann['question']
        answer = ann['answer']
        
        highlight = list(ann['template_arguments']['object'].values())
        attribute = ann['template_arguments']['attribute']  # slake-->>merge whole match
        try:
            mask_value = ann["template_arguments"]["mask_value"]["0"]
        except KeyError:
            mask_value = 0  # mimic no mask_value
        
        
        # Safely get the bounding box coordinates.
        try:
            bbox = ann['template_arguments']['bbox']['0']
            bbox = tuple(map(float, bbox))
        except KeyError:
            # If 'bbox' key doesn't exist, use a default empty bbox.
            bbox = (0.0, 0.0, 0.0, 0.0) 

        # Return a dictionary containing all the prepared data.
        return {
            "image": image,
            "image_mask":image_mask,
            "question": question,
            "answer": answer,
            "image_id": self.img_ids[ann["image_ids"][0]], # The unique integer index for the image
            "image_path": image_path,
            "highlights": highlight,
            "attribute": attribute,
            "mask_value":mask_value,
            "bbox": bbox
        }


class RegionVQADataset_ABL(Dataset):
    def __init__(self, annotation_file='', vis_root='', transform=None, preprocess=None):
        """
        Initialize the dataset.

        Parameters:
            annotation_file (str): Path to the annotation file containing image IDs and captions.
            vis_root (str): Root directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a PIL image.
        """
        # Load the annotation data from a JSON file.
        with open(annotation_file, 'r') as file:
            self.annotation = json.load(file)

        self.vis_root = vis_root
        self.img_ids = {ann['image_ids'][0]: idx for idx, ann in enumerate(self.annotation)}
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.annotation)

    def __getitem__(self, index):
        """
        Retrieve a sample from the dataset at the specified index.
        """
        ann = self.annotation[index]
        image_id = ann['image_ids'][0]

        if 'MIMIC' in self.vis_root:
            # --img_root ../../00_Data/MIMIC-CXR/images_512 \
            image_id = image_id.split('_')[0] + '.png'
            image_path = os.path.join(self.vis_root, image_id)
            # no image_mask_path

            image = Image.open(image_path).convert('RGB')
            image_mask = 0
            # image_mask = Image.open(image_path).convert('L')  # 随便给的，防止报错 mimic用不到

        elif 'OBScan' in self.vis_root:
            image_id0 = image_id + '.jpg'
            # --img_root '../../00_Data/MedRegion/OBScan-Region' \ (/images or masks )
            image_path = os.path.join(self.vis_root, "images", image_id0)
            mask_filename = image_id + "_mask.png"
            image_mask_path = os.path.join(self.vis_root, "masks", mask_filename)

            image = Image.open(image_path).convert('RGB')
            # image_mask = Image.open(image_mask_path).convert('L')
            image_mask = 0
        elif 'SLAKE' in self.vis_root:
            # --img_root ../../00_Data/MedRegion/SLAKE-Region \ (/images or new_masks )
            image_path = os.path.join(self.vis_root, "images", image_id)
            base_name = os.path.splitext(image_id)[0]
            mask_filename = base_name + "_mask.png"
            image_mask_path = os.path.join(self.vis_root, "new_masks", mask_filename)

            image = Image.open(image_path).convert('RGB')
            image_mask = 0
            # image_mask = Image.open(image_mask_path).convert('L')
        else:
            print('Not supported dataset type')
            quit()

        if self.transform:
            image = self.transform(image)


        question = ann['question']
        answer = ann['answer']
        highlight = list(ann['template_arguments']['object'].values())
        attribute = ann['template_arguments']['attribute']  # slake-->>merge whole match

        try:
            mask_value = ann["template_arguments"]["mask_value"]["0"]
        except KeyError:
            mask_value = 0  # mimic no mask_value

        # Safely get the bounding box coordinates.
        try:
            bbox = ann['template_arguments']['bbox']['0']
            bbox = tuple(map(float, bbox))
        except KeyError:

            bbox = (0.0, 0.0, 0.0, 0.0)


        # Return a dictionary containing all the prepared data.
        return {
            "image": image,
            "image_mask": image_mask,
            "question": question,
            "answer": answer,
            "image_id": self.img_ids[ann["image_ids"][0]],  # The unique integer index for the image
            "image_path": image_path,
            "highlights": highlight,
            "attribute": attribute,
            "mask_value": mask_value,
            "bbox": bbox
        }