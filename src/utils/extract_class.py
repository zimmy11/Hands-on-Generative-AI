import os
import shutil
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
# Path where your 'annotations' and 'train2017' folders are located
DATA_DIR = '../../data'
DATA_TYPE = 'val2017' # Change to 'val2017' if you want to extract validation data
ANN_FILE = os.path.join(DATA_DIR, 'annotations', f'instances_{DATA_TYPE}.json')
SRC_IMG_DIR = os.path.join(DATA_DIR, DATA_TYPE)

# The class you want to extract (e.g., 'dog', 'cat', 'car', 'person')
TARGET_CLASS = 'dog'

# Destination folder
DEST_DIR = f'../../data/val_{TARGET_CLASS}_cropped'

# If True: Crops the object using the Bounding Box (Recommended for unconditional training)
# If False: Copies the entire original image (Risky: object might be small or off-center)
CROP_TO_BBOX = True 

# Optional padding around the object (in pixels) to give it some context
PADDING = 10 


# ----------------------

def main():
    # 1. Verify Annotation File Exists
    if not os.path.exists(ANN_FILE):
        print(f"Error: Annotation file not found: {ANN_FILE}")
        print("Please download 'annotations_trainval2017.zip' and extract it.")
        return

    # 2. Initialize COCO API
    print(f"Loading COCO annotations from {ANN_FILE}...")
    coco = COCO(ANN_FILE)

    # 3. Get Category ID for the target class
    catIds = coco.getCatIds(catNms=[TARGET_CLASS])
    if not catIds:
        print(f"Error: Class '{TARGET_CLASS}' not found in COCO categories.")
        return

    # 4. Get all Image IDs that contain this category
    imgIds = coco.getImgIds(catIds=catIds)
    print(f"Found {len(imgIds)} images containing '{TARGET_CLASS}'.")

    # 5. Create Destination Directory
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    print(f"Extracting images to: {DEST_DIR} ...")
    
    count = 0
    # Loop through the images
    for img_data in tqdm(coco.loadImgs(imgIds)):
        src_path = os.path.join(SRC_IMG_DIR, img_data['file_name'])
        
        # Check if image file actually exists locally
        if not os.path.exists(src_path):
            # Useful if you only have a partial download of COCO images
            continue

        if CROP_TO_BBOX:
            # Load specific annotations (bboxes) for this image and this class
            annIds = coco.getAnnIds(imgIds=img_data['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)

            try:
                original_image = Image.open(src_path).convert('RGB')
            except Exception as e:
                print(f"Error opening {src_path}: {e}")
                continue
            
            # An image might contain multiple dogs. We save them as separate files.
            for i, ann in enumerate(anns):
                # Bounding Box format: [x, y, width, height]
                x, y, w, h = ann['bbox']
                
                # Add padding ensuring we don't go out of image boundaries
                x1 = max(0, int(x) - PADDING)
                y1 = max(0, int(y) - PADDING)
                x2 = min(img_data['width'], int(x + w) + PADDING)
                y2 = min(img_data['height'], int(y + h) + PADDING)
                
                # Filter out objects that are too tiny (e.g., dogs in the far background)
                # These are bad for training a 32x32 or 64x64 generator
                if (x2 - x1) < 32 or (y2 - y1) < 32:
                    continue

                # Perform the crop
                cropped_img = original_image.crop((x1, y1, x2, y2))
                
                # Save with unique name: originalName_objIndex.jpg
                base_name = os.path.splitext(img_data['file_name'])[0]
                save_name = f"{base_name}_obj{i}.jpg"
                cropped_img.save(os.path.join(DEST_DIR, save_name))
                count += 1
                
        else:
            # Simple copy of the full image
            shutil.copy(src_path, os.path.join(DEST_DIR, img_data['file_name']))
            count += 1

    print(f"\nDone! Saved {count} images to '{DEST_DIR}'.")
    print(f"You can now point your LatentDataset to '{DEST_DIR}'.")

if __name__ == "__main__":
    main()