import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pandas as pd

torch.cuda.empty_cache()

class ImageTextDataset(Dataset):
    def __init__(self, img_dir, text_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        self.captions_df = pd.read_csv(text_file)
        
    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, idx):
        img_filename = str(self.captions_df.iloc[idx]['public_id']) + ".jpg"
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        caption = str(self.captions_df.iloc[idx]['caption_gt'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, caption
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),             
])

img_dir = '/notebooks/zeroshot_caption/dataset/val'
text_file = '/notebooks/zeroshot_caption/dataset/nice-val-5k.csv'
batch_size = 8

# Create Dataset
dataset = ImageTextDataset(img_dir=img_dir, text_file=text_file, transform=transform)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

from transformers import AdamW

# Prepare optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Number of training epochs
num_epochs = 1

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in data_loader:
        images = batch[0]
        captions = batch[1]

        # Preprocess images and captions
        inputs = [processor(image, text, return_tensors="pt", padding="max_length").to("cuda", torch.float16) for image, text in zip(images, captions)]

        # Concatenate batch data
        input_ids = torch.cat([input["input_ids"] for input in inputs], dim=0)
        attention_mask = torch.cat([input["attention_mask"] for input in inputs], dim=0)
        pixel_values = torch.cat([input["pixel_values"] for input in inputs], dim=0)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # Save the model after each epoch
    model.save_pretrained(f"blip_finetuned_epoch_{epoch}")
