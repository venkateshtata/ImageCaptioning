import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from PIL import Image
import random
import pandas as pd

IMAGE_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 128

class Projection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Projection, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, tokenizer, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = self.captions[idx]
        encoded_caption = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Masking tokens randomly
        input_ids = encoded_caption['input_ids'].squeeze()
        labels = input_ids.clone()
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < 0.15) * (input_ids != 101) * (input_ids != 102) * (input_ids != 0)
        selection = []

        for i in range(input_ids.shape[0]):
            if mask_arr[i]:
                selection.append(i)

        input_ids[selection] = 103

        return image, input_ids, labels

transform = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 9.224, 9.225])])


df = pd.read_csv("./dataset/nice-val-5k.csv")

image_paths = []
captions = []

for i in df.index:
    image_paths.append("./dataset/val/" + str(df['public_id'][i]) + ".jpg")
    captions.append(df['caption_gt'][i])




tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = ImageCaptionDataset(image_paths, captions, tokenizer, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

vision_model = resnet50(pretrained=True)
for param in vision_model.parameters():
    param.requires_grad = False
vision_model.eval()

bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
bert_model.train()

optimizer = AdamW(bert_model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    for images, input_ids, labels in dataloader:
        
        with torch.no_grad():
            image_features = vision_model(images)
            
        input_ids = input_ids.squeeze(1)
        
        concatenated_features = torch.cat((image_features, input_ids), dim=1)
        
        outputs = bert_model(input_ids=concatenated_features, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item()}")


# model_save_path = "./models/bert_image_captioning_model.pth"
# torch.save(bert_model.state_dict(), model_save_path)

# Load the model
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# bert_model.load_state_dict(torch.load(model_save_path))
# bert_model.eval()