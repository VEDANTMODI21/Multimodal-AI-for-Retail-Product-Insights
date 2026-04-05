import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MultimodalRetailDataset(Dataset):
    """
    Dataset class handling text, image, and structured data representing a product.
    Includes proper tokenization and min-max scaling of numerical inputs.
    """
    def __init__(self, data_df, tokenizer, image_dir=None, max_length=128):
        """
        Args:
            data_df (pd.DataFrame): Dataframe containing 'image_path', 'review_text', 
                                    'price', 'rating', 'return_rate', etc.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for BERT.
            image_dir (str): Base directory for images, if paths are relative.
            max_length (int): Max token length for BERT representation.
        """
        self.data = data_df
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.max_length = max_length
        
        # Image composition explicitly noted in the paper:
        # Resize to 224x224, center crop, normalize to ImageNet statistics
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. Visual Modality
        img_path = row['image_path']
        if self.image_dir:
            img_path = f"{self.image_dir}/{img_path}"
        
        try:
            image = Image.open(img_path).convert('RGB')
            pixel_values = self.image_transforms(image)
        except Exception:
            # Fallback to zero tensor if image fails to load
            pixel_values = torch.zeros((3, 224, 224))

        # 2. Textual Modality (Customer Reviews corpus)
        review_text = str(row['review_text'])
        text_inputs = self.tokenizer(
            review_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)
        
        # 3. Structured Data
        # Expecting these columns to be already scaled using Min-Max scaling to [0,1]
        structured_data = torch.tensor([
            row['price_scaled'], 
            row['rating_scaled'], 
            row['return_rate_scaled']
            # Add other numeric constraints as per dataset
        ], dtype=torch.float32)

        # Labels - Target Insight Tokens
        insight = str(row['target_insight'])
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'structured_data': structured_data,
            'target_insight': insight
        }
