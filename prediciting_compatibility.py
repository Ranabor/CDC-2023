from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
import os
from dataset import SongDataset


class ClubCompatibilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.linear = torch.nn.Linear(1, 768)
        self.linear.to(self.device)
        self.t5 = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
        self.t5.to(self.device)
        self.t5_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
        self.t5_tokenizer.to(self.device)
        self.yes_id, self.no_id = 4273, 150
        
        prompt = "Would this song be played at the club? Yes or No."
        tokenized_prompt = self.t5_tokenizer(prompt, return_tensor='pt')
        self.prompt_embeds = self.t5.encoder.embed_tokens(tokenized_prompt)
        
        for name, param in self.t5.named_parameters():
            param.requires_grad = False
            
        def forward(self, features, labels):
            input = torch.Tensor(features)
            labels = torch.Tensor([self.yes_id if label == 'yes' else self.no_id for label in labels])
            feature_embeds = self.linear(input)
            
            input_embeds = torch.cat((self.prompt_embeds, feature_embeds), dim=0)
            outputs = self.t5(input_embeds=input_embeds, labels=labels)
            if self.training:     
                
                return outputs
            else:
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                yes_probs = torch.index_select(probabilities, 1, [self.yes_id])
                no_probs = torch.index_select(probabilities, 1, [self.no_id])
                predictions = yes_probs / (yes_probs + no_probs)
                predictions = torch.where(predictions >= 0.5, 1.0, 0.0)
                actual = labels
                return predictions, actual
            
def main():
    
    def collate_fn(batch):
        return {
            'features': torch.stack([item['features'] for item in batch]),
            'labels': torch.stack([item['label'] for item in batch])
        }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ClubCompatibilityModel()
    optimizer = AdamW(model.parameters(), lr = 1e-5)
    batch_size = 8
    epochs = [1, 2, 3]
    
    dataset = SongDataset()
    
    
    train_dataset, valid_dataset, test_dataset = random_split(train_dataset, [0.6, 0.2, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    for epoch in epochs:
        print(">>> Epoch {epoch}")
        model.train()
        progress = tqdm(total=len(train_dataloader), desc='club_compatibility_index')
        for i, data in enumerate(train_dataloader):
            output = model(data['features'], data['labels'])
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
        progress.close()
        
        model.eval()
        progress = tqdm(total=len(valid_dataloader), desc='club_compatibility_index')
        
        total_correct = 0
        for i, data in enumerate(valid_dataloader):
            with torch.no_grad():
                predictions, actual = model(data['features'], data['labels'])
                total_correct += torch.sum(predictions == actual)
            progress.update()
        print(f"Accuracy: {total_correct/len(valid_dataloader)}")
        progress.close()
        
        for i, data in enumerate(valid_dataloader):
            output = model(data['features'])
        progress.close()
        
        torch.save(
            model.state_dict(),
            os.path.join("./checkpoints", f"vit_bart_latest-{epoch}.pt"),
        )
    
if __name__=='__main__':
    main()