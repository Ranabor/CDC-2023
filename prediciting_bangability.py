from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os

class BangabilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.linear = torch.nn.Linear(1, 768)
        self.t5 = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
        self.t5_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
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
            return outputs
                
            
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BangabilityModel()
    optimizer = AdamW(model.parameters(), lr = 1e-5)
    batch_size = 8
    epochs = [1, 2, 3]
    
    dataset = SongDataset()
    def collate_fn(batch):
        return {
            'features': torch.stack([item['features'] for item in batch]),
            'labels': torch.stack([item['label'] for item in batch])
        }
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    for epoch in epochs:
        print(">>> Epoch {epoch}")
        progress = tqdm(total=len(train_dataloader), desc='bangability')
        for i, data in enumerate(train_dataloader):
            output = model(data['features'], data['labels'])
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
        progress.close()
        
        torch.save(
            model.state_dict(),
            os.path.join("./checkpoints", f"vit_bart_latest-{epoch}.pt"),
        )
    
if __name__=='__main__':
    main()