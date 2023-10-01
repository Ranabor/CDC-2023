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
        self.yes_id, self.no_id = 4273, 150
        
        prompt = "Would this song be played at the club? Yes or No."
        tokenized_prompt = self.t5_tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
        
        self.prompt_embeds = self.t5.encoder.embed_tokens(tokenized_prompt).to(self.device)
        
        # for name, param in self.t5.named_parameters():
        #     param.requires_grad = False
            
    def forward(self, features, labels):
        input = torch.Tensor(features).to(self.device)
        labels = torch.tensor([[self.yes_id] if label == 'yes' else [self.no_id] for label in labels], dtype=torch.int32).to(self.device)
        # print(input.unsqueeze(dim=2).shape)
        feature_embeds = self.linear(input.unsqueeze(dim=2))
        # print(feature_embeds.shape)
        prompt_embeds = self.prompt_embeds.repeat(feature_embeds.shape[0], 1, 1)
        input_embeds = torch.cat((prompt_embeds, feature_embeds), dim=1)
        # print(input_embeds.shape)
        outputs = self.t5(inputs_embeds=input_embeds.cuda(), labels=labels.type(torch.LongTensor).cuda())
        if self.training:     
            return outputs
        else:
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=2).squeeze()
            yes_probs = torch.index_select(probabilities, 1, torch.tensor([self.yes_id]).cuda())
            no_probs = torch.index_select(probabilities, 1, torch.tensor([self.no_id]).cuda())
            predictions = yes_probs / (yes_probs + no_probs)
            predictions = torch.where(predictions >= 0.5, 4273, 150)
            actual = labels
            return predictions, actual
            
def main():
    
    def collate_fn(batch):
        return {
            'features': torch.stack([item['features'] for item in batch]),
            'labels': [item['labels'] for item in batch]
        }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ClubCompatibilityModel()
    optimizer = AdamW(model.parameters(), lr = 1e-6)
    batch_size = 8
    epochs = [i for i in range(1, 11)]
    
    dataset = SongDataset()
    
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    for epoch in epochs:
        print(f">>> Epoch {epoch}")
        model.train()
        progress = tqdm(total=len(train_dataloader), desc='club_compatibility_index')
        for i, data in enumerate(train_dataloader):
            output = model(data['features'], data['labels'])
            loss = output.loss
            loss.backward(retain_graph=True)
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
                total_correct += torch.sum(predictions.squeeze() == actual.squeeze())
            progress.update()
        progress.close()
        print(f"Accuracy: {total_correct/len(valid_dataset)}")
        
        torch.save(
            model.state_dict(),
            os.path.join("./checkpoints", f"club_compatibility_index-{epoch}.pt"),
        )
    
    model.eval()
    progress = tqdm(total=len(test_dataloader), desc='club_compatibility_index')
    
    total_correct = 0
    for i, data in enumerate(test_dataloader):
        with torch.no_grad():
            predictions, actual = model(data['features'], data['labels'])
            total_correct += torch.sum(predictions.squeeze() == actual.squeeze())
        progress.update()
    progress.close()
    print(f"Accuracy: {total_correct/len(valid_dataset)}")
        
    
if __name__=='__main__':
    main()