from transformers import Wav2Vec2Model
import torch 
import torch.nn as nn

class model(nn.Module):
    def __init__(self, num_classes: int =6, freeze_base: bool = True):
        super().__init__()
        
        self.wave2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if freeze_base:
            for param in self.wave2vec.parameters():
                param.requires_grad = False
            
            for param in self.wave2vec.encoder.layers[-2:].parameters():
                param.requires_grad = True
        

        hidden_size = self.wave2vec.config.hidden_size
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size,128),
            nn.Tanh(),
            nn.Linear(128,1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size,512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512,256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, values: torch.Tensor) ->torch.Tensor:
        output = self.wave2vec(values)
        hidden_states = output.last_hidden_state
        attn_logits = self.attention(hidden_states)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = (hidden_states*attn_weights).sum(dim=1)
        
        return self.classifier(pooled)