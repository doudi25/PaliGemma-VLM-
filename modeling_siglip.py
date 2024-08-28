import torch
import torch.nn as nn
import torch.nn.functional as F
class SiglipVisionConfig:
    def __init__(self,hidden_size=768,intermediate_size=3072,num_hidden_layers=12,num_attention_heads=12,num_channels=3,image_size=224,patch_size=16,layer_norm_eps=1e-6,attention_dropout=0.0,num_image_tokens:int=None,**kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens
class SiglipVisionTransformer(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)

    def forward(self,pixel_values:torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embeddings = nn.Conv2d(in_channels=config.num_channels,out_channels=self.embed_dim,kernel_size=self.patch_size,stride=self.patch_size,padding='valid')
        self.num_patches = (self.image_size//self.patch_size)**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions,self.embed_dim)
        self.register_buffer( 'position_ids',torch.arange(self.num_positions).expand((1,-1)),persistent=False)

    def forward(self,pixel_values:torch.FloatTensor)->torch.Tensor:
        _, _, height, width = pixel_values.shape
        # we input to the convolution tensor of shape [batch_size,img_height,img_width,channels] , we get as an output [batch_size,patch_height,patch_width,depth_of_conv=out_channels]
        # patch_height=(height-kernel_size+padding=0)/stride +1
        patch_embeds = self.patch_embeddings(pixel_values)
        # the flatten convert shape into [batch_size,embed_dim,num_patches] then we tranpose it to [batch_size,num_patchs,embed_dim]
        embeddings = patch_embeds.flatten(2).transpose(1,2)
        # positional_encoding as a learnable weight matrix of size [num_patches,embedding_dim]
        embeddings = embeddings+self.position_embedding(self.position_ids)
        return embeddings
class SiglipEncoderLayer(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)

    def forward(self,hidden_states:torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual+hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual+hidden_states
        return hidden_states
class SiglipMLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear1 = nn.Linear(self.hidden_size,self.intermediate_size)
        self.linear2 = nn.Linear(self.intermediate_size,self.hidden_size)

    def forward(self,hidden_size:torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(hidden_size)))

class SiglipAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.embedding_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim= self.embedding_dim//self.num_heads
        self.scale = self.head_dim**(-0.5)
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.q_proj = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.out_proj = nn.Linear(self.embedding_dim,self.embedding_dim)

    def forward(self,hidden_states):
        batch_size, seq_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = query.view(hidden_states.shape[0],hidden_states.shape[1],self.num_heads, self.head_dim).transpose(1,2)
        key = key.view(hidden_states.shape[0],hidden_states.shape[1],self.num_heads,self.head_dim).transpose(1,2)
        value = value.view(hidden_states.shape[0],hidden_states.shape[1],self.num_heads,self.head_dim).transpose(1,2)
        attention_scores = (torch.matmul(query,key.transpose(-1,-2))*self.scale)
        if attention_scores.size() != (batch_size,self.num_heads,seq_len,seq_len):
            raise ValueError(
                f'Attention weights should be of size{(batch_size,self.num_heads,seq_len,seq_len)},but is'
                f'{attention_scores.size()}')
        attention_scores = F.softmax(attention_scores,dim=-1,dtype=torch.float32).to(query.dtype)
        attention_scores = F.dropout(attention_scores,p=self.dropout,training=self.training)
        attention_out = attention_scores@value
        if attention_out.size() != (batch_size,self.num_heads,seq_len,self.head_dim):
            raise ValueError(
                f'attn_output should be of size{(batch_size,self.num_heads,seq_len,self.head_dim)},but is'
                f'{attention_out.size()}'
            )
        hidden_states = attention_out.transpose(1,2).contiguous().view(batch_size,seq_len,self.embedding_dim)
        return self.out_proj(hidden_states),attention_scores

class SiglipEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,inputs_embeds):
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)