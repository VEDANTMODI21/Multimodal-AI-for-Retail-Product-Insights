import torch
import torch.nn as nn
from transformers import ViTModel, BertModel, LlamaForCausalLM, LlamaConfig
from peft import get_peft_model, LoraConfig, TaskType

class MultimodalRetailInsightModel(nn.Module):
    """
    Multimodal AI Framework Integrating Visual, Textual, and Structured Data.
    
    1. ViT-Base/16 for Visual Features (P_V)
    2. BERT-base-uncased for Textual Features (P_T)
    3. MLP for Structured Data (P_S)
    4. Late Fusion via Concatenation and Projection (P_F)
    5. Generation via Llama-2-7B with LoRA fine-tuning
    """
    def __init__(self, num_structured_features, llama_model_name="meta-llama/Llama-2-7b-hf"):
        super(MultimodalRetailInsightModel, self).__init__()
        
        # 1. Visual Feature Extraction: ViT-Base/16
        # Pretrained on ImageNet-21k, output dim: 768
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.visual_dim = self.vit.config.hidden_size # 768
        
        # 2. Textual Feature Extraction: BERT-base-uncased
        # Output dim: 768
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_dim = self.bert.config.hidden_size # 768
        
        # 3. Structured Data Processing: MLP
        self.structured_dim = 128
        self.structured_mlp = nn.Sequential(
            nn.Linear(num_structured_features, 64),
            nn.ReLU(),
            nn.Linear(64, self.structured_dim)
        )
        
        # 4. Fusion Layer (Concatenation -> ReLU Projection -> Heavy Dropout)
        fused_input_dim = self.visual_dim + self.text_dim + self.structured_dim
        self.fusion_dim = 512
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_input_dim, self.fusion_dim),
            nn.ReLU(),
            # Dropout (0.1) solves the "Modality Dominance" problem discussed in the paper
            nn.Dropout(0.1) 
        )
        
        # 5. LLM Setup (Llama-2 with LoRA)
        self.llm_config = LlamaConfig.from_pretrained(llama_model_name)
        self.llm_embedding_dim = self.llm_config.hidden_size # 4096 typically for 7B
        
        # Mapper from multimodal fused space to LLM token embedding space
        self.fusion_to_llm_proj = nn.Linear(self.fusion_dim, self.llm_embedding_dim)
        
        # Note: Actual LoRA instantiations with peft and loading huge weights 
        # is generally handled outside of this module if GPU memory is a concern.
        # However, for completeness, here is how LoRA is set up.
        '''
        self.llm = LlamaForCausalLM.from_pretrained(llama_model_name, torch_dtype=torch.float16, device_map="auto")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        self.llm = get_peft_model(self.llm, peft_config)
        '''

    def forward(self, pixel_values, input_ids, attention_mask, structured_data):
        """
        Forward pass to extract multimodal features and get fused representations.
        
        Args:
            pixel_values: torch.Tensor of shape (batch, 3, 224, 224)
            input_ids: torch.Tensor of shape (batch, seq_len)
            attention_mask: torch.Tensor of shape (batch, seq_len)
            structured_data: torch.Tensor of shape (batch, num_structured_features)
            
        Returns:
            Dictionary containing:
            - fusion_features: (batch, 512) The core multimodal insight vector `h_f`
            - llm_virtual_tokens: (batch, llm_embed_dim) Mapped vector to be concatenated as a prompt prefix
        """
        # Visual processing
        vit_outputs = self.vit(pixel_values=pixel_values)
        # Extract [CLS] token representation `hv`
        h_v = vit_outputs.last_hidden_state[:, 0, :] 
        
        # Textual processing
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract [CLS] token representation `ht`
        h_t = bert_outputs.last_hidden_state[:, 0, :] 
        
        # Structured processing `hs`
        h_s = self.structured_mlp(structured_data)
        
        # Modality Fusion
        h_concat = torch.cat([h_v, h_t, h_s], dim=-1)
        h_f = self.fusion_layer(h_concat)
        
        # Projection to LLM Space
        llm_virtual_tokens = self.fusion_to_llm_proj(h_f)
        
        return {
            "fusion_features": h_f,
            "llm_virtual_tokens": llm_virtual_tokens
        }
