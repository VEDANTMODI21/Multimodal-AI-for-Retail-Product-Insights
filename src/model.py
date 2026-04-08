"""
Multimodal Retail Insight Model — Complete Implementation.

Architecture (per paper):
  Product Image  → ViT-Base/16   → h_v (768-d)  ─┐
  Customer Reviews → BERT-base   → h_t (768-d)  ─┼─→ Concat → ReLU Proj → h_f (512-d) → LLM → Insight
  Structured Data  → MLP         → h_s (128-d)  ─┘

LLM Backend:
  - Default: distilgpt2 (~82M params) for 8GB VRAM
  - Production: Llama-2 7B with LoRA (requires ≥16GB VRAM + HF token)
"""
import torch
import torch.nn as nn
from transformers import (
    ViTModel, BertModel,
    GPT2LMHeadModel, GPT2Tokenizer,
    AutoModelForCausalLM, AutoTokenizer
)
from peft import get_peft_model, LoraConfig, TaskType


class MultimodalRetailInsightModel(nn.Module):
    """
    Complete multimodal framework integrating Visual, Textual, and Structured Data
    with a generative LLM for producing human-readable business insights.
    """
    
    def __init__(
        self,
        num_structured_features=3,
        llm_model_name="distilgpt2",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        freeze_encoders=False
    ):
        super().__init__()
        
        self.llm_model_name = llm_model_name
        
        # ═══════════════════════════════════════════════════════════════════
        # 1. VISUAL ENCODER — ViT-Base/16 (Pretrained on ImageNet-21k)
        # ═══════════════════════════════════════════════════════════════════
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.visual_dim = self.vit.config.hidden_size  # 768
        
        if freeze_encoders:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # ═══════════════════════════════════════════════════════════════════
        # 2. TEXTUAL ENCODER — BERT-base-uncased
        # ═══════════════════════════════════════════════════════════════════
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.text_dim = self.bert.config.hidden_size  # 768
        
        if freeze_encoders:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # ═══════════════════════════════════════════════════════════════════
        # 3. STRUCTURED DATA PROCESSOR — MLP + Min-Max (done in dataset)
        # ═══════════════════════════════════════════════════════════════════
        self.structured_dim = 128
        self.structured_mlp = nn.Sequential(
            nn.Linear(num_structured_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, self.structured_dim),
            nn.ReLU()
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # 4. FUSION LAYER — Concatenation + ReLU Projection + Dropout
        #    h_f = ReLU(W_f [h_v ∥ h_t ∥ h_s] + b_f)
        #    Dropout (0.1) prevents Modality Dominance
        # ═══════════════════════════════════════════════════════════════════
        fused_input_dim = self.visual_dim + self.text_dim + self.structured_dim
        self.fusion_dim = 512
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_input_dim, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Anti-modality-dominance dropout
            nn.LayerNorm(self.fusion_dim)
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # 5. LLM — distilgpt2 (default) or Llama-2 with LoRA
        # ═══════════════════════════════════════════════════════════════════
        print(f"Loading LLM: {llm_model_name}")
        
        if "gpt2" in llm_model_name.lower():
            self.llm = GPT2LMHeadModel.from_pretrained(llm_model_name)
            self.llm_tokenizer = GPT2Tokenizer.from_pretrained(llm_model_name)
            self.llm_embedding_dim = self.llm.config.n_embd  # 768 for distilgpt2
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name, torch_dtype=torch.float16, device_map="auto"
            )
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.llm_embedding_dim = self.llm.config.hidden_size
        
        # Ensure pad token exists
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            self.llm.config.pad_token_id = self.llm_tokenizer.eos_token_id
        
        # Apply LoRA to reduce trainable parameters
        if use_lora:
            if "gpt2" in llm_model_name.lower():
                target_modules = ["c_attn"]
            else:
                target_modules = ["q_proj", "v_proj"]
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()
        
        # ═══════════════════════════════════════════════════════════════════
        # 6. PROJECTION: Fusion Space → LLM Embedding Space
        #    Maps h_f (512-d) to N virtual tokens in LLM embedding space
        # ═══════════════════════════════════════════════════════════════════
        self.num_virtual_tokens = 4  # 4 virtual prefix tokens
        self.fusion_to_llm_proj = nn.Sequential(
            nn.Linear(self.fusion_dim, self.llm_embedding_dim * self.num_virtual_tokens),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Prompt prefix embedding for the system directive
        self.prompt_template = (
            "Product analysis: Based on the multimodal product representation "
            "including visual features, customer review sentiment, and structured data, "
            "provide a concise 2-sentence business insight: "
        )
    
    def get_fusion_features(self, pixel_values, input_ids, attention_mask, structured_data):
        """
        Extract and fuse multimodal features.
        Returns h_f: (batch, 512) fused representation.
        """
        # Visual: ViT [CLS] token → h_v (768-d)
        vit_out = self.vit(pixel_values=pixel_values)
        h_v = vit_out.last_hidden_state[:, 0, :]
        
        # Textual: BERT [CLS] token → h_t (768-d)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_t = bert_out.last_hidden_state[:, 0, :]
        
        # Structured: MLP → h_s (128-d)
        h_s = self.structured_mlp(structured_data)
        
        # Fusion: h_f = ReLU(W_f [h_v ∥ h_t ∥ h_s] + b_f)
        h_concat = torch.cat([h_v, h_t, h_s], dim=-1)
        h_f = self.fusion_layer(h_concat)
        
        return h_f
    
    def forward(self, pixel_values, input_ids, attention_mask, structured_data,
                labels_input_ids=None, labels_attention_mask=None):
        """
        Full forward pass — fuse multimodal features and compute LM loss.
        
        Args:
            pixel_values: (B, 3, 224, 224) product images
            input_ids: (B, seq_len) BERT tokenized review text
            attention_mask: (B, seq_len) BERT attention mask
            structured_data: (B, num_features) scaled structured features
            labels_input_ids: (B, label_len) LLM tokenized target insight
            labels_attention_mask: (B, label_len) attention mask for labels
        
        Returns:
            dict with 'loss', 'fusion_features', 'logits'
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # Step 1: Get fused multimodal representation
        h_f = self.get_fusion_features(pixel_values, input_ids, attention_mask, structured_data)
        
        # Step 2: Project fusion features to virtual tokens
        # (B, 512) → (B, num_virtual_tokens * llm_embed_dim)
        virtual_embeds = self.fusion_to_llm_proj(h_f)
        # Reshape to (B, num_virtual_tokens, llm_embed_dim)
        virtual_embeds = virtual_embeds.view(
            batch_size, self.num_virtual_tokens, self.llm_embedding_dim
        )
        
        # Step 3: Tokenize prompt template
        prompt_tokens = self.llm_tokenizer(
            self.prompt_template,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=60
        ).to(device)
        
        # Get prompt embeddings from LLM's word embedding layer
        if hasattr(self.llm, 'base_model'):
            # LoRA wrapped model
            word_embeddings = self.llm.base_model.model.transformer.wte
        else:
            word_embeddings = self.llm.transformer.wte
        
        prompt_embeds = word_embeddings(prompt_tokens["input_ids"])
        # Expand to batch: (1, prompt_len, embed_dim) → (B, prompt_len, embed_dim) 
        prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
        prompt_len = prompt_embeds.shape[1]
        
        if labels_input_ids is not None:
            # ── TRAINING MODE ──
            # Get target embeddings
            target_embeds = word_embeddings(labels_input_ids)
            target_len = target_embeds.shape[1]
            
            # Concatenate: [virtual_tokens | prompt | target_insight]
            full_embeds = torch.cat([virtual_embeds, prompt_embeds, target_embeds], dim=1)
            
            # Build attention mask
            virtual_attn = torch.ones(batch_size, self.num_virtual_tokens, device=device)
            prompt_attn = torch.ones(batch_size, prompt_len, device=device)
            full_attn = torch.cat([virtual_attn, prompt_attn, labels_attention_mask.float()], dim=1)
            
            # Build labels: -100 for virtual+prompt (don't compute loss), actual tokens for target
            ignore_labels = torch.full(
                (batch_size, self.num_virtual_tokens + prompt_len),
                fill_value=-100, dtype=torch.long, device=device
            )
            target_labels = labels_input_ids.clone()
            target_labels[labels_attention_mask == 0] = -100
            full_labels = torch.cat([ignore_labels, target_labels], dim=1)
            
            # Forward through LLM
            outputs = self.llm(
                inputs_embeds=full_embeds,
                attention_mask=full_attn,
                labels=full_labels
            )
            
            return {
                "loss": outputs.loss,
                "fusion_features": h_f,
                "logits": outputs.logits
            }
        else:
            # ── INFERENCE MODE (no labels) ──
            full_embeds = torch.cat([virtual_embeds, prompt_embeds], dim=1)
            virtual_attn = torch.ones(batch_size, self.num_virtual_tokens, device=device)
            prompt_attn = torch.ones(batch_size, prompt_len, device=device)
            full_attn = torch.cat([virtual_attn, prompt_attn], dim=1)
            
            outputs = self.llm(
                inputs_embeds=full_embeds,
                attention_mask=full_attn
            )
            
            return {
                "loss": None,
                "fusion_features": h_f,
                "logits": outputs.logits
            }
    
    @torch.no_grad()
    def generate_insight(self, pixel_values, input_ids, attention_mask, structured_data,
                         max_new_tokens=80, temperature=0.3, top_p=0.9):
        """
        Generate a business insight from multimodal inputs.
        Uses autoregressive decoding with the LLM.
        """
        self.eval()
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # Get fusion features + virtual tokens
        h_f = self.get_fusion_features(pixel_values, input_ids, attention_mask, structured_data)
        virtual_embeds = self.fusion_to_llm_proj(h_f)
        virtual_embeds = virtual_embeds.view(
            batch_size, self.num_virtual_tokens, self.llm_embedding_dim
        )
        
        # Prompt embeddings
        prompt_tokens = self.llm_tokenizer(
            self.prompt_template,
            return_tensors="pt", padding=False, truncation=True, max_length=60
        ).to(device)
        
        if hasattr(self.llm, 'base_model'):
            word_embeddings = self.llm.base_model.model.transformer.wte
        else:
            word_embeddings = self.llm.transformer.wte
        
        prompt_embeds = word_embeddings(prompt_tokens["input_ids"])
        prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
        
        # Initial context = virtual tokens + prompt
        context_embeds = torch.cat([virtual_embeds, prompt_embeds], dim=1)
        context_attn = torch.ones(batch_size, context_embeds.shape[1], device=device)
        
        # Autoregressive generation
        generated_ids = []
        past_key_values = None
        
        for step in range(max_new_tokens):
            if step == 0:
                outputs = self.llm(
                    inputs_embeds=context_embeds,
                    attention_mask=context_attn,
                    use_cache=True
                )
            else:
                # Use cache for efficient generation
                last_token_embed = word_embeddings(next_token_id)
                new_attn = torch.ones(batch_size, 1, device=device)
                context_attn = torch.cat([context_attn, new_attn], dim=1)
                
                outputs = self.llm(
                    inputs_embeds=last_token_embed,
                    attention_mask=context_attn,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :] / max(temperature, 0.01)
            
            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = -float("inf")
            
            probs = torch.softmax(sorted_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            next_token_id = sorted_indices.gather(-1, sampled_idx)
            
            generated_ids.append(next_token_id)
            
            # Stop at EOS
            if (next_token_id == self.llm_tokenizer.eos_token_id).all():
                break
        
        # Decode generated tokens
        generated_ids = torch.cat(generated_ids, dim=-1)
        insights = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return insights
    
    def freeze_encoders(self):
        """Freeze ViT and BERT parameters (for initial training stability)."""
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False
        print("ViT and BERT encoders FROZEN")
    
    def unfreeze_encoders(self):
        """Unfreeze ViT and BERT for fine-tuning."""
        for param in self.vit.parameters():
            param.requires_grad = True
        for param in self.bert.parameters():
            param.requires_grad = True
        print("ViT and BERT encoders UNFROZEN for fine-tuning")
