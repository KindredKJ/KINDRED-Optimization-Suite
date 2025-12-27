import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple
import numpy as np
from deap import base, creator, tools
import logging
from datasets import load_dataset
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

# Copyright
__copyright__ = "Copyright (c) 2025 Kindred Cox - KAS-System (Kindred Autonomous System)"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KindredConfig:
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 512,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        intermediate_size: int = 2048,
        head_dim: int = None,
        max_position_embeddings: int = 520,
        rms_norm_eps: float = 1e-8,
        rope_theta: float = 10000.0,
        num_mem_tokens: int = 8,
        ssm_dim: int = 128,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.num_mem_tokens = num_mem_tokens
        self.ssm_dim = ssm_dim
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

class KindredRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class KindredRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 520, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device="cpu", dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class KindredAttention(nn.Module):
    def __init__(self, config: KindredConfig):
        super().__init__()
        self.config = config
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = KindredRotaryEmbedding(config.head_dim, config.max_position_embeddings)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None, use_cache: bool = False):
        bsz, q_len, _ = hidden_states.size()
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(self.config.hidden_size, dim=2)
        query = rearrange(query, 'b q (h d) -> b h q d', h=self.num_heads)
        key = rearrange(key, 'b q (h d) -> b h q d', h=self.num_heads)
        value = rearrange(value, 'b q (h d) -> b h q d', h=self.num_heads)
        cos, sin = self.rotary_emb(value, seq_len=q_len)
        query = (query * cos) + (rotate_half(query) * sin)
        key = (key * cos) + (rotate_half(key) * sin)
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        past_key_value = (key, value) if use_cache else None
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = rearrange(attn_output, 'b h q d -> b q (h d)')
        return self.o_proj(attn_output), past_key_value

class KindredMLP(nn.Module):
    def __init__(self, config: KindredConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor):
        return self.fc2(F.gelu(self.fc1(x)))

class KindredBlock(nn.Module):
    def __init__(self, config: KindredConfig):
        super().__init__()
        self.attn = KindredAttention(config)
        self.mlp = KindredMLP(config)
        self.norm1 = KindredRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = KindredRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None, use_cache: bool = False):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_output, past_key_value = self.attn(hidden_states, attention_mask, position_ids, past_key_value, use_cache)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, past_key_value

class KindredLLM(nn.Module):
    def __init__(self, config: KindredConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([KindredBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = KindredRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.revenue_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, past_key_values: Optional[list] = None, labels: Optional[torch.Tensor] = None):
        bsz, seq_len = input_ids.size()
        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is not None:
            attention_mask = (attention_mask == 0).float() * -1e9
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        past_key_values = past_key_values or [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            hidden_states, past_key_values[i] = layer(hidden_states, attention_mask, position_ids, past_key_values[i], self.config.use_cache)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        revenue_pred = self.revenue_head(hidden_states).mean(dim=1)
        outputs = {"logits": logits, "past_key_values": past_key_values, "revenue_pred": revenue_pred}
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
            outputs["loss"] = loss
        return outputs

    def evolve_weights(self, population_size=10, generations=5, dataset_path="fine_tune_data.jsonl"):
        try:
            if "FitnessMax" in creator.__dict__:
                del creator.FitnessMax
            if "Individual" in creator.__dict__:
                del creator.Individual
            creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            toolbox = base.Toolbox()
            toolbox.register("attr_float", np.random.uniform, -0.1, 0.1)
            param_count = sum(p.numel() for p in self.parameters())
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=param_count)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            def evaluate(individual):
                params = torch.tensor(individual).view_as(list(self.parameters())[0].data)
                original = list(self.parameters())[0].data.clone()
                list(self.parameters())[0].data = params
                try:
                    dataset = load_dataset("json", data_files=dataset_path, split="train") if os.path.exists(dataset_path) else None
                    if dataset:
                        sample = dataset[0]
                        inputs = torch.tensor([tokenizer.encode(sample["input"])])
                        labels = torch.tensor([tokenizer.encode(sample["output"])])
                    else:
                        inputs = torch.randint(0, self.config.vocab_size, (1, 10))
                        labels = inputs.clone()
                    outputs = self(inputs, labels=labels)
                    loss = outputs["loss"].item() if "loss" in outputs else 1.0
                    revenue = outputs["revenue_pred"].mean().item()
                    efficiency = 1.0
                except:
                    loss = 1.0
                    revenue = 0.0
                    efficiency = 1.0
                list(self.parameters())[0].data = original
                return (1.0 / (loss + 1e-8), revenue, efficiency)

            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
            toolbox.register("select", tools.selNSGA2)
            population = toolbox.population(n=population_size)
            for gen in range(generations):
                offspring = toolbox.select(population, len(population))
                offspring = list(map(toolbox.clone, offspring))
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                for mutant in offspring:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                population[:] = offspring
                fits = toolbox.map(toolbox.evaluate, population)
                for ind, fit in zip(population, fits):
                    ind.fitness.values = fit
            best = tools.selBest(population, 1)[0]
            best_params = torch.tensor(best).view_as(list(self.parameters())[0].data)
            list(self.parameters())[0].data = best_params
            logger.info("Evolution completed")
        except Exception as e:
            logger.error(f"Evolution failed: {e}")

    def upgrade_architecture(self):
        if self.config.num_hidden_layers < 16:
            self.config.num_hidden_layers += 1
            self.layers.append(KindredBlock(self.config))
            logger.info(f"Architecture upgraded to {self.config.num_hidden_layers} layers")