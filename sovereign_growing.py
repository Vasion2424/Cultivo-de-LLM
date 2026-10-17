from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURACION DEL ENTRENAMIENTO
# ============================================================
D_MODEL = 32
HEADS = 2
CONTEXT = 256
VOCAB_SIZE = 256                 # Bytes UTF-8

LEARNING_RATE = 0.0002
REPLAY_RATIO = 0.50
BATCH_SIZE = 2
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42

EVAL_INTERVAL = 200
STAGNATION_EVALS = 4
PLATEAU_TOLERANCE = 0.01
MAX_GENERALIZATION_GAP = 0.30
ACTIVE_THRESHOLD = 0.85
USED_THRESHOLD = 0.70

CREATE_INTERVAL = 50
CONSERVE_INTERVAL = 10
WINDOW = 200
PRUNE_PROBABILITY = 0.25

MODEL_FAMILY = "sovereign_growing_dynamic_v4"
SUPPORTED_MODEL_FAMILIES = {"sovereign_growing_dynamic_v3", "sovereign_growing_dynamic_v4"}
FORMAT_VERSION = 4
SUPPORTED_FORMAT_VERSIONS = {3, 4}


# ============================================================
# 1. FFN: CAPACIDAD Y AMPLIACION DE ANCHURA
# ============================================================
class SovereignFFN(nn.Module):
    def __init__(self, d_model=32, capacity=32, initial_active=24):
        super().__init__()

        self.d_model = int(d_model)
        self.capacity = int(capacity)
        self.initial_active = min(int(initial_active), self.capacity)

        if self.d_model < 1 or self.capacity < 1:
            raise ValueError("d_model y capacity deben ser positivos.")

        self.lin_in = nn.Linear(self.d_model, self.capacity)
        self.lin_out = nn.Linear(self.capacity, self.d_model)

        active = torch.zeros(self.capacity, dtype=torch.bool)
        active[:self.initial_active] = True

        mask_in = torch.zeros(self.d_model, self.capacity)
        mask_in[:, :self.initial_active] = 1.0

        mask_out = torch.zeros(self.capacity, self.d_model)
        mask_out[:self.initial_active, :] = 1.0

        self.register_buffer("active", active)
        self.register_buffer("mask_in", mask_in)
        self.register_buffer("mask_out", mask_out)

    def forward(self, x):
        weight_in = self.lin_in.weight * self.mask_in.T

        pre = F.linear(
            x,
            weight_in,
            self.lin_in.bias * self.active
        )
        hidden = F.gelu(pre) * self.active

        weight_out = self.lin_out.weight * self.mask_out.T
        output = F.linear(hidden, weight_out, self.lin_out.bias)

        input_use = (
            x.detach().unsqueeze(-1)
            * weight_in.T[None, None, :, :]
        ).abs().mean(dim=(0, 1))

        output_use = (
            hidden.detach().unsqueeze(-1)
            * weight_out.T[None, None, :, :]
        ).abs().mean(dim=(0, 1))

        node_activity = hidden.detach().abs().mean(dim=(0, 1))

        return output, input_use, output_use, node_activity

    def expand_dimensions(self, new_d_model=None, new_capacity=None):
        """Expande dimensiones sin reducirlas y conserva los pesos existentes."""
        new_d_model = self.d_model if new_d_model is None else int(new_d_model)
        new_capacity = self.capacity if new_capacity is None else int(new_capacity)

        if new_d_model < self.d_model or new_capacity < self.capacity:
            raise ValueError("No se permite reducir dimensiones de la FFN.")
        if new_d_model == self.d_model and new_capacity == self.capacity:
            return

        old_d = self.d_model
        old_c = self.capacity
        device = self.lin_in.weight.device
        dtype = self.lin_in.weight.dtype

        new_in = nn.Linear(new_d_model, new_capacity).to(device=device, dtype=dtype)
        new_out = nn.Linear(new_capacity, new_d_model).to(device=device, dtype=dtype)

        with torch.no_grad():
            # Las conexiones nuevas empiezan en cero; las reglas de creación
            # y el gradiente podrán desarrollarlas sin alterar de golpe lo viejo.
            new_in.weight.zero_()
            new_in.bias.zero_()
            new_out.weight.zero_()
            new_out.bias.zero_()

            new_in.weight[:old_c, :old_d].copy_(self.lin_in.weight)
            new_in.bias[:old_c].copy_(self.lin_in.bias)
            new_out.weight[:old_d, :old_c].copy_(self.lin_out.weight)
            new_out.bias[:old_d].copy_(self.lin_out.bias)

        self.lin_in = new_in
        self.lin_out = new_out

        active = torch.zeros(new_capacity, dtype=torch.bool, device=device)
        active[:old_c] = self.active
        self.active = active

        mask_in = torch.zeros(
            new_d_model, new_capacity,
            dtype=self.mask_in.dtype, device=device
        )
        mask_in[:old_d, :old_c] = self.mask_in
        self.mask_in = mask_in

        mask_out = torch.zeros(
            new_capacity, new_d_model,
            dtype=self.mask_out.dtype, device=device
        )
        mask_out[:old_c, :old_d] = self.mask_out
        self.mask_out = mask_out

        self.d_model = new_d_model
        self.capacity = new_capacity

    def expand_capacity(self, new_capacity):
        self.expand_dimensions(new_capacity=new_capacity)


# ============================================================
# 2. BLOQUE TRANSFORMER
# ============================================================
class Block(nn.Module):
    def __init__(self, d_model=32, heads=2, capacity=32):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, heads, batch_first=True
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = SovereignFFN(d_model, capacity)

    def forward(self, x, causal_mask):
        normalized = self.ln1(x)

        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False
        )

        x = x + attended

        ff_output, input_use, output_use, node_activity = (
            self.ffn(self.ln2(x))
        )

        x = x + ff_output

        return x, (input_use, output_use, node_activity)


# ============================================================
# 3. TRANSFORMER CON ARQUITECTURA DINAMICA
# ============================================================
class MiniLM(nn.Module):
    def __init__(
        self,
        capacity=32,
        depth=1,
        d_model=D_MODEL,
        heads=HEADS,
        context=CONTEXT
    ):
        super().__init__()

        capacity, depth = int(capacity), int(depth)
        d_model, heads, context = int(d_model), int(heads), int(context)
        if min(capacity, depth, d_model, heads, context) < 1:
            raise ValueError("Capacidad, profundidad, d_model, heads y context deben ser positivos.")
        if d_model % heads != 0:
            raise ValueError(
                f"d_model={d_model} debe ser divisible por heads={heads} "
                "para MultiheadAttention."
            )

        self.capacity = capacity
        self.d_model = d_model
        self.heads = heads
        self.context = context

        self.token = nn.Embedding(VOCAB_SIZE, self.d_model)
        self.pos = nn.Embedding(self.context, self.d_model)

        self.blocks = nn.ModuleList([
            Block(
                d_model=self.d_model,
                heads=self.heads,
                capacity=self.capacity
            )
            for _ in range(depth)
        ])

        self.ln_final = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, VOCAB_SIZE)

    @property
    def depth(self):
        return len(self.blocks)

    def forward(self, idx):
        length = idx.size(1)
        if length > self.context:
            raise ValueError(
                f"Entrada de {length} bytes; máximo {self.context}."
            )

        positions = torch.arange(length, device=idx.device)
        x = self.token(idx) + self.pos(positions)[None, :, :]

        causal_mask = torch.triu(
            torch.ones(
                length, length,
                dtype=torch.bool,
                device=idx.device
            ),
            diagonal=1
        )

        signals = []
        for block in self.blocks:
            x, block_signals = block(x, causal_mask)
            signals.append(block_signals)

        return self.head(self.ln_final(x)), signals

    @staticmethod
    def _resize_layer_norm(old_layer, new_size):
        if old_layer.normalized_shape == (new_size,):
            return old_layer
        device = old_layer.weight.device if old_layer.elementwise_affine else None
        dtype = old_layer.weight.dtype if old_layer.elementwise_affine else None
        kwargs = {"device": device, "dtype": dtype} if device is not None else {}
        new_layer = nn.LayerNorm(
            new_size, eps=old_layer.eps,
            elementwise_affine=old_layer.elementwise_affine,
            **kwargs
        )
        old_size = old_layer.normalized_shape[0]
        if old_layer.elementwise_affine:
            with torch.no_grad():
                new_layer.weight[:old_size].copy_(old_layer.weight)
                new_layer.bias[:old_size].copy_(old_layer.bias)
                # Nuevos canales comienzan neutros, pero entrenables.
                new_layer.weight[old_size:].fill_(1.0)
                new_layer.bias[old_size:].zero_()
        return new_layer

    @staticmethod
    def _resize_attention(old_attn, new_d_model, new_heads):
        old_d = old_attn.embed_dim
        old_heads = old_attn.num_heads
        if old_d == new_d_model and old_heads == new_heads:
            return old_attn

        device = old_attn.out_proj.weight.device
        dtype = old_attn.out_proj.weight.dtype
        new_attn = nn.MultiheadAttention(
            embed_dim=new_d_model,
            num_heads=new_heads,
            dropout=old_attn.dropout,
            bias=old_attn.in_proj_bias is not None,
            add_bias_kv=(old_attn.bias_k is not None),
            add_zero_attn=old_attn.add_zero_attn,
            batch_first=True
        ).to(device=device, dtype=dtype)

        # Conserva las porciones Q/K/V aprendidas, respetando que PyTorch
        # concatena esas tres matrices en bloques separados de embed_dim.
        if old_attn.in_proj_weight is not None and new_attn.in_proj_weight is not None:
            with torch.no_grad():
                new_attn.in_proj_weight[:, :].normal_(0.0, 0.02)
                if new_attn.in_proj_bias is not None:
                    new_attn.in_proj_bias.zero_()
                for part in range(3):
                    old_rows = slice(part * old_d, (part + 1) * old_d)
                    new_rows = slice(part * new_d_model, part * new_d_model + old_d)
                    new_attn.in_proj_weight[new_rows, :old_d].copy_(
                        old_attn.in_proj_weight[old_rows, :old_d]
                    )
                    if old_attn.in_proj_bias is not None:
                        new_attn.in_proj_bias[new_rows].copy_(
                            old_attn.in_proj_bias[old_rows]
                        )
                    if new_d_model > old_d:
                        # Las proyecciones viejas ignoran las nuevas entradas
                        # inicialmente; pueden aprenderlas durante el cultivo.
                        new_attn.in_proj_weight[new_rows, old_d:].zero_()

        with torch.no_grad():
            new_attn.out_proj.weight[:old_d, :old_d].copy_(old_attn.out_proj.weight)
            new_attn.out_proj.bias[:old_d].copy_(old_attn.out_proj.bias)
            if new_d_model > old_d:
                # El canal viejo no recibe una perturbación aleatoria de los nuevos.
                new_attn.out_proj.weight[:old_d, old_d:].zero_()
                # Las salidas nuevas pueden activarse y recibir gradientes.
                new_attn.out_proj.bias[old_d:].zero_()

            if old_attn.bias_k is not None and new_attn.bias_k is not None:
                new_attn.bias_k[:, :, :old_d].copy_(old_attn.bias_k)
                new_attn.bias_k[:, :, old_d:].zero_()
            if old_attn.bias_v is not None and new_attn.bias_v is not None:
                new_attn.bias_v[:, :, :old_d].copy_(old_attn.bias_v)
                new_attn.bias_v[:, :, old_d:].zero_()

        return new_attn

    def expand_geometry(
        self,
        new_capacity=None,
        new_d_model=None,
        new_heads=None,
        new_context=None
    ):
        """Aumenta dimensiones estructurales conservando los pesos compatibles."""
        new_capacity = self.capacity if new_capacity is None else int(new_capacity)
        new_d_model = self.d_model if new_d_model is None else int(new_d_model)
        new_heads = self.heads if new_heads is None else int(new_heads)
        new_context = self.context if new_context is None else int(new_context)

        if (new_capacity < self.capacity or new_d_model < self.d_model
                or new_heads < self.heads or new_context < self.context):
            raise ValueError("El cultivo estructural solo permite aumentar dimensiones.")
        if min(new_capacity, new_d_model, new_heads, new_context) < 1:
            raise ValueError("Las dimensiones deben ser positivas.")
        if new_d_model % new_heads != 0:
            raise ValueError("d_model nuevo debe ser divisible por heads nuevos.")
        if (new_capacity, new_d_model, new_heads, new_context) == (
                self.capacity, self.d_model, self.heads, self.context):
            return

        old_d, old_context = self.d_model, self.context
        device = self.token.weight.device
        dtype = self.token.weight.dtype

        if new_d_model != old_d:
            old_token = self.token
            new_token = nn.Embedding(VOCAB_SIZE, new_d_model).to(device=device, dtype=dtype)
            with torch.no_grad():
                new_token.weight[:, :old_d].copy_(old_token.weight)
                new_token.weight[:, old_d:].zero_()
            self.token = new_token

            old_head = self.head
            new_head = nn.Linear(new_d_model, VOCAB_SIZE).to(
                device=old_head.weight.device, dtype=old_head.weight.dtype
            )
            with torch.no_grad():
                new_head.weight[:, :old_d].copy_(old_head.weight)
                new_head.weight[:, old_d:].zero_()
                new_head.bias.copy_(old_head.bias)
            self.head = new_head

            self.ln_final = self._resize_layer_norm(self.ln_final, new_d_model).to(device)

        if new_context != old_context or new_d_model != old_d:
            old_pos = self.pos
            new_pos = nn.Embedding(new_context, new_d_model).to(
                device=old_pos.weight.device, dtype=old_pos.weight.dtype
            )
            with torch.no_grad():
                rows = min(old_context, new_context)
                cols = min(old_d, new_d_model)
                new_pos.weight[:rows, :cols].copy_(old_pos.weight[:rows, :cols])
                if new_d_model > old_d:
                    new_pos.weight[:rows, old_d:].zero_()
            self.pos = new_pos

        for block in self.blocks:
            if new_d_model != old_d:
                block.ln1 = self._resize_layer_norm(block.ln1, new_d_model).to(device)
                block.ln2 = self._resize_layer_norm(block.ln2, new_d_model).to(device)
                block.attn = self._resize_attention(
                    block.attn, new_d_model, new_heads
                )
            elif new_heads != self.heads:
                block.attn = self._resize_attention(
                    block.attn, new_d_model, new_heads
                )

            if new_d_model != old_d or new_capacity != self.capacity:
                block.ffn.expand_dimensions(
                    new_d_model=new_d_model,
                    new_capacity=new_capacity
                )

        self.capacity = new_capacity
        self.d_model = new_d_model
        self.heads = new_heads
        self.context = new_context

    def expand_width(self, new_capacity):
        if new_capacity <= self.capacity:
            raise ValueError("La nueva anchura debe ser mayor.")
        old_capacity = self.capacity
        self.expand_geometry(new_capacity=new_capacity)
        return old_capacity, self.capacity

    def add_block(self):
        """Añade un bloque aproximadamente identidad al comienzo."""
        device = self.token.weight.device
        block = Block(
            d_model=self.d_model,
            heads=self.heads,
            capacity=self.capacity
        ).to(device)

        with torch.no_grad():
            block.attn.out_proj.weight.zero_()
            block.attn.out_proj.bias.zero_()
            block.ffn.lin_out.weight.zero_()
            block.ffn.lin_out.bias.zero_()

        self.blocks.append(block)
        return block

    def architecture_config(self):
        return {
            "capacity": self.capacity,
            "depth": self.depth,
            "d_model": self.d_model,
            "heads": self.heads,
            "context": self.context,
            "vocab_size": VOCAB_SIZE
        }


# ============================================================
# 4. REGLAS DE VIDA POR BLOQUE
# ============================================================
class SovereignRules:
    def __init__(
        self,
        ffn,
        seed=42,
        prune_probability=PRUNE_PROBABILITY,
        conserve_interval=CONSERVE_INTERVAL,
        create_interval=CREATE_INTERVAL,
        window=WINDOW,
        min_active=4
    ):
        self.ffn = ffn
        self.rng = torch.Generator(device="cpu").manual_seed(seed + 77777)

        self.prune_probability = float(prune_probability)
        self.conserve_interval = int(conserve_interval)
        self.create_interval = int(create_interval)
        self.window = int(window)

        self.capacity = ffn.capacity
        self.d_model = ffn.d_model
        self.min_active = min(int(min_active), self.capacity)
        self.cycle = 0

        self.life_in = torch.zeros(self.d_model, self.capacity)
        self.life_out = torch.zeros(self.capacity, self.d_model)
        self.node_life = torch.zeros(self.capacity)

        n = ffn.initial_active
        self.life_in[:, :n] = 10
        self.life_out[:n, :] = 10
        self.node_life[:n] = 5

        self.in_mutated = torch.zeros_like(self.life_in, dtype=torch.bool)
        self.out_mutated = torch.zeros_like(self.life_out, dtype=torch.bool)
        self.node_mutated = torch.zeros_like(self.node_life, dtype=torch.bool)

        self.usage_in = torch.zeros(
            self.window, self.d_model, self.capacity, dtype=torch.uint8
        )
        self.usage_out = torch.zeros(
            self.window, self.capacity, self.d_model, dtype=torch.uint8
        )
        self.usage_node = torch.zeros(
            self.window, self.capacity, dtype=torch.uint8
        )

        self.counters = {
            "created_neurons": 0,
            "created_paths": 0,
            "pruned_paths": 0,
            "pruned_neurons": 0,
            "conserved_paths": 0,
            "conserved_neurons": 0,
            "mutated_paths": 0,
            "mutated_neurons": 0
        }

        self.sync_masks()

    @torch.no_grad()
    def sync_masks(self):
        active = self.node_life > 0

        self.ffn.active.copy_(active.to(self.ffn.active.device))

        mask_in = ((self.life_in > 0) & active.unsqueeze(0)).float()
        mask_out = ((self.life_out > 0) & active.unsqueeze(1)).float()

        self.ffn.mask_in.copy_(mask_in.to(self.ffn.mask_in.device))
        self.ffn.mask_out.copy_(mask_out.to(self.ffn.mask_out.device))

    @torch.no_grad()
    def tick(self, input_use, output_use, node_activity):
        self.cycle += 1
        c = self.cycle
        slot = c % self.window

        x = input_use.detach().float().cpu()
        y = output_use.detach().float().cpu()
        activity = node_activity.detach().float().cpu()

        mask_in = self.ffn.mask_in.detach().cpu().bool()
        mask_out = self.ffn.mask_out.detach().cpu().bool()
        active = self.node_life > 0

        tx = max(float(x.max()) * 0.10, 1e-12)
        ty = max(float(y.max()) * 0.10, 1e-12)
        tn = max(float(activity.max()) * 0.10, 1e-12)

        use_in = (x > tx) & mask_in
        use_out = (y > ty) & mask_out
        use_node = (activity > tn) & active

        self.usage_in[slot].copy_(use_in.to(torch.uint8))
        self.usage_out[slot].copy_(use_out.to(torch.uint8))
        self.usage_node[slot].copy_(use_node.to(torch.uint8))

        # CONSERVE Y PODA
        if c % self.conserve_interval == 0:
            ids = [
                (c - k) % self.window
                for k in range(self.conserve_interval)
            ]

            node_count = self.usage_node[ids].sum(0)
            input_count = self.usage_in[ids].sum(0)
            output_count = self.usage_out[ids].sum(0)

            keep_nodes = (node_count >= 8) & (self.node_life > 0)
            keep_in = (input_count >= 8) & mask_in
            keep_out = (output_count >= 8) & mask_out

            self.node_life[keep_nodes] += 1
            self.life_in[keep_in] += 1
            self.life_out[keep_out] += 1

            self.counters["conserved_neurons"] += int(keep_nodes.sum())
            self.counters["conserved_paths"] += int(
                keep_in.sum() + keep_out.sum()
            )

            old_in = self.life_in.clone()
            old_out = self.life_out.clone()
            old_nodes = self.node_life.clone()

            unused_in = torch.nonzero(
                (self.life_in > 0) & (self.usage_in[ids].sum(0) == 0),
                as_tuple=False
            ).tolist()

            for i, h in unused_in:
                if torch.rand((), generator=self.rng).item() < self.prune_probability:
                    self.life_in[i, h] = max(
                        0, self.life_in[i, h].item() - 1
                    )

            unused_out = torch.nonzero(
                (self.life_out > 0) & (self.usage_out[ids].sum(0) == 0),
                as_tuple=False
            ).tolist()

            for h, j in unused_out:
                if torch.rand((), generator=self.rng).item() < self.prune_probability:
                    self.life_out[h, j] = max(
                        0, self.life_out[h, j].item() - 1
                    )

            unused_nodes = torch.nonzero(
                (self.node_life > 0) & (self.usage_node[ids].sum(0) == 0),
                as_tuple=False
            ).flatten().tolist()

            for h in unused_nodes:
                if int((self.node_life > 0).sum()) <= self.min_active:
                    break

                if torch.rand((), generator=self.rng).item() < self.prune_probability:
                    self.node_life[h] = max(
                        0, self.node_life[h].item() - 1
                    )

            self.counters["pruned_paths"] += int(
                ((old_in > 0) & (self.life_in <= 0)).sum()
                + ((old_out > 0) & (self.life_out <= 0)).sum()
            )

            self.counters["pruned_neurons"] += int(
                ((old_nodes > 0) & (self.node_life <= 0)).sum()
            )

        # MUTACION
        if c % self.window == 0:
            node_count = self.usage_node.sum(0)
            input_count = self.usage_in.sum(0)
            output_count = self.usage_out.sum(0)

            mutate_nodes = (
                (node_count >= 120)
                & (self.node_life > 0)
                & ~self.node_mutated
            )

            self.node_mutated[mutate_nodes] = True
            self.node_life[mutate_nodes] += 30

            self.counters["mutated_neurons"] += int(mutate_nodes.sum())

            mutate_in = (
                (input_count >= 120)
                & (self.life_in > 0)
                & ~self.in_mutated
            )
            mutate_out = (
                (output_count >= 120)
                & (self.life_out > 0)
                & ~self.out_mutated
            )

            self.in_mutated[mutate_in] = True
            self.out_mutated[mutate_out] = True

            self.life_in[mutate_in] += 60
            self.life_out[mutate_out] += 60

            self.counters["mutated_paths"] += int(
                mutate_in.sum() + mutate_out.sum()
            )

        # CREAR
        if c % self.create_interval == 0:
            reserve = torch.nonzero(self.node_life <= 0).flatten()

            if len(reserve) > 0:
                h = int(reserve[-1])
                x_idx = int(torch.argmax(x).item()) // self.capacity
                y_idx = int(torch.argmax(y.sum(0)).item())

                self.node_life[h] = 6
                self.life_in[x_idx, h] = 11
                self.life_out[h, y_idx] = 11

                self.node_mutated[h] = False
                self.in_mutated[:, h] = False
                self.out_mutated[h, :] = False

                self.ffn.lin_in.weight[h, x_idx].normal_(0, 0.18)
                self.ffn.lin_out.weight[y_idx, h].normal_(0, 0.18)

                self.counters["created_neurons"] += 1
                self.counters["created_paths"] += 2

        self.sync_masks()

    def expand(self, new_capacity, new_d_model=None):
        """Redimensiona memoria/reglas conservando historia y habilitando los nuevos canales."""
        new_capacity = int(new_capacity)
        new_d_model = self.d_model if new_d_model is None else int(new_d_model)
        if new_capacity < self.capacity or new_d_model < self.d_model:
            raise ValueError("No se permite reducir capacidad ni d_model.")
        if new_capacity == self.capacity and new_d_model == self.d_model:
            self.sync_masks()
            return

        old_capacity, old_d_model = self.capacity, self.d_model

        def pad(tensor, shape):
            result = torch.zeros(shape, dtype=tensor.dtype)
            slices = tuple(slice(0, min(a, b)) for a, b in zip(tensor.shape, shape))
            result[slices] = tensor[slices]
            return result

        self.life_in = pad(self.life_in, (new_d_model, new_capacity))
        self.life_out = pad(self.life_out, (new_capacity, new_d_model))
        self.node_life = pad(self.node_life, (new_capacity,))
        self.in_mutated = pad(self.in_mutated, (new_d_model, new_capacity))
        self.out_mutated = pad(self.out_mutated, (new_capacity, new_d_model))
        self.node_mutated = pad(self.node_mutated, (new_capacity,))
        self.usage_in = pad(self.usage_in, (self.window, new_d_model, new_capacity))
        self.usage_out = pad(self.usage_out, (self.window, new_capacity, new_d_model))
        self.usage_node = pad(self.usage_node, (self.window, new_capacity))

        self.capacity = new_capacity
        self.d_model = new_d_model
        self.min_active = min(self.min_active, self.capacity)

        if new_d_model > old_d_model:
            # Nuevos canales conectan inicialmente con neuronas ya vivas, pero
            # sus pesos arrancan en cero: pueden aprender sin saltos aleatorios.
            active = self.node_life > 0
            self.life_in[old_d_model:new_d_model, :] = active.float().unsqueeze(0)
            self.life_out[:new_capacity, old_d_model:new_d_model] = active.float().unsqueeze(1)

        self.sync_masks()

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "cycle": self.cycle,
            "counters": self.counters,
            "life_in": self.life_in,
            "life_out": self.life_out,
            "node_life": self.node_life,
            "in_mutated": self.in_mutated,
            "out_mutated": self.out_mutated,
            "node_mutated": self.node_mutated,
            "usage_in": self.usage_in,
            "usage_out": self.usage_out,
            "usage_node": self.usage_node,
            "rng_state": self.rng.get_state()
        }

    def load_state_dict(self, state):
        if int(state["capacity"]) != self.ffn.capacity:
            raise ValueError("Reglas y FFN tienen capacidades distintas.")

        self.capacity = int(state["capacity"])
        self.cycle = int(state["cycle"])
        self.counters = dict(state["counters"])

        for name in (
            "life_in", "life_out", "node_life",
            "in_mutated", "out_mutated", "node_mutated",
            "usage_in", "usage_out", "usage_node"
        ):
            setattr(self, name, state[name].clone().cpu())

        self.rng.set_state(state["rng_state"])
        self.sync_masks()

    def report(self):
        active = self.node_life > 0
        recently_used = self.usage_node.sum(0) > 0

        active_count = int(active.sum())
        used_count = int((active & recently_used).sum())

        return {
            **self.counters,
            "cycles": self.cycle,
            "active_neurons": active_count,
            "recently_used_neurons": used_count,
            "capacity": self.capacity,
            "active_ratio": active_count / max(1, self.capacity),
            "used_ratio": used_count / max(1, self.capacity),
        }


# ============================================================
# 5. DATOS Y EVALUACION
# ============================================================
def read_raw(filename, context=CONTEXT):
    path = Path(filename)

    if not path.is_file():
        raise FileNotFoundError(f"No existe el dataset: {path}")

    data = path.read_text(encoding="utf-8-sig").strip().encode("utf-8")

    context = int(context)
    if len(data) < context + 10:
        raise ValueError(
            f"{path} es demasiado corto para contexto {context}: {len(data)} bytes disponibles."
        )

    return data


def to_tensor(data):
    return torch.tensor(list(data), dtype=torch.long)


def split_bytes(data, context=CONTEXT):
    context = int(context)
    cut = int(len(data) * 0.95)
    cut = max(context + 2, min(cut, len(data) - 2))
    return data[:cut], data[cut:]


def get_batch(source, batch_size=BATCH_SIZE, context=CONTEXT):
    context = int(context)
    max_start = len(source) - context - 1

    if max_start < 0:
        raise ValueError("Corpus de entrenamiento demasiado corto.")

    starts = torch.randint(0, max_start + 1, (batch_size,))

    x = torch.stack([
        source[int(i):int(i) + context] for i in starts
    ])

    y = torch.stack([
        source[int(i) + 1:int(i) + context + 1] for i in starts
    ])

    return x, y


def mixed_batch(new_data, replay_data=None, context=CONTEXT):
    if replay_data is None:
        x, y = get_batch(new_data, context=context)
        return x, y, "nuevo:100%"

    new_x, new_y = get_batch(new_data, 1, context=context)
    old_x, old_y = get_batch(replay_data, 1, context=context)

    x = torch.cat([new_x, old_x], dim=0)
    y = torch.cat([new_y, old_y], dim=0)

    order = torch.randperm(BATCH_SIZE)

    return x[order], y[order], "nuevo:50%/anterior:50%"


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()

    total_loss = 0.0
    total_bytes = 0

    for start in range(0, len(data) - 1, model.context):
        chunk = data[start:start + model.context + 1]

        if len(chunk) < 2:
            continue

        x = chunk[:-1].unsqueeze(0).to(device)
        y = chunk[1:].unsqueeze(0).to(device)

        logits, _ = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
            reduction="sum"
        )

        total_loss += float(loss.item())
        total_bytes += y.numel()

    model.train()
    return total_loss / max(1, total_bytes)


def macro_validation_score(current_losses, baseline_losses, best_domain_losses=None, retention_tolerance=0.10):
    """Pérdida de validación equilibrada entre corpus y con tolerancia a regresión.

    Cada corpus pesa lo mismo independientemente de su longitud. El score es
    adimensional (pérdida actual / pérdida de referencia); menor es mejor.
    Una regresión frente al mejor resultado individual solo se penaliza cuando
    supera retention_tolerance, para no bloquear el cultivo por fluctuaciones pequeñas.
    """
    keys = [key for key in current_losses if key in baseline_losses]
    if not keys:
        raise ValueError("No hay corpus de validación en común para calcular el score.")

    ratios = {
        key: float(current_losses[key]) / max(float(baseline_losses[key]), 1e-8)
        for key in keys
    }
    base_score = sum(ratios.values()) / len(ratios)

    penalties = []
    if best_domain_losses:
        for key in keys:
            best = max(float(best_domain_losses.get(key, baseline_losses[key])), 1e-8)
            regression = float(current_losses[key]) / best - 1.0
            penalties.append(max(0.0, regression - retention_tolerance))
    retention_penalty = (sum(penalties) / len(penalties)) * 0.25 if penalties else 0.0
    return base_score + retention_penalty, ratios, retention_penalty


def default_best_checkpoint_path(checkpoint_path):
    path = Path(checkpoint_path).expanduser().resolve()
    suffix = path.suffix or ".pt"
    return path.with_name(path.stem + ".best" + suffix)


# ============================================================
# 6. CHECKPOINTS
# ============================================================
def load_checkpoint(filename, device):
    """Carga el checkpoint principal y recurre a .bak si falta o está dañado."""
    path = Path(filename).expanduser().resolve()
    backup_path = path.with_suffix(path.suffix + ".bak")
    candidates = [path, backup_path]
    errors = []

    for candidate in candidates:
        if not candidate.is_file():
            continue

        try:
            try:
                ckpt = torch.load(
                    candidate, map_location=device, weights_only=False
                )
            except TypeError:
                ckpt = torch.load(candidate, map_location=device)

            if ckpt.get("model_family") not in SUPPORTED_MODEL_FAMILIES:
                raise ValueError("Checkpoint de arquitectura incompatible.")
            if ckpt.get("format_version") not in SUPPORTED_FORMAT_VERSIONS:
                raise ValueError("Versión de checkpoint incompatible.")

            if candidate == backup_path:
                print(
                    "[RECUPERACION] El checkpoint principal no se pudo usar. "
                    f"Se restaurará la copia de seguridad: {backup_path}",
                    flush=True
                )
            return ckpt

        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            if candidate == path:
                print(
                    "[AVISO] No se pudo cargar el checkpoint principal; "
                    "se comprobará el respaldo .bak.",
                    flush=True
                )

    if not errors:
        raise FileNotFoundError(
            f"No existe el checkpoint ni su respaldo: {path} y {backup_path}"
        )

    raise RuntimeError(
        "No se pudo recuperar ningún checkpoint. Detalles: " + " | ".join(errors)
    )


def save_checkpoint(path, model, rules, optimizer, global_step, metrics):
    """Guarda atómicamente; conserva el checkpoint anterior en <archivo>.bak."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    backup_path = path.with_suffix(path.suffix + ".bak")
    backup_temp_path = path.with_suffix(path.suffix + ".bak.tmp")

    config = {
        **model.architecture_config(),
        "learning_rate": LEARNING_RATE,
        "replay_ratio": REPLAY_RATIO,
    }

    payload = {
        "model_family": MODEL_FAMILY,
        "format_version": FORMAT_VERSION,
        "model_state": model.state_dict(),
        "rules_state": [rule.state_dict() for rule in rules],
        "optimizer_state": optimizer.state_dict(),
        "global_step": int(global_step),
        "config": config,
        "metrics": metrics,
    }

    try:
        # Escribir y sincronizar el archivo temporal antes de tocar el válido.
        with temp_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())

        # Protege la última versión válida antes de reemplazar el checkpoint.
        # Si el primer guardado falla, no hay archivo previo que copiar.
        if path.is_file():
            with path.open("rb") as source, backup_temp_path.open("wb") as dest:
                shutil.copyfileobj(source, dest)
                dest.flush()
                os.fsync(dest.fileno())
            os.replace(backup_temp_path, backup_path)

        # El reemplazo es atómico: el archivo principal no queda escrito a medias.
        os.replace(temp_path, path)

    except BaseException:
        for temporary in (temp_path, backup_temp_path):
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass
        raise

    return path

def save_metrics_json(path, metrics):
    """Escribe el JSON de métricas de forma atómica y sincronizada."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(metrics, indent=2, ensure_ascii=False))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise


def make_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


# ============================================================
# 7. DIAGNOSTICO Y CRECIMIENTO AUTOMATICO
# ============================================================
def capacity_pressure(rules):
    reports = [rule.report() for rule in rules]

    pressure = bool(reports) and all(
        report["active_ratio"] >= ACTIVE_THRESHOLD
        and report["used_ratio"] >= USED_THRESHOLD
        for report in reports
    )

    return pressure, reports


def _next_width(current, maximum):
    if maximum is not None and current >= maximum:
        return current
    if current < 48:
        target = 48
    elif current < 64:
        target = 64
    else:
        target = max(current + 1, (current * 3 + 1) // 2)
    if maximum is not None:
        target = min(target, maximum)
    return target if target > current else current


def _next_context(current, maximum, data_ceiling):
    if maximum is not None and current >= maximum:
        return current
    target = max(current + 1, (current * 3 + 1) // 2)
    if maximum is not None:
        target = min(target, maximum)
    if data_ceiling is not None:
        target = min(target, data_ceiling)
    return target if target > current else current


def expand_model(
    model, rules, max_capacity, max_depth, max_d_model, max_heads,
    max_context, seed, growth_events, data_context_ceiling=None
):
    """Elige de forma rotativa una dimensión que aún pueda crecer.

    La elección solo se invoca tras estancamiento, brecha aceptable y presión
    de capacidad. None en un límite significa que el programa no impone ese tope.
    """
    width_target = _next_width(model.capacity, max_capacity)
    context_target = _next_context(model.context, max_context, data_context_ceiling)

    width_allowed = width_target > model.capacity
    depth_allowed = max_depth is None or model.depth < max_depth
    next_heads = model.heads + 1
    head_dim = model.d_model // model.heads
    next_d_model = head_dim * next_heads
    attention_allowed = (
        (max_heads is None or next_heads <= max_heads)
        and (max_d_model is None or next_d_model <= max_d_model)
        and next_d_model > model.d_model
    )
    context_allowed = context_target > model.context

    eligible = {
        "width": width_allowed,
        "depth": depth_allowed,
        "attention": attention_allowed,
        "context": context_allowed,
    }
    order = ["width", "depth", "attention", "context"]
    last_type = growth_events[-1].get("type") if growth_events else None
    start_index = (order.index(last_type) + 1) % len(order) if last_type in order else 0
    growth_type = None
    for offset in range(len(order)):
        candidate = order[(start_index + offset) % len(order)]
        if eligible[candidate]:
            growth_type = candidate
            break

    if growth_type is None:
        print(
            "[SIN CRECIMIENTO] Se agotaron los límites configurados o el corpus "
            "no contiene suficientes bytes para ampliar el contexto."
        )
        return None

    if growth_type == "width":
        old = model.capacity
        model.expand_geometry(new_capacity=width_target)
        for rule in rules:
            rule.expand(model.capacity, model.d_model)
        for rule in rules:
            rule.sync_masks()
        print(f"\n[CRECIMIENTO] Anchura FFN: {old} -> {model.capacity} por bloque.")
        print("[INFO] Las neuronas FFN nuevas empiezan inactivas.")
        return "width"

    if growth_type == "depth":
        if max_depth is not None and model.depth >= max_depth:
            return None
        block = model.add_block()
        rule = SovereignRules(block.ffn, seed=seed + 1000 + model.depth)
        rule.sync_masks()
        rules.append(rule)
        print(f"\n[CRECIMIENTO] Profundidad Transformer: {model.depth - 1} -> {model.depth} bloques.")
        print("[INFO] El bloque nuevo empieza aproximadamente como identidad.")
        return "depth"

    if growth_type == "attention":
        old_d, old_heads = model.d_model, model.heads
        model.expand_geometry(new_d_model=next_d_model, new_heads=next_heads)
        for rule in rules:
            rule.expand(model.capacity, model.d_model)
        for rule in rules:
            rule.sync_masks()
        print(
            f"\n[CRECIMIENTO] Dimensión interna/cabezas: d_model {old_d} -> "
            f"{model.d_model}; heads {old_heads} -> {model.heads}."
        )
        print("[INFO] Se conserva la parte compatible de los pesos y se habilitan conexiones para los canales nuevos.")
        return "attention"

    if growth_type == "context":
        old = model.context
        model.expand_geometry(new_context=context_target)
        print(f"\n[CRECIMIENTO] Contexto: {old} -> {model.context} bytes por secuencia.")
        return "context"

    return None


def consider_growth(
    model,
    rules,
    optimizer,
    train_history,
    validation_history,
    train_loss,
    val_loss,
    max_capacity,
    max_depth,
    max_d_model,
    max_heads,
    max_context,
    data_context_ceiling,
    step,
    growth_events,
    evaluations_since_growth=999999,
    post_growth_grace_evals=8,
):
    """Prueba crecimiento tras un estancamiento medido, sin exigir saturación estricta.

    Con presión de capacidad puede crecer tras cuatro evaluaciones estancadas.
    Si no hay presión, exige un plateau persistente de al menos ocho evaluaciones.
    Después de una expansión deja tiempo para medir el efecto antes de volver a crecer.
    """
    n = min(len(validation_history), len(train_history))
    if n < STAGNATION_EVALS:
        return optimizer, False

    old_val = validation_history[-STAGNATION_EVALS]
    new_val = validation_history[-1]
    old_train = train_history[-STAGNATION_EVALS]
    new_train = train_history[-1]

    val_gain = (old_val - new_val) / max(abs(old_val), 1e-8)
    train_gain = (old_train - new_train) / max(abs(old_train), 1e-8)
    gap = (val_loss - train_loss) / max(abs(train_loss), 1e-8)

    print(
        f"[DIAGNOSTICO] Mejora macro-validación={val_gain * 100:+.2f}% | "
        f"mejora train-probe={train_gain * 100:+.2f}% | "
        f"brecha principal={gap * 100:+.2f}%"
    )

    # Una caída fuerte de validación puede ser ruido o un problema de datos.
    # No se congela el cultivo por una brecha alta aislada: esta se informa,
    # pero solo posponemos crecimiento si las métricas acaban de empeorar mucho.
    if val_gain < -0.02 or train_gain < -0.05:
        print("[DECISION] Posponer crecimiento una evaluación: las métricas empeoraron de forma marcada.")
        return optimizer, False

    stalled = (val_gain < PLATEAU_TOLERANCE and train_gain < PLATEAU_TOLERANCE)
    persistent_plateau = False
    long_val_gain = None
    long_train_gain = None
    if n >= 8:
        long_val_gain = (validation_history[-8] - validation_history[-1]) / max(abs(validation_history[-8]), 1e-8)
        long_train_gain = (train_history[-8] - train_history[-1]) / max(abs(train_history[-8]), 1e-8)
        persistent_plateau = (
            long_val_gain < PLATEAU_TOLERANCE
            and long_train_gain < PLATEAU_TOLERANCE
        )

    pressure, reports = capacity_pressure(rules)
    if not (stalled and pressure) and not persistent_plateau:
        print("[DECISION] Mantener estructura: aún hay progreso o el estancamiento es demasiado reciente.")
        return optimizer, False

    grace = max(1, int(post_growth_grace_evals))
    if evaluations_since_growth < grace:
        print(
            f"[DECISION] Evaluar la expansión actual antes de otra: "
            f"{evaluations_since_growth}/{grace} evaluaciones desde el último crecimiento."
        )
        return optimizer, False

    if pressure:
        print("[DECISION] Estancamiento con presión de capacidad: se permite probar crecimiento.")
    else:
        print(
            "[DECISION] Estancamiento prolongado sin saturación estricta: "
            "se permite una expansión exploratoria para comprobar si ayuda."
        )
    if gap > MAX_GENERALIZATION_GAP:
        print("[AVISO] Brecha de generalización alta; se permite el experimento, pero se vigilará la retención por corpus.")

    old_architecture = model.architecture_config()
    kind = expand_model(
        model=model,
        rules=rules,
        max_capacity=max_capacity,
        max_depth=max_depth,
        max_d_model=max_d_model,
        max_heads=max_heads,
        max_context=max_context,
        seed=SEED,
        growth_events=growth_events,
        data_context_ceiling=data_context_ceiling,
    )
    if kind is None:
        return optimizer, False

    growth_events.append({
        "step": int(step),
        "type": kind,
        "old_architecture": old_architecture,
        "new_architecture": model.architecture_config(),
        "old_capacity": old_architecture["capacity"],
        "new_capacity": model.capacity,
        "old_depth": old_architecture["depth"],
        "new_depth": model.depth,
        "validation_plateau_gain": val_gain,
        "training_plateau_gain": train_gain,
        "long_validation_plateau_gain": long_val_gain,
        "long_training_plateau_gain": long_train_gain,
        "generalization_gap": gap,
        "capacity_pressure": pressure,
        "block_reports_before_growth": reports,
    })

    optimizer = make_optimizer(model)
    print("[INFO] Optimizador reiniciado tras la expansión estructural.")
    print(f"[INFO] Se observarán al menos {grace} evaluaciones antes de permitir otra expansión.")
    return optimizer, True

def final_diagnosis(
    model,
    rules,
    initial_train_loss,
    final_train_loss,
    initial_eval,
    final_eval,
    primary_name,
    max_capacity,
    max_depth,
    max_d_model,
    max_heads,
    max_context,
    growth_events
):
    train_gain = (
        initial_train_loss - final_train_loss
    ) / max(abs(initial_train_loss), 1e-8)

    initial_val = initial_eval[primary_name]
    final_val = final_eval[primary_name]

    val_gain = (
        initial_val - final_val
    ) / max(abs(initial_val), 1e-8)

    gap = (final_val - final_train_loss) / max(abs(final_train_loss), 1e-8)

    pressure, reports = capacity_pressure(rules)
    capacity_limit_text = str(max_capacity) if max_capacity is not None else "sin límite"
    depth_limit_text = str(max_depth) if max_depth is not None else "sin límite"
    d_model_limit_text = str(max_d_model) if max_d_model is not None else "sin límite"
    heads_limit_text = str(max_heads) if max_heads is not None else "sin límite"
    context_limit_text = str(max_context) if max_context is not None else "sin límite"

    print("\n" + "=" * 56)
    print("DIAGNOSTICO FINAL DEL MINI-GPT")
    print("=" * 56)
    print(f"Bloques Transformer: {model.depth}/{depth_limit_text}")
    print(f"Anchura FFN por bloque: {model.capacity}/{capacity_limit_text}")
    print(f"d_model: {model.d_model}/{d_model_limit_text} | heads: {model.heads}/{heads_limit_text}")
    print(f"Contexto: {model.context}/{context_limit_text} bytes | vocabulario byte: {VOCAB_SIZE} fijo")
    print(f"Mejora entrenamiento: {train_gain * 100:+.2f}%")
    print(f"Mejora validación actual: {val_gain * 100:+.2f}%")
    print(f"Brecha validación/entrenamiento: {gap * 100:+.2f}%")

    for index, report in enumerate(reports, start=1):
        print(
            f"Bloque {index}: activas "
            f"{report['active_neurons']}/{report['capacity']} "
            f"({report['active_ratio'] * 100:.1f}%), "
            f"uso reciente={report['used_ratio'] * 100:.1f}%"
        )

    print("\n--- MENSAJE DEL SISTEMA ---")

    if gap > MAX_GENERALIZATION_GAP:
        message = (
            "No ampliar todavía. La validación queda muy por detrás "
            "del entrenamiento; investigar datos, replay y sobreajuste."
        )
    elif val_gain >= PLATEAU_TOLERANCE:
        message = (
            "La validación sigue mejorando. Mantener la estructura actual "
            "y volver a evaluar antes de expandir."
        )
    elif pressure:
        message = (
            "Hay presión estructural. Si la validación se estanca, el cultivo puede "
            "probar anchura, profundidad, dimensión/cabezas de atención o contexto "
            "según las opciones y datos disponibles."
        )
    elif train_gain > PLATEAU_TOLERANCE and val_gain < PLATEAU_TOLERANCE:
        message = (
            "Entrenamiento mejora pero validación no. No ampliar aún: "
            "investigar sobreajuste o diferencias entre corpus."
        )
    else:
        message = (
            "No hay evidencia suficiente para afirmar que haga falta "
            "más capacidad. Mantener y reunir nuevas evaluaciones."
        )

    print(message)

    # Si los corpus muestran tendencias opuestas, sugiere estudiar
    # una red especializada, pero nunca la crea automáticamente.
    domain_changes = {}

    for name in initial_eval:
        before = initial_eval[name]
        after = final_eval[name]
        change = (before - after) / max(abs(before), 1e-8)
        domain_changes[name] = change
        print(
            f"Corpus {name}: {before:.5f} -> {after:.5f} "
            f"({change * 100:+.2f}% mejora)"
        )

    improved = [name for name, value in domain_changes.items() if value >= 0.05]
    worsened = [name for name, value in domain_changes.items() if value <= -0.05]

    specialist = "No hay señal clara de conflicto entre corpus."
    if improved and worsened:
        specialist = (
            "Posible conflicto entre dominios: mejora en "
            + ", ".join(improved)
            + " y empeora en "
            + ", ".join(worsened)
            + ". Probar replay acumulativo antes de decidir si conviene "
              "una red especializada."
        )

    print("\n[ESPECIALIZACION] " + specialist)

    if growth_events:
        print("\nExpansiones realizadas:")
        for event in growth_events:
            print(
                f"Paso {event['step']}: {event['type']} | "
                f"arquitectura {event.get('old_architecture', {})} -> "
                f"{event.get('new_architecture', {'capacity': event.get('new_capacity'), 'depth': event.get('new_depth')})}"
            )
    else:
        print("\nNo hubo expansiones automáticas en esta ejecución.")

    print(
        "\nEste diagnóstico mide pérdida y actividad estructural; "
        "no mide directamente comprensión semántica."
    )
    print("=" * 56)

    return {
        "train_gain": train_gain,
        "validation_gain": val_gain,
        "generalization_gap": gap,
        "capacity_pressure": pressure,
        "message": message,
        "domain_changes": domain_changes,
        "specialist_message": specialist,
        "block_reports": reports,
        "growth_events": growth_events,
    }


# ============================================================
# 8. ENTRENAMIENTO
# ============================================================
def train(args):
    # Las rutas de salida se vuelven absolutas para que siempre sepamos dónde están.
    args.out = str(Path(args.out).expanduser().resolve())
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Carga primero el checkpoint para conocer el contexto real al reanudar.
    ckpt = load_checkpoint(args.checkpoint, device) if args.checkpoint else None
    if ckpt is not None:
        data_context = int(ckpt.get("config", {}).get("context", CONTEXT))
    else:
        data_context = args.context if args.context is not None else CONTEXT

    dataset_raw = read_raw(args.dataset, context=data_context)
    new_train_raw, new_val_raw = split_bytes(dataset_raw, context=data_context)

    new_train = to_tensor(new_train_raw)
    new_val = to_tensor(new_val_raw)

    replay_train = None
    replay_val_raw = None
    replay_path = None

    if args.replay_dataset:
        replay_path = Path(args.replay_dataset).resolve()
        replay_raw = read_raw(args.replay_dataset, context=data_context)
        replay_train_raw, replay_val_raw = split_bytes(replay_raw, context=data_context)
        replay_train = to_tensor(replay_train_raw)

    primary_name = f"{args.dataset} [holdout 5%]"
    eval_sets = {primary_name: new_val}
    dataset_path = Path(args.dataset).resolve()

    for filename in args.eval_dataset:
        resolved = Path(filename).resolve()

        if resolved == dataset_path:
            continue

        if replay_path is not None and resolved == replay_path:
            eval_sets[filename] = to_tensor(replay_val_raw)
            continue

        raw = read_raw(filename, context=data_context)
        stem = Path(filename).stem.lower()

        if stem.endswith(("_val", "_test", "_validation")):
            heldout = raw
        else:
            _, heldout = split_bytes(raw, context=data_context)

        eval_sets[filename] = to_tensor(heldout)

    optimizer_state = None
    global_step = 0
    prior_events = []

    if args.checkpoint:
        if any(value is not None for value in (
            args.capacity, args.depth, args.d_model, args.heads, args.context
        )):
            raise ValueError(
                "Al reanudar, no uses --capacity, --depth, --d-model, --heads ni --context. "
                "La estructura inicial se restaura desde el checkpoint; los límites máximos siguen disponibles."
            )

        config = ckpt["config"]

        model = MiniLM(
            capacity=int(config["capacity"]),
            depth=int(config["depth"]),
            d_model=int(config["d_model"]),
            heads=int(config["heads"]),
            context=int(config["context"])
        ).to(device)

        model.load_state_dict(ckpt["model_state"])

        if len(ckpt["rules_state"]) != model.depth:
            raise ValueError("Las reglas no coinciden con la profundidad.")

        rules = []

        for i, block in enumerate(model.blocks):
            rule = SovereignRules(block.ffn, seed=SEED + i)
            rule.load_state_dict(ckpt["rules_state"][i])
            rules.append(rule)

        global_step = int(ckpt["global_step"])
        optimizer_state = ckpt.get("optimizer_state")
        prior_events = (
            ckpt.get("metrics", {}).get("diagnosis", {}).get("growth_events", [])
        )

        if args.max_capacity is not None and model.capacity > args.max_capacity:
            raise ValueError("max-capacity es menor que la capacidad del checkpoint.")
        if args.max_depth is not None and model.depth > args.max_depth:
            raise ValueError("max-depth es menor que la profundidad del checkpoint.")
        if args.max_d_model is not None and model.d_model > args.max_d_model:
            raise ValueError("max-d-model es menor que d_model del checkpoint.")
        if args.max_heads is not None and model.heads > args.max_heads:
            raise ValueError("max-heads es menor que heads del checkpoint.")
        if args.max_context is not None and model.context > args.max_context:
            raise ValueError("max-context es menor que el contexto del checkpoint.")

        print(
            f"[CHECKPOINT] Restaurado: FFN={model.capacity}, profundidad={model.depth}, "
            f"d_model={model.d_model}, heads={model.heads}, contexto={model.context}"
        )

    else:
        initial_capacity = args.capacity if args.capacity is not None else 32
        initial_depth = args.depth if args.depth is not None else 1
        initial_d_model = args.d_model if args.d_model is not None else D_MODEL
        initial_heads = args.heads if args.heads is not None else HEADS
        initial_context = args.context if args.context is not None else CONTEXT

        if args.max_capacity is not None and initial_capacity > args.max_capacity:
            raise ValueError("Capacidad inicial superior a max-capacity.")
        if args.max_depth is not None and initial_depth > args.max_depth:
            raise ValueError("Profundidad inicial superior a max-depth.")
        if args.max_d_model is not None and initial_d_model > args.max_d_model:
            raise ValueError("d_model inicial superior a max-d-model.")
        if args.max_heads is not None and initial_heads > args.max_heads:
            raise ValueError("heads inicial superior a max-heads.")
        if args.max_context is not None and initial_context > args.max_context:
            raise ValueError("contexto inicial superior a max-context.")

        model = MiniLM(
            capacity=initial_capacity,
            depth=initial_depth,
            d_model=initial_d_model,
            heads=initial_heads,
            context=initial_context
        ).to(device)

        rules = [
            SovereignRules(block.ffn, seed=SEED + i)
            for i, block in enumerate(model.blocks)
        ]

    for rule in rules:
        rule.sync_masks()

    # El contexto puede crecer hasta donde los datos de entrenamiento permitan
    # formar secuencias completas; con más datos, ese techo natural aumenta.
    context_sources = [len(new_train)]
    if replay_train is not None:
        context_sources.append(len(replay_train))
    data_context_ceiling = max(1, min(context_sources) - 1)

    optimizer = make_optimizer(model)

    if optimizer_state:
        try:
            optimizer.load_state_dict(optimizer_state)
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_RATE
        except (ValueError, RuntimeError, KeyError):
            print("[AVISO] Se reinicia el optimizador.")
            optimizer = make_optimizer(model)

    probe = new_train[-min(8192, len(new_train)):]
    initial_train_probe = evaluate(model, probe, device)

    initial_eval = {
        name: evaluate(model, data, device)
        for name, data in eval_sets.items()
    }

    # Control del cultivo: evalúa todos los corpus, registra el mejor estado
    # validado, mide estancamiento y puede detenerse tras un plateau prolongado.
    eval_signature = sorted(eval_sets.keys())
    stored_control = {}
    if ckpt is not None:
        stored_control = ckpt.get("metrics", {}).get("cultivation_control", {}) or {}

    default_best_path = default_best_checkpoint_path(args.out)
    stored_best_path = stored_control.get("best_checkpoint_path")
    best_path = Path(stored_best_path).expanduser().resolve() if stored_best_path else default_best_path
    control_compatible = (
        isinstance(stored_control, dict)
        and stored_control.get("eval_signature") == eval_signature
        and best_path.is_file()
        and isinstance(stored_control.get("baseline_eval"), dict)
    )

    if control_compatible:
        control_state = dict(stored_control)
        control_state["best_checkpoint_path"] = str(best_path)
        control_state["eval_signature"] = eval_signature
        control_state.setdefault("best_domain_losses", dict(initial_eval))
        control_state.setdefault("best_macro_score", 1.0)
        control_state.setdefault("best_checkpoint_score", control_state["best_macro_score"])
        control_state.setdefault("patience_reference_score", control_state["best_macro_score"])
        control_state.setdefault("stale_evaluations", 0)
        control_state.setdefault("total_evaluations", 0)
        control_state.setdefault("evaluations_since_growth", args.post_growth_grace_evals)
        control_state.setdefault("growth_validation_history", [])
        control_state.setdefault("growth_train_history", [])
        control_state.setdefault("evaluation_history", [])
        control_state.setdefault("last_growth_global_step", None)
        control_state.setdefault("last_growth_score", control_state.get("best_macro_score", 1.0))
        control_state.setdefault("growth_trial_assessed_for_step", None)
        control_state.setdefault("growth_trials_since_best", 0)
        control_state.setdefault("stop_reason", None)
        print("[CONTROL CULTIVO] Se recupera el historial de validación y estancamiento del checkpoint.")
    else:
        best_path = default_best_path
        baseline_score, _, _ = macro_validation_score(initial_eval, initial_eval, initial_eval, args.retention_tolerance)
        control_state = {
            "version": 1,
            "eval_signature": eval_signature,
            "baseline_eval": {key: float(value) for key, value in initial_eval.items()},
            "best_domain_losses": {key: float(value) for key, value in initial_eval.items()},
            "best_macro_losses": {key: float(value) for key, value in initial_eval.items()},
            "best_macro_score": float(baseline_score),
            "best_checkpoint_score": float(baseline_score),
            "patience_reference_score": float(baseline_score),
            "stale_evaluations": 0,
            "total_evaluations": 0,
            # Before the first expansion, do not impose a post-growth waiting period.
            "evaluations_since_growth": max(1, int(args.post_growth_grace_evals)),
            "growth_validation_history": [],
            "growth_train_history": [],
            "evaluation_history": [],
            "last_growth_global_step": None,
            "last_growth_score": float(baseline_score),
            "growth_trial_assessed_for_step": None,
            "growth_trials_since_best": 0,
            "best_step": global_step,
            "best_checkpoint_path": str(best_path),
            "stop_reason": None,
        }
        print("[CONTROL CULTIVO] Se inicia un nuevo registro para los corpus evaluados.")

    # Reanudar manualmente un cultivo que se declaró estabilizado inicia una nueva
    # campaña de prueba: conserva sus mejores métricas, pero reinicia los contadores
    # que de otro modo volverían a detenerlo en la primera evaluación.
    if control_compatible and stored_control.get("stop_reason"):
        print("[CONTROL CULTIVO] Nueva campaña solicitada tras una parada previa: se reinician contadores de paciencia y expansiones fallidas.")
        control_state["stale_evaluations"] = 0
        control_state["growth_trials_since_best"] = 0
        control_state["evaluations_since_growth"] = max(1, int(args.post_growth_grace_evals))
        control_state["growth_validation_history"] = []
        control_state["growth_train_history"] = []
        control_state["last_growth_global_step"] = None
        control_state["growth_trial_assessed_for_step"] = None

    # El conjunto de evaluación debe ser el mismo para comparar scores entre etapas.
    control_state["best_checkpoint_path"] = str(best_path)
    control_state["stop_reason"] = None
    if (not control_compatible) or (not best_path.is_file()):
        best_initial_metrics = {
            "dataset": args.dataset,
            "replay_dataset": args.replay_dataset,
            "global_step": global_step,
            "autosave_stage": "initial_best_validation",
            "final_config": model.architecture_config(),
            "initial_eval": initial_eval,
            "cultivation_control": control_state,
            "diagnosis": {"growth_events": prior_events},
        }
        save_checkpoint(best_path, model, rules, optimizer, global_step, best_initial_metrics)
        print(f"[MEJOR CHECKPOINT] Estado inicial guardado en {best_path}", flush=True)

    print("\n========== CULTIVO ==========")
    print(f"Dataset: {args.dataset}")
    print(f"Replay: {args.replay_dataset or 'ninguno'}")
    print(
        f"Arquitectura inicial/actual: FFN={model.capacity}, profundidad={model.depth}, "
        f"d_model={model.d_model}, heads={model.heads}, contexto={model.context} bytes"
    )
    print(
        "Límites opcionales: "
        f"FFN={args.max_capacity if args.max_capacity is not None else 'sin límite'}, "
        f"profundidad={args.max_depth if args.max_depth is not None else 'sin límite'}, "
        f"d_model={args.max_d_model if args.max_d_model is not None else 'sin límite'}, "
        f"heads={args.max_heads if args.max_heads is not None else 'sin límite'}, "
        f"contexto={args.max_context if args.max_context is not None else 'sin límite'}"
    )
    print(f"Vocabulario: {VOCAB_SIZE} valores byte (cubre todo UTF-8 en bytes)")
    print(f"Techo de contexto por datos disponibles: {data_context_ceiling} bytes")
    print(f"Learning rate fijo: {LEARNING_RATE}")
    print(f"Replay fijo: {REPLAY_RATIO:.2f}")
    print(f"Pasos: {args.steps}")
    print(f"Autoguardado cada: {args.autosave_interval} pasos y en cada evaluación")
    print(f"Evaluación conjunta: {len(eval_sets)} corpus (peso igual por corpus)")
    print(f"Parada por estancamiento: {args.early_stop_patience_evals} evaluaciones sin mejora útil (0=desactivada)")
    print(f"Paciencia tras crecer: {args.post_growth_grace_evals} evaluaciones")
    print(f"Checkpoint principal: {Path(args.out).resolve()}")
    print(f"Mejor checkpoint validado: {best_path}")
    print(f"Respaldo anterior: {Path(args.out).resolve()}.bak")
    print("=============================\n")

    model.train()

    train_losses = []
    validation_history = list(control_state.get("growth_validation_history", []))
    train_probe_history = list(control_state.get("growth_train_history", []))
    growth_events = list(prior_events)
    last_train_probe_loss = initial_train_probe
    last_validation_loss = initial_eval[primary_name]
    metrics_path = Path(args.out).with_suffix(".metrics.json")

    # Crear un checkpoint inicial antes de entrenar. Así, incluso un fallo
    # temprano conserva al menos el estado desde el que comenzó esta ejecución.
    initial_metrics = {
        "dataset": args.dataset,
        "replay_dataset": args.replay_dataset,
        "steps_this_run": 0,
        "global_step": global_step,
        "autosave": True,
        "autosave_stage": "initial",
        "final_config": model.architecture_config(),
        "train_probe_loss_at_save": last_train_probe_loss,
        "validation_loss_at_save": last_validation_loss,
        "initial_eval": initial_eval,
        "cultivation_control": control_state,
        "best_checkpoint_path": str(best_path),
        "diagnosis": {"growth_events": growth_events},
    }
    save_checkpoint(
        args.out, model, rules, optimizer, global_step, initial_metrics
    )
    save_metrics_json(metrics_path, initial_metrics)
    print(f"[AUTOGUARDADO] Estado inicial guardado: {Path(args.out).resolve()}", flush=True)

    # El motivo de parada se conserva en el informe final; completar --steps no
    # se interpreta automáticamente como convergencia del modelo.
    stop_reason = None
    steps_completed = 0

    # Ctrl+C se trata como parada voluntaria: se guarda inmediatamente el estado
    # actual y se sale sin ejecutar la evaluación final larga.
    try:
        for step in range(1, args.steps + 1):
            x, y, source_mix = mixed_batch(new_train, replay_train, context=model.context)

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, signals = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            for rule, (input_use, output_use, node_activity) in zip(rules, signals):
                rule.tick(input_use, output_use, node_activity)

            global_step += 1
            steps_completed = step
            train_losses.append(float(loss.item()))

            if step % 100 == 0 or step == 1 or step == args.steps:
                reports = [rule.report() for rule in rules]
                active = sum(r["active_neurons"] for r in reports)
                available = sum(r["capacity"] for r in reports)

                print(
                    f"Paso {step}/{args.steps} | loss={loss.item():.4f} | "
                    f"activas={active}/{available} | "
                    f"anchura={model.capacity} | profundidad={model.depth} | "
                    f"{source_mix}"
                )

            saved_this_step = False
            if step % EVAL_INTERVAL == 0 and step < args.steps:
                # Valida todos los corpus en cada revisión, no solo el corpus actual.
                current_eval = {name: evaluate(model, data, device) for name, data in eval_sets.items()}
                train_probe = evaluate(model, probe, device)
                current_score, ratios, retention_penalty = macro_validation_score(
                    current_eval,
                    control_state["baseline_eval"],
                    control_state.get("best_domain_losses"),
                    args.retention_tolerance,
                )

                last_train_probe_loss = float(train_probe)
                last_validation_loss = float(current_eval[primary_name])
                validation_history.append(float(current_score))
                train_probe_history.append(float(train_probe))
                control_state["total_evaluations"] = int(control_state.get("total_evaluations", 0)) + 1
                control_state["evaluations_since_growth"] = int(control_state.get("evaluations_since_growth", 0)) + 1

                best_observed = float(control_state.get("best_macro_score", float("inf")))
                best_checkpoint_score = float(control_state.get("best_checkpoint_score", best_observed))
                patience_reference = float(control_state.get("patience_reference_score", best_checkpoint_score))
                significant_improvement = current_score < patience_reference - args.min_improvement

                if current_score < best_observed:
                    control_state["best_macro_score"] = float(current_score)
                    control_state["best_macro_losses"] = {k: float(v) for k, v in current_eval.items()}
                    control_state["best_step"] = int(global_step)

                # Rastrea el mejor loss individual de cada dominio aunque el macro-score
                # no mejore: el control de retención debe reconocer todo progreso útil.
                domain_best = control_state.setdefault("best_domain_losses", {})
                for name, value in current_eval.items():
                    domain_best[name] = min(float(domain_best.get(name, value)), float(value))

                if significant_improvement:
                    control_state["stale_evaluations"] = 0
                    control_state["patience_reference_score"] = float(current_score)
                else:
                    control_state["stale_evaluations"] = int(control_state.get("stale_evaluations", 0)) + 1

                eval_record = {
                    "global_step": int(global_step),
                    "macro_score": float(current_score),
                    "domain_losses": {k: float(v) for k, v in current_eval.items()},
                    "baseline_ratios": {k: float(v) for k, v in ratios.items()},
                    "retention_penalty": float(retention_penalty),
                    "architecture": model.architecture_config(),
                    "stale_evaluations": int(control_state["stale_evaluations"]),
                }
                history = control_state.setdefault("evaluation_history", [])
                history.append(eval_record)
                if len(history) > 250:
                    del history[:-250]

                print(
                    f"[EVALUACION GLOBAL] paso={global_step} | macro={current_score:.6f} "
                    f"| train-probe={train_probe:.5f} | estancamiento={control_state['stale_evaluations']} "
                    f"| penalización-retención={retention_penalty:.5f}"
                )
                for name, value in current_eval.items():
                    print(
                        f"  [CORPUS] {name}: {value:.5f} "
                        f"| referencia={control_state['baseline_eval'][name]:.5f} "
                        f"| ratio={ratios[name]:.4f}"
                    )

                # Actualiza el mejor checkpoint validado si mejora al menos el umbral.
                if current_score < best_checkpoint_score - args.min_improvement:
                    control_state["best_checkpoint_score"] = float(current_score)
                    best_metrics = {
                        "dataset": args.dataset,
                        "replay_dataset": args.replay_dataset,
                        "steps_this_run": step,
                        "global_step": global_step,
                        "autosave_stage": "best_macro_validation",
                        "final_config": model.architecture_config(),
                        "eval_losses": {k: float(v) for k, v in current_eval.items()},
                        "macro_validation_score": float(current_score),
                        "cultivation_control": control_state,
                        "diagnosis": {"growth_events": growth_events},
                    }
                    save_checkpoint(best_path, model, rules, optimizer, global_step, best_metrics)
                    print(f"[MEJOR CHECKPOINT] Validación conjunta mejoró a {current_score:.6f}: {best_path}", flush=True)

                # Evalúa una sola vez cada expansión, y solo después de dar margen
                # suficiente y comprobar que el modelo vuelve a estar estancado.
                last_growth_step = control_state.get("last_growth_global_step")
                growth_trial_due = (
                    last_growth_step is not None
                    and int(control_state.get("evaluations_since_growth", 0)) >= args.post_growth_grace_evals
                    and control_state.get("growth_trial_assessed_for_step") != last_growth_step
                    and len(validation_history) >= STAGNATION_EVALS
                )
                if growth_trial_due:
                    recent_val_gain = (validation_history[-STAGNATION_EVALS] - validation_history[-1]) / max(abs(validation_history[-STAGNATION_EVALS]), 1e-8)
                    recent_train_gain = (train_probe_history[-STAGNATION_EVALS] - train_probe_history[-1]) / max(abs(train_probe_history[-STAGNATION_EVALS]), 1e-8)
                    pre_growth_score = float(control_state.get("last_growth_score", current_score))
                    post_growth_useful = current_score < pre_growth_score - args.min_improvement
                    post_growth_plateau = (recent_val_gain < PLATEAU_TOLERANCE and recent_train_gain < PLATEAU_TOLERANCE)
                    if post_growth_useful:
                        control_state["growth_trials_since_best"] = 0
                        control_state["growth_trial_assessed_for_step"] = last_growth_step
                        print(
                            f"[RESULTADO CRECIMIENTO] La expansión del paso {last_growth_step} "
                            f"aportó mejora útil: score {pre_growth_score:.6f} -> {current_score:.6f}."
                        )
                    elif post_growth_plateau:
                        control_state["growth_trials_since_best"] = int(control_state.get("growth_trials_since_best", 0)) + 1
                        control_state["growth_trial_assessed_for_step"] = last_growth_step
                        print(
                            f"[RESULTADO CRECIMIENTO] No se observa mejora macro útil tras la expansión "
                            f"del paso {last_growth_step}; experimentos estructurales sin beneficio consecutivos: "
                            f"{control_state['growth_trials_since_best']}/{args.max_unproductive_growths if args.max_unproductive_growths else 'sin límite'}."
                        )

                max_failed = args.max_unproductive_growths
                growth_trials_exhausted = (
                    max_failed > 0
                    and int(control_state.get("growth_trials_since_best", 0)) >= max_failed
                )

                if growth_trials_exhausted:
                    control_state["stop_reason"] = (
                        f"Se probaron {control_state['growth_trials_since_best']} expansiones estructurales "
                        "sin mejora útil de validación después del periodo de evaluación. "
                        "El checkpoint más reciente y el mejor checkpoint validado se conservaron."
                    )
                    stop_reason = control_state["stop_reason"]
                    print("[CULTIVO ESTABILIZADO] " + stop_reason, flush=True)
                    grew = False
                    optimizer = optimizer
                else:
                    # El macro-score conjunto decide el estancamiento; una brecha alta por sí
                    # sola no bloquea la exploración si el modelo lleva tiempo estancado.
                    optimizer, grew = consider_growth(
                        model=model,
                        rules=rules,
                        optimizer=optimizer,
                        train_history=train_probe_history,
                        validation_history=validation_history,
                        train_loss=train_probe,
                        val_loss=current_eval[primary_name],
                        max_capacity=args.max_capacity,
                        max_depth=args.max_depth,
                        max_d_model=args.max_d_model,
                        max_heads=args.max_heads,
                        max_context=args.max_context,
                        data_context_ceiling=data_context_ceiling,
                        step=global_step,
                        growth_events=growth_events,
                        evaluations_since_growth=control_state["evaluations_since_growth"],
                        post_growth_grace_evals=args.post_growth_grace_evals,
                    )

                    if grew:
                        validation_history.clear()
                        train_probe_history.clear()
                        control_state["growth_validation_history"] = []
                        control_state["growth_train_history"] = []
                        control_state["evaluations_since_growth"] = 0
                        control_state["last_growth_global_step"] = int(global_step)
                        control_state["last_growth_score"] = float(current_score)
                        control_state["growth_trial_assessed_for_step"] = None
                        control_state["stale_evaluations"] = 0
                        # Se concede paciencia nueva: una ampliación no se considera fallida
                        # hasta que haya tenido tiempo real de entrenarse y validarse.
                        control_state["patience_reference_score"] = float(current_score)

                control_state["growth_validation_history"] = list(validation_history[-16:])
                control_state["growth_train_history"] = list(train_probe_history[-16:])
                control_state["last_architecture"] = model.architecture_config()

                # La parada automática ocurre DESPUÉS de dar al crecimiento la oportunidad
                # de responder al plateau y solo tras la paciencia especificada.
                early_stop = (
                    (args.early_stop_patience_evals > 0
                     and int(control_state["stale_evaluations"]) >= args.early_stop_patience_evals
                     and int(control_state["evaluations_since_growth"]) >= args.post_growth_grace_evals)
                    or growth_trials_exhausted
                )
                if early_stop:
                    if not control_state.get("stop_reason"):
                        control_state["stop_reason"] = (
                            f"Sin mejora útil del score macro durante {control_state['stale_evaluations']} evaluaciones "
                            "y tras dar tiempo suficiente a la última expansión."
                        )
                    stop_reason = control_state["stop_reason"]
                    print("[CULTIVO ESTABILIZADO] " + control_state["stop_reason"], flush=True)

                evaluation_metrics = {
                    "dataset": args.dataset,
                    "replay_dataset": args.replay_dataset,
                    "steps_this_run": step,
                    "global_step": global_step,
                    "autosave": True,
                    "autosave_stage": "evaluation",
                    "autosave_interval": args.autosave_interval,
                    "final_config": model.architecture_config(),
                    "train_probe_loss_at_save": last_train_probe_loss,
                    "validation_loss_at_save": last_validation_loss,
                    "initial_eval": initial_eval,
                    "current_eval": {k: float(v) for k, v in current_eval.items()},
                    "macro_validation_score": float(current_score),
                    "cultivation_control": control_state,
                    "best_checkpoint_path": str(best_path),
                    "diagnosis": {"growth_events": growth_events},
                }
                save_checkpoint(args.out, model, rules, optimizer, global_step, evaluation_metrics)
                save_metrics_json(metrics_path, evaluation_metrics)
                saved_this_step = True
                print(
                    f"[AUTOGUARDADO][EVALUACION] Paso {step}: {Path(args.out).resolve()} "
                    f"(mejor validado: {best_path})",
                    flush=True,
                )

                if early_stop:
                    stop_reason = control_state["stop_reason"]
                    break

            # El historial reciente para juzgar crecimiento se guarda junto al checkpoint.
            if not saved_this_step and step % args.autosave_interval == 0:
                periodic_metrics = {
                    "dataset": args.dataset,
                    "replay_dataset": args.replay_dataset,
                    "steps_this_run": step,
                    "global_step": global_step,
                    "autosave": True,
                    "autosave_stage": "periodic",
                    "autosave_interval": args.autosave_interval,
                    "final_config": model.architecture_config(),
                    "train_probe_loss_at_save": last_train_probe_loss,
                    "validation_loss_at_save": last_validation_loss,
                    "cultivation_control": control_state,
                    "best_checkpoint_path": str(best_path),
                    "diagnosis": {"growth_events": growth_events},
                }
                save_checkpoint(args.out, model, rules, optimizer, global_step, periodic_metrics)
                save_metrics_json(metrics_path, periodic_metrics)
                print(f"[AUTOGUARDADO] Paso {step}/{args.steps}: {Path(args.out).resolve()}", flush=True)

    except KeyboardInterrupt:
        print("\n[INTERRUPCION] Ctrl+C recibido. Guardando el estado actual...", flush=True)
        control_state["growth_validation_history"] = list(validation_history[-16:])
        control_state["growth_train_history"] = list(train_probe_history[-16:])
        control_state["last_architecture"] = model.architecture_config()
        interrupted_metrics = {
            "dataset": args.dataset,
            "replay_dataset": args.replay_dataset,
            "steps_this_run": locals().get("step", 0),
            "global_step": global_step,
            "autosave": True,
            "autosave_stage": "manual_interrupt",
            "autosave_interval": args.autosave_interval,
            "interrupted_by_user": True,
            "final_config": model.architecture_config(),
            "last_training_loss": train_losses[-1] if train_losses else None,
            "train_probe_loss_at_save": last_train_probe_loss,
            "validation_loss_at_save": last_validation_loss,
            "initial_eval": initial_eval,
            "cultivation_control": control_state,
            "best_checkpoint_path": str(best_path),
            "diagnosis": {"growth_events": growth_events},
        }
        try:
            save_checkpoint(
                args.out, model, rules, optimizer, global_step, interrupted_metrics
            )
            save_metrics_json(metrics_path, interrupted_metrics)
            print(
                f"[GUARDADO POR INTERRUPCION] Checkpoint: {Path(args.out).resolve()}",
                flush=True
            )
            print(
                f"[RESPALDO] {Path(args.out).resolve()}.bak",
                flush=True
            )
            print(
                "[DETENIDO] Puedes reanudar usando ese checkpoint con --checkpoint.",
                flush=True
            )
        except Exception as save_exc:
            print(
                "[ERROR DE GUARDADO] No se pudo guardar el estado al interrumpir: "
                f"{type(save_exc).__name__}: {save_exc}",
                flush=True
            )
            print(
                "El checkpoint anterior y su .bak se dejan sin borrar; "
                "revisa las rutas y el espacio en disco.",
                flush=True
            )
        return

    final_train_probe = evaluate(model, probe, device)
    final_eval = {name: evaluate(model, data, device) for name, data in eval_sets.items()}
    final_macro_score, final_ratios, final_retention_penalty = macro_validation_score(
        final_eval, control_state["baseline_eval"], control_state.get("best_domain_losses"), args.retention_tolerance
    )

    for name, value in final_eval.items():
        control_state.setdefault("best_domain_losses", {})[name] = min(
            float(control_state["best_domain_losses"].get(name, value)), float(value)
        )

    if final_macro_score < float(control_state.get("best_macro_score", float("inf"))):
        control_state["best_macro_score"] = float(final_macro_score)
        control_state["best_macro_losses"] = {k: float(v) for k, v in final_eval.items()}
        control_state["best_step"] = int(global_step)

    # La evaluación final también puede convertirse en el mejor checkpoint validado.
    if final_macro_score < float(control_state.get("best_checkpoint_score", float("inf"))):
        control_state["best_checkpoint_score"] = float(final_macro_score)
        final_best_metrics = {
            "dataset": args.dataset,
            "replay_dataset": args.replay_dataset,
            "steps_this_run": steps_completed,
            "global_step": global_step,
            "autosave_stage": "best_macro_validation_final",
            "final_config": model.architecture_config(),
            "eval_losses": {k: float(v) for k, v in final_eval.items()},
            "macro_validation_score": float(final_macro_score),
            "cultivation_control": control_state,
            "diagnosis": {"growth_events": growth_events},
        }
        save_checkpoint(best_path, model, rules, optimizer, global_step, final_best_metrics)
        print(f"[MEJOR CHECKPOINT] Evaluación final guardada: {best_path}", flush=True)

    control_state["final_macro_score"] = float(final_macro_score)
    control_state["final_domain_losses"] = {k: float(v) for k, v in final_eval.items()}
    control_state["final_baseline_ratios"] = {k: float(v) for k, v in final_ratios.items()}
    control_state["final_retention_penalty"] = float(final_retention_penalty)
    control_state["growth_validation_history"] = list(validation_history[-16:])
    control_state["growth_train_history"] = list(train_probe_history[-16:])
    control_state["last_architecture"] = model.architecture_config()
    control_state["best_checkpoint_path"] = str(best_path)
    control_state["stop_reason"] = stop_reason or control_state.get("stop_reason")

    diagnosis = final_diagnosis(
        model=model,
        rules=rules,
        initial_train_loss=initial_train_probe,
        final_train_loss=final_train_probe,
        initial_eval=initial_eval,
        final_eval=final_eval,
        primary_name=primary_name,
        max_capacity=args.max_capacity,
        max_depth=args.max_depth,
        max_d_model=args.max_d_model,
        max_heads=args.max_heads,
        max_context=args.max_context,
        growth_events=growth_events
    )
    diagnosis["final_macro_validation_score"] = float(final_macro_score)
    diagnosis["best_macro_validation_score"] = float(control_state.get("best_macro_score", final_macro_score))
    diagnosis["automatic_stop_reason"] = control_state.get("stop_reason")

    metrics = {
        "dataset": args.dataset,
        "replay_dataset": args.replay_dataset,
        "steps_requested": args.steps,
        "steps_this_run": steps_completed,
        "early_stopped": bool(stop_reason),
        "stop_reason": stop_reason,
        "global_step": global_step,
        "max_capacity": args.max_capacity,
        "max_depth": args.max_depth,
        "max_d_model": args.max_d_model,
        "max_heads": args.max_heads,
        "max_context": args.max_context,
        "final_config": model.architecture_config(),
        "initial_train_probe_loss": initial_train_probe,
        "final_train_probe_loss": final_train_probe,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "final_macro_validation_score": float(final_macro_score),
        "best_checkpoint_path": str(best_path),
        "cultivation_control": control_state,
        "diagnosis": diagnosis,
    }

    save_checkpoint(args.out, model, rules, optimizer, global_step, metrics)
    save_metrics_json(metrics_path, metrics)

    if stop_reason:
        print("\n[CULTIVO ESTABILIZADO] " + stop_reason, flush=True)
    else:
        print(
            "\n[FIN DE EJECUCION] Se alcanzaron los pasos solicitados. "
            "Esto no demuestra por sí solo que el modelo esté convergido.",
            flush=True,
        )
    print(f"[OK] Checkpoint de última estructura: {Path(args.out).resolve()}")
    print(f"[OK] Mejor checkpoint validado: {best_path}")
    print(f"[OK] Respaldo del checkpoint final: {Path(args.out).resolve()}.bak")
    print(f"[OK] Diagnóstico: {metrics_path.resolve()}")


# ============================================================
# 9. CONSOLA INDEPENDIENTE
# ============================================================
@torch.no_grad()
def chat(args):
    torch.set_num_threads(2)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ckpt = load_checkpoint(args.checkpoint, device)
    config = ckpt["config"]

    model = MiniLM(
        capacity=int(config["capacity"]),
        depth=int(config["depth"]),
        d_model=int(config["d_model"]),
        heads=int(config["heads"]),
        context=int(config["context"])
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"\nMini-GPT: {args.checkpoint}")
    print(f"Bloques: {model.depth}; anchura: {model.capacity}")
    print(f"Contexto: {model.context} bytes")
    print("Consola de inferencia: no modifica los pesos.")
    print("Escribe /limpiar o /salir.\n")

    history = b""

    while True:
        try:
            message = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if message.lower() in {"/salir", "/exit", "/quit"}:
            break

        if message.lower() == "/limpiar":
            history = b""
            print("[Contexto borrado.]")
            continue

        prompt = history + (
            f"Usuario: {message}\nAsistente:"
        ).encode("utf-8")

        generated = bytearray()

        for _ in range(args.max_new_bytes):
            current = (prompt + bytes(generated))[-model.context:]

            x = torch.tensor(
                [list(current)],
                dtype=torch.long,
                device=device
            )

            logits, _ = model(x)
            scores = logits[0, -1, :] / args.temperature

            k = min(args.top_k, scores.numel())
            values, indices = torch.topk(scores, k)

            probs = torch.softmax(values, dim=-1)
            selected = torch.multinomial(probs, 1)
            generated.append(int(indices[selected].item()))

            if b"\nUsuario:" in generated:
                del generated[generated.find(b"\nUsuario:"):]
                break

        response = bytes(generated)
        print("Modelo:", response.decode("utf-8", errors="replace"), "\n")

        history += (
            f"Usuario: {message}\nAsistente:".encode("utf-8")
            + response + b"\n"
        )

        history = history[-model.context * 3:]


# ============================================================
# 10. INTERFAZ DE COMANDOS
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sovereign Growing Mini-GPT: cultivo adaptativo de FFN, "
            "profundidad, dimensión/cabezas de atención y contexto"
        )
    )

    parser.add_argument("--dataset")
    parser.add_argument("--replay-dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", default="checkpoints/etapa1.pt")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Número de pasos de entrenamiento (entero positivo).")
    parser.add_argument(
        "--autosave-interval", type=int, default=100,
        help="Guardar checkpoint cada N pasos. Predeterminado: 100."
    )
    parser.add_argument(
        "--early-stop-patience-evals", type=int, default=30,
        help="Parar tras N evaluaciones sin mejora útil conjunta. 0 desactiva la parada automática."
    )
    parser.add_argument(
        "--post-growth-grace-evals", type=int, default=8,
        help="Evaluaciones mínimas para medir una expansión antes de otra expansión o parada."
    )
    parser.add_argument(
        "--min-improvement", type=float, default=0.001,
        help="Mejora relativa útil del score conjunto (0.001 = 0,1 por ciento)."
    )
    parser.add_argument(
        "--retention-tolerance", type=float, default=0.10,
        help="Regresión por corpus tolerada frente a su mejor pérdida antes de penalizar (0.10 = 10 por ciento)."
    )
    parser.add_argument(
        "--max-unproductive-growths", type=int, default=6,
        help="Parar si N expansiones seguidas no aportan mejora útil tras ser evaluadas. 0 desactiva este control."
    )

    # Sin choices predefinidos: puede indicarse cualquier entero positivo.
    # Estos parámetros solo se usan al crear un modelo nuevo, no al reanudar.
    parser.add_argument("--capacity", type=int, default=None,
                        help="Anchura FFN inicial. Predeterminado: 32.")
    parser.add_argument("--depth", type=int, default=None,
                        help="Profundidad inicial. Predeterminado: 1 bloque.")
    parser.add_argument("--d-model", type=int, default=None,
                        help="Dimensión interna inicial. Predeterminado: 32.")
    parser.add_argument("--heads", type=int, default=None,
                        help="Cabezas de atención iniciales. Predeterminado: 2.")
    parser.add_argument("--context", type=int, default=None,
                        help="Longitud inicial de contexto en bytes. Predeterminado: 256.")

    # Todos los topes son opcionales. Si se omiten, el código no impone un máximo.
    parser.add_argument("--max-capacity", type=int, default=None,
                        help="Límite opcional de anchura FFN automática.")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Límite opcional de profundidad automática.")
    parser.add_argument("--max-d-model", type=int, default=None,
                        help="Límite opcional de dimensión interna.")
    parser.add_argument("--max-heads", type=int, default=None,
                        help="Límite opcional de cabezas de atención.")
    parser.add_argument("--max-context", type=int, default=None,
                        help="Límite opcional de contexto en bytes.")

    parser.add_argument(
        "--eval-dataset", action="append", default=[]
    )

    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--max-new-bytes", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)

    args = parser.parse_args()

    if args.chat:
        if not args.checkpoint:
            parser.error("--chat necesita --checkpoint.")
        if (
            args.max_new_bytes < 1
            or args.temperature <= 0
            or args.top_k < 1
        ):
            parser.error("Parámetros de generación inválidos.")
        chat(args)
        return

    if not args.dataset:
        parser.error("El entrenamiento necesita --dataset.")

    if args.steps < 1:
        parser.error("--steps debe ser positivo.")
    if args.autosave_interval < 1:
        parser.error("--autosave-interval debe ser positivo.")
    if args.early_stop_patience_evals < 0:
        parser.error("--early-stop-patience-evals no puede ser negativo.")
    if args.post_growth_grace_evals < 1:
        parser.error("--post-growth-grace-evals debe ser al menos 1.")
    if args.min_improvement < 0:
        parser.error("--min-improvement no puede ser negativo.")
    if args.retention_tolerance < 0:
        parser.error("--retention-tolerance no puede ser negativo.")
    if args.max_unproductive_growths < 0:
        parser.error("--max-unproductive-growths no puede ser negativo.")
    if args.capacity is not None and args.capacity < 1:
        parser.error("--capacity debe ser positivo.")
    if args.depth is not None and args.depth < 1:
        parser.error("--depth debe ser positivo.")
    if args.max_capacity is not None and args.max_capacity < 1:
        parser.error("--max-capacity debe ser positivo.")
    if args.max_depth is not None and args.max_depth < 1:
        parser.error("--max-depth debe ser positivo.")
    if args.d_model is not None and args.d_model < 1:
        parser.error("--d-model debe ser positivo.")
    if args.heads is not None and args.heads < 1:
        parser.error("--heads debe ser positivo.")
    if args.context is not None and args.context < 1:
        parser.error("--context debe ser positivo.")
    if args.max_d_model is not None and args.max_d_model < 1:
        parser.error("--max-d-model debe ser positivo.")
    if args.max_heads is not None and args.max_heads < 1:
        parser.error("--max-heads debe ser positivo.")
    if args.max_context is not None and args.max_context < 1:
        parser.error("--max-context debe ser positivo.")

    init_d = args.d_model if args.d_model is not None else D_MODEL
    init_h = args.heads if args.heads is not None else HEADS
    if init_d % init_h != 0:
        parser.error("--d-model debe ser divisible por --heads.")

    if (args.max_capacity is not None and args.capacity is not None
            and args.capacity > args.max_capacity):
        parser.error("--capacity no puede superar --max-capacity.")
    if (args.max_depth is not None and args.depth is not None
            and args.depth > args.max_depth):
        parser.error("--depth no puede superar --max-depth.")
    if (args.max_d_model is not None and args.d_model is not None
            and args.d_model > args.max_d_model):
        parser.error("--d-model no puede superar --max-d-model.")
    if (args.max_heads is not None and args.heads is not None
            and args.heads > args.max_heads):
        parser.error("--heads no puede superar --max-heads.")
    if (args.max_context is not None and args.context is not None
            and args.context > args.max_context):
        parser.error("--context no puede superar --max-context.")

    if args.replay_dataset and not args.checkpoint:
        parser.error("--replay-dataset requiere --checkpoint.")

    train(args)


if __name__ == "__main__":
    main()
