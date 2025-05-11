import numpy as np
import pickle
from IPython.display import display
import torch
import time
import abc
from typing import Optional, Union, Tuple, List, Callable, Dict, Any
from collections import defaultdict


# Базовый абстрактный класс контроллера
class VectorControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, vector, place_in_unet: str):
        vector = self.forward(vector, place_in_unet)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()
            self.cur_step += 1
        return vector


class VectorStore(VectorControl):
    def __init__(
        self, 
        steering_vectors=None, 
        steer=True, 
        steer_only_up=False, 
        alpha=10, 
        beta=2, 
        steer_back=False, 
        device='cpu'
    ):
        super(VectorStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.vector_store = defaultdict(dict)
        self.steering_vectors = steering_vectors
        self.steer = steer
        self.steer_only_up = steer_only_up
        self.alpha = alpha
        self.beta = beta
        self.steer_back = steer_back
        self.device = device

    def reset(self):
        super(VectorStore, self).reset()
        self.step_store = self.get_empty_store()
        self.vector_store = defaultdict(dict)
        
    @staticmethod
    def get_empty_store():
        return {"down": [], "up": [], 'mid': []}

    def forward(self, vector, place_in_unet: str):
        # steering 
        if self.steer:
            if place_in_unet in ['up', 'mid'] or (place_in_unet == 'down' and not self.steer_only_up): 
                # Безопасное получение индекса шага
                max_step = max(self.steering_vectors.keys())
                num_steer = 0 if len(list(self.steering_vectors.keys()))==1 else min(self.cur_step, max_step)

                # Безопасное получение индекса слоя
                layer_idx = len(self.step_store[place_in_unet])
                if layer_idx >= len(self.steering_vectors[num_steer][place_in_unet]):
                    layer_idx = len(self.steering_vectors[num_steer][place_in_unet]) - 1

                steering_vector = self.steering_vectors[num_steer][place_in_unet][layer_idx]
                steering_vector = torch.tensor(steering_vector).to(self.device).view(1, 1, -1)

                # Сохраняем текущую норму компонентов вектора
                norm = torch.norm(vector, dim=2, keepdim=True)

                # Вычисляем скалярное произведение (similarity) между векторами
                sim = torch.tensordot(
                    vector, 
                    steering_vector,
                    dims=([2], [2])
                ).view(vector.size()[0], vector.size()[1], 1)

                if self.steer_back:
                    # Управление назад (удаление понятия из вектора)
                    # Используем только положительное скалярное произведение
                    sim_pos = torch.where(sim > 0, sim, 0)
                    vector = vector - (self.beta * sim_pos) * steering_vector.expand(1, vector.size()[1], -1)
                else:
                    # Стандартный вариант без скалярного произведения как множителя
                    vector = vector + (self.alpha) * steering_vector.expand(1, vector.size()[1], -1)

                # Ренормализация
                vector = vector / torch.norm(vector, dim=2, keepdim=True)
                vector = vector * norm

        # Сохраняем активацию для дальнейшего вычисления векторов управления
        self.step_store[place_in_unet].append(vector.data.cpu().numpy()[len(vector)//2:].mean(axis=0).mean(axis=0))

        return vector

    def between_steps(self):
        self.vector_store[self.cur_step] = self.step_store
        self.step_store = self.get_empty_store()


# Функция регистрации контроллера для модели Unet
def register_vector_control(model, controller):
    def block_forward(self, place_in_unet):
        def forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.Tensor:
            if cross_attention_kwargs is not None:
                if cross_attention_kwargs.get("scale", None) is not None:
                    pass  # logger warning removed
    
            # 0. Self-Attention
            batch_size = hidden_states.shape[0]
    
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            else:
                raise ValueError("Incorrect norm used")
    
            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # 1. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
    
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
        
            y = torch.norm(attn_output, dim=2, keepdim=True) 
    
            if self.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output
    
            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
    
            # 1.2 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
    
            # 3. Cross-Attention
            if self.attn2 is not None:
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.norm_type == "ada_norm_single":
                    norm_hidden_states = hidden_states
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")
    
                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)
    
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                
                # Добавляем контроллер
                attn_output = controller(attn_output, place_in_unet)
                
                # Отключаем специфичную логику, которая может мешать стилизации
                # if controller.cur_att_layer in list(range(24, 35)):
                #     attn_output = attn_output*0
                    
                hidden_states = attn_output + hidden_states

            # 4. Feed-forward
            if self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm3(hidden_states)
    
            if self.norm_type == "ada_norm_zero":
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    
            if self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
    
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)
    
            if self.norm_type == "ada_norm_zero":
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.norm_type == "ada_norm_single":
                ff_output = gate_mlp * ff_output
    
            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
                
            # Отладочная информация (отключена)
            if False:  # Для отключения вывода
                print(controller.cur_att_layer-1)
                x = torch.norm(attn_output, dim=2, keepdim=True) / torch.norm(hidden_states, dim=2, keepdim=True)
                print('CA', place_in_unet, x.mean().item())
                
                x = y / torch.norm(hidden_states, dim=2, keepdim=True)
                print('SA',place_in_unet, x.mean().item())
                
                x = torch.norm(ff_output, dim=2, keepdim=True) / torch.norm(hidden_states, dim=2, keepdim=True)
                print('FF',place_in_unet, x.mean().item())
                print()

            return hidden_states
        return forward

    def register_recr(net_, count, place_in_unet):
        '''
        Рекурсивная регистрация контроллера для всех BasicTransformerBlock в модели
        '''
        if net_.__class__.__name__ == 'BasicTransformerBlock':
            net_.forward = block_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    block_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            block_count += register_recr(net[1], 0, "down")
            print('down', block_count)
        elif "up" in net[0]:
            block_count += register_recr(net[1], 0, "up")
            print('up', block_count)
        if "mid" in net[0]:
            block_count += register_recr(net[1], 0, "mid")
            print('mid', block_count)
    controller.num_att_layers = block_count