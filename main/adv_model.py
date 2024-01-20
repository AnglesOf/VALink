import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BTModel


class SmoothL1Loss_self(nn.Module):
    def __init__(self):
        super(SmoothL1Loss_self, self).__init__()
        self.MSELoss = nn.SmoothL1Loss()

    def forward(self, logits, labels):
        one_hot_targets = np.zeros((labels.shape[0], 2))
        for i, target in enumerate(labels):
            one_hot_targets[i, target] = 1
        one_hot_targets = torch.tensor(one_hot_targets).to(logits.device)
        loss = self.MSELoss(logits, one_hot_targets)
        return loss


class Center_loss(nn.Module):
    def __init__(self):
        super(Center_loss, self).__init__()
        self.MSELoss = SmoothL1Loss_self()

    def forward(self, logits, target):
        mseLoss = self.MSELoss(logits, target)
        return mseLoss


class AdaptiveLoss(nn.Module):
    def __init__(self):
        super(AdaptiveLoss, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.Center_loss = Center_loss()
        self.lossCLS_pre = -100
        self.lossCE_pre = -100

    def forward(self, logits, labels, eps=1e-6, alph=1):
        lossCLS = self.CrossEntropyLoss(logits, labels)
        lossCE = self.Center_loss(logits, labels)
        if self.lossCLS_pre == -100:
            rate_CLS = 2
            rate_CE = 2
        else:
            rate_CLS = abs((lossCLS - self.lossCLS_pre) + eps) / abs((self.lossCLS_pre) + eps)
            rate_CE = abs((lossCE - self.lossCE_pre) + eps) / abs((self.lossCE_pre) + eps)
        self.lossCLS_pre = lossCLS
        self.lossCE_pre = lossCE

        weight_CLS = math.log2(rate_CLS + 1) / (math.log2(rate_CLS + 1) + math.log2(rate_CE + 1))
        weight_CE = math.log2(rate_CE + 1) / (math.log2(rate_CLS + 1) + math.log2(rate_CE + 1))
        T = (weight_CLS + eps) / (weight_CE + eps) if weight_CLS > weight_CE else (weight_CE + eps) / (weight_CLS + eps)
        weight_CLS = math.log2(rate_CLS / T + 1) / (math.log2(rate_CLS / T + 1) + math.log2(rate_CE / T + 1)) * alph
        weight_CE = math.log2(rate_CE / T + 1) / (math.log2(rate_CLS / T + 1) + math.log2(rate_CE / T + 1)) * alph

        loss = weight_CLS * lossCLS + weight_CE * lossCE
        return lossCE, lossCLS, lossCE, weight_CLS, weight_CE


class ADVModel(nn.Module):
    def __init__(self, textEncoder, codeEncoder, text_hidden_size, code_hidden_size,
                 num_class):
        super(ADVModel, self).__init__()
        self.model = BTModel(textEncoder, codeEncoder, text_hidden_size, code_hidden_size, num_class)
        self.lossF = AdaptiveLoss()

    def kl(self, inputs, targets, reduction="batchmean"):
        return F.kl_div(F.log_softmax(inputs, dim=-1),
                        F.softmax(targets, dim=-1),
                        reduction=reduction)

    def adv_project(self, grad, norm_type='inf', eps=1e-6):
        if norm_type == 'l2':
            return grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            return grad.sign()
        else:
            return grad / (grad.abs().max(-1, keepdim=True)[0] + eps)

    def forward(self, text_input_ids=None, code_input_ids=None, labels=None, adv_flag=True):
        logits, text_input_ids, code_input_ids = self.model(text_input_ids, code_input_ids)
        if not adv_flag:
            loss, lossCLS, lossCE, weight_CLS, weight_CE = self.lossF(logits, labels)
            return loss, torch.softmax(logits, -1), lossCLS, lossCE, torch.Tensor([weight_CLS]).to(
                loss.device), torch.Tensor([weight_CE]).to(loss.device)
        else:
            text_embed, code_embed = self.model.get_embeddings(
                text_input_ids.clone().detach(), code_input_ids.clone().detach())
            text_noise = torch.Tensor(text_embed.shape).normal_(0, 1) * 1e-5  # default 1e-5
            code_noise = torch.Tensor(code_embed.shape).normal_(0, 1) * 1e-5  # default 1e-5
            text_noise = text_noise.to(logits.device)
            code_noise = code_noise.to(logits.device)
            text_noise.requires_grad_()
            code_noise.requires_grad_()

            text_embed = torch.tensor(text_embed.tolist())
            code_embed = torch.tensor(code_embed.tolist())
            text_embed = text_embed.to(logits.device)
            code_embed = code_embed.to(logits.device)
            text_embed = text_embed + text_noise
            code_embed = code_embed + code_noise
            adv_logits, text_input_ids, code_input_ids = self.model(text_input_ids=text_input_ids,
                                                                    code_input_ids=code_input_ids,
                                                                    text_inputs_embeds=text_embed,
                                                                    code_inputs_embeds=code_embed)

            adv_loss = self.kl(adv_logits, logits.detach(), reduction="batchmean")
            del adv_logits
            text_delta_grad, = torch.autograd.grad(adv_loss, text_noise, only_inputs=True, retain_graph=True)
            code_delta_grad, = torch.autograd.grad(adv_loss, code_noise, only_inputs=True)
            del adv_loss
            text_delta_grad = text_delta_grad.norm()
            code_delta_grad = code_delta_grad.norm()
            torch.cuda.empty_cache()
            if torch.isnan(text_delta_grad) or torch.isinf(text_delta_grad):
                return None
            if torch.isnan(code_delta_grad) or torch.isinf(code_delta_grad):
                return None

            # line 6 inner sum
            text_noise = text_noise + text_delta_grad * 1e-3  # default 1e-3
            del text_delta_grad
            code_noise = code_noise + code_delta_grad * 1e-3  # default 1e-3
            del code_delta_grad
            # line 6 projection
            text_noise = self.adv_project(text_noise, norm_type='l2', eps=1e-6)  # default 1e-6
            code_noise = self.adv_project(code_noise, norm_type='l2', eps=1e-6)  # default 1e-6
            text_noise = torch.tensor(text_noise.tolist())
            code_noise = torch.tensor(code_noise.tolist())
            text_noise = text_noise.to(logits.device)
            text_embed = torch.tensor(text_embed.tolist())
            code_embed = torch.tensor(code_embed.tolist())
            text_embed = text_embed.to(logits.device)
            code_embed = code_embed.to(logits.device)
            text_embed = text_embed + text_noise
            code_embed = code_embed + code_noise

            del text_noise
            del code_noise
            torch.cuda.empty_cache()
            adv_logits, text_input_ids, code_input_ids = self.model(text_input_ids=text_input_ids,
                                                                    code_input_ids=code_input_ids,
                                                                    text_inputs_embeds=text_embed,
                                                                    code_inputs_embeds=code_embed)
            del text_input_ids
            del code_input_ids
            del text_embed
            del code_embed
            adv_loss = self.kl(adv_logits, logits.detach()) + self.kl(logits, adv_logits.detach())
            return adv_loss, torch.softmax(logits, -1)
