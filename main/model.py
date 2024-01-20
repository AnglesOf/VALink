import torch.nn as nn
import torch


class BTModel(nn.Module):
    def __init__(self, textEncoder, codeEncoder, text_hidden_size, code_hidden_size, num_class):
        super(BTModel, self).__init__()
        self.textEncoder = textEncoder
        self.codeEncoder = codeEncoder

        self.text_hidden_size = text_hidden_size
        self.code_hidden_size = code_hidden_size
        self.num_class = num_class
        for param in self.textEncoder.parameters():
            param.requires_grad = True
        for param in self.codeEncoder.parameters():
            param.requires_grad = True
        fc_input_size = code_hidden_size + text_hidden_size
        self.fc = nn.Linear(fc_input_size, int(fc_input_size / 2))
        self.fc1 = nn.Linear(int(fc_input_size / 2),
                             int(fc_input_size / 4))
        self.fc2 = nn.Linear(int(fc_input_size / 4), int(fc_input_size / 6))
        self.fc3 = nn.Linear(int(fc_input_size / 6), num_class)
        self.relu = nn.ReLU()

    def forward(self, text_input_ids=None, code_input_ids=None, text_inputs_embeds=None, code_inputs_embeds=None):
        if text_input_ids is not None and code_input_ids is not None and text_inputs_embeds is not None and code_inputs_embeds is not None:
            mask = text_input_ids.ne(1).int()  # pad_token_id=1, past_key_values_length=0
            text_position_ids = ((torch.cumsum(mask, dim=1).type_as(mask) + 0) * mask).long() + 1
            mask2 = code_input_ids.ne(0).int()  # pad_token_id=0, past_key_values_length=0
            code_position_ids = ((torch.cumsum(mask2, dim=1).type_as(mask2) + 0) * mask2).long() + 0
            text_output = self.textEncoder(position_ids=text_position_ids, inputs_embeds=text_inputs_embeds, attention_mask=text_input_ids.ne(1))[1]
            code_output = self.codeEncoder(position_ids=code_position_ids, inputs_embeds=code_inputs_embeds, attention_mask=code_input_ids.ne(1))[1]
        elif text_input_ids is not None and code_input_ids is not None and text_inputs_embeds is None and code_inputs_embeds is None:
            text_output = self.textEncoder(input_ids=text_input_ids, attention_mask=text_input_ids.ne(1))[1]  # [batch_size, hiddensize]
            code_output = self.codeEncoder(input_ids=code_input_ids, attention_mask=code_input_ids.ne(1))[1]  # [batch_size, hiddensize]
        else:
            raise ValueError("value is not complete")
        logits = self.fc(torch.cat([code_output, text_output], dim=-1) * 10)
        del code_output
        del text_output
        logits = self.fc1(logits)
        logits = self.fc2(logits)
        logits = self.fc3(logits)
        return logits, text_input_ids, code_input_ids

    def get_embeddings(self, text_input_ids, code_input_ids):
        return self.textEncoder.get_input_embeddings()(text_input_ids), self.codeEncoder.get_input_embeddings()(code_input_ids)
