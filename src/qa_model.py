import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel, AutoModel
from transformers import RobertaForSequenceClassification


class RoBertaClassifier(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2
        )

    def forward(self, input_ids, attention_mask, answer=None):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits

        if answer is not None:
            # pred = logits.view(-1)
            # answer = answer.view(-1)
            loss = F.cross_entropy(logits, answer)
            return torch.argmax(logits, axis=1), loss

        return torch.argmax(logits, axis=1)


# class RoBertaClassifier(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.roberta = AutoModel.from_pretrained("roberta-base")
#         D_in, hidden_dim, D_out = 768, configs["hidden_dim"], 1

#         hidden_layers = []
#         hidden_layers.append(nn.Linear(D_in, hidden_dim))
#         for _ in range(configs["n_cls_layers"]):
#             hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
#             hidden_layers.append(nn.ReLU())
#             hidden_layers.append(nn.Dropout(configs["dropout"]))
#         hidden_layers.append(nn.Linear(hidden_dim, D_out))
#         self.classifier = nn.Sequential(*hidden_layers)

#         if configs["freeze_bert"]:
#             # for layer in self.roberta.encoder.layer[:23]:
#             #     for param in layer.parameters():
#             #         param.requires_grad = False
#             for param in self.roberta.parameters():
#                 param.requires_grad = False

#     def forward(self, input_ids, attention_mask, answer=None):
#         outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state_cls = outputs[0][:, 0, :]
#         logits = self.classifier(last_hidden_state_cls)
#         logits = torch.sigmoid(logits)

#         if answer is not None:
#             pred = logits.reshape(-1)
#             answer = answer.reshape(-1)
#             loss = F.binary_cross_entropy(pred, answer)
#             return torch.round(logits), loss

#         return torch.round(logits)