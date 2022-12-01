import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel
import pdb

class my_BertsoftmaxNER(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    def __init__(self, config):
        super(my_BertsoftmaxNER, self).__init__(config)
        self.num_labels = config.num_labels
    
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
        self.init_weights()

    def forward(self, input_ids, src_probs=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, loss_ignore_index=-100):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        if src_probs is not None:
            # ## KL Divergence
            # loss_KD_fct = KLDivLoss(reduction="mean")
            # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_log_probs = log_probs.view(-1, self.num_labels)[active_loss]
            #     active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]
            #
            #     loss_KD = loss_KD_fct(active_log_probs, active_src_probs)
            # else:
            #     loss_KD = loss_KD_fct(log_probs, src_probs)

            # ## CrossEntropy
            # loss_KD_fct = CrossEntropyLoss()
            # src_labels = torch.argmax(src_probs.view(-1, self.num_labels), dim=-1)
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_logits = logits.view(-1, self.num_labels)[active_loss]
            #     active_src_labels = src_labels[active_loss]
            #
            #     loss_KD = loss_KD_fct(active_logits, active_src_labels)
            # else:
            #     loss_KD = loss_KD_fct(logits.view(-1, self.num_labels), src_labels)

            ## L2 Norm
            loss_KD_fct = MSELoss(reduction="mean")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                inactive_subword = labels.view(-1) == loss_ignore_index
                active_loss[inactive_subword] = 0
                active_probs = probs.view(-1, self.num_labels)[active_loss]
                active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]

                loss_KD = loss_KD_fct(active_probs, active_src_probs)
            else:
                loss_KD = loss_KD_fct(probs, src_probs)

            outputs = (loss_KD,) + outputs

        return outputs  # (loss_KD), (loss), scores, (hidden_states), (attentions)






class my_BertsoftmaxNER_12class(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    def __init__(self, config):
        super(my_BertsoftmaxNER_12class, self).__init__(config)
        self.num_labels = config.num_labels
    
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.linears = nn.ModuleList([nn.Linear(config.hidden_size, config.num_labels) for i in range(8)])

        self.omega=nn.Parameter(torch.Tensor(8,1))
    
        self.init_weights()

        nn.init.uniform_(self.omega,-1.0,1.0)

    def forward(self, input_ids, src_probs=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, loss_ignore_index=-100, weight=0.1):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        all_hidden_states_train = outputs[2][4:12]
        layer_logits = ()
        for hs, layer_linear in zip(all_hidden_states_train,self.linears):
            layer_logits = layer_logits + (layer_linear(self.dropout(hs)).to(input_ids.device),)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + (layer_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss_12 = loss_fct(active_logits, active_labels)
                loss_8 = None
                for i in range(8):
                  loss_layer = loss_fct(layer_logits[i].view(-1, self.num_labels)[active_loss], active_labels).unsqueeze(0)
                  loss_8 = loss_layer if loss_8 is None else torch.cat((loss_8,loss_layer),dim=0)
                loss = loss_12 + torch.matmul(loss_8,F.softmax(self.omega,dim=0)).squeeze(0)*weight
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        if src_probs is not None:
            # ## KL Divergence
            # loss_KD_fct = KLDivLoss(reduction="mean")
            # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_log_probs = log_probs.view(-1, self.num_labels)[active_loss]
            #     active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]
            #
            #     loss_KD = loss_KD_fct(active_log_probs, active_src_probs)
            # else:
            #     loss_KD = loss_KD_fct(log_probs, src_probs)

            # ## CrossEntropy
            # loss_KD_fct = CrossEntropyLoss()
            # src_labels = torch.argmax(src_probs.view(-1, self.num_labels), dim=-1)
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_logits = logits.view(-1, self.num_labels)[active_loss]
            #     active_src_labels = src_labels[active_loss]
            #
            #     loss_KD = loss_KD_fct(active_logits, active_src_labels)
            # else:
            #     loss_KD = loss_KD_fct(logits.view(-1, self.num_labels), src_labels)

            ## L2 Norm
            loss_KD_fct = MSELoss(reduction="mean")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                inactive_subword = labels.view(-1) == loss_ignore_index
                active_loss[inactive_subword] = 0
                active_probs = probs.view(-1, self.num_labels)[active_loss]
                active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]

                loss_KD = loss_KD_fct(active_probs, active_src_probs)
            else:
                loss_KD = loss_KD_fct(probs, src_probs)

            outputs = (loss_KD,) + outputs

        return outputs  # (loss_KD), (loss), scores, (hidden_states), (attentions)


class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class my_BertsoftmaxNER_MMD(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    def __init__(self, config):
        super(my_BertsoftmaxNER_MMD, self).__init__(config)
        self.num_labels = config.num_labels
    
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
        self.init_weights()

    def forward(self, input_ids, src_probs=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, loss_ignore_index=-100,
                input_ids_other=None, attention_mask_other=None,token_type_ids_other=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        cls_noen = None
        if input_ids_other is not None:
            outputs_noen = self.bert(input_ids_other,
                            attention_mask=attention_mask_other,
                            token_type_ids=token_type_ids_other
                            )
            cls_noen = self.dropout(outputs_noen[0])[:,0]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if cls_noen is not None:
                MMD = MMDLoss()
                loss_mmd = MMD(sequence_output[:,0],cls_noen)
                
                loss += loss_mmd/20
                
            outputs = (loss,) + outputs

        if src_probs is not None:
            # ## KL Divergence
            # loss_KD_fct = KLDivLoss(reduction="mean")
            # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_log_probs = log_probs.view(-1, self.num_labels)[active_loss]
            #     active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]
            #
            #     loss_KD = loss_KD_fct(active_log_probs, active_src_probs)
            # else:
            #     loss_KD = loss_KD_fct(log_probs, src_probs)

            # ## CrossEntropy
            # loss_KD_fct = CrossEntropyLoss()
            # src_labels = torch.argmax(src_probs.view(-1, self.num_labels), dim=-1)
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_logits = logits.view(-1, self.num_labels)[active_loss]
            #     active_src_labels = src_labels[active_loss]
            #
            #     loss_KD = loss_KD_fct(active_logits, active_src_labels)
            # else:
            #     loss_KD = loss_KD_fct(logits.view(-1, self.num_labels), src_labels)

            ## L2 Norm
            loss_KD_fct = MSELoss(reduction="mean")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                inactive_subword = labels.view(-1) == loss_ignore_index
                active_loss[inactive_subword] = 0
                active_probs = probs.view(-1, self.num_labels)[active_loss]
                active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]

                loss_KD = loss_KD_fct(active_probs, active_src_probs)
            else:
                loss_KD = loss_KD_fct(probs, src_probs)

            outputs = (loss_KD,) + outputs

        return outputs  # (loss_KD), (loss), scores, (hidden_states), (attentions)




class my_BertsoftmaxNER_MMD_KD(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    def __init__(self, config):
        super(my_BertsoftmaxNER_MMD_KD, self).__init__(config)
        self.num_labels = config.num_labels
    
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
        self.init_weights()

    def forward(self, input_ids, src_probs=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, loss_ignore_index=-100,
                input_ids_other=None, attention_mask_other=None,token_type_ids_other=None, tea_cls_en=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        cls_other = None
        if input_ids_other is not None:
            outputs_noen = self.bert(input_ids_other,
                            attention_mask=attention_mask_other,
                            token_type_ids=token_type_ids_other
                            )
            cls_other = self.dropout(outputs_noen[0])[:,0]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if (cls_other is not None) and (tea_cls_en is None):
                MMD = MMDLoss()
                loss_mmd = MMD(sequence_output[:,0],cls_other)
                
                loss += loss_mmd/20
                
            outputs = (loss,) + outputs
        

        if src_probs is not None:


            ## L2 Norm
            loss_KD_fct = MSELoss(reduction="mean")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                inactive_subword = labels.view(-1) == loss_ignore_index
                active_loss[inactive_subword] = 0
                active_probs = probs.view(-1, self.num_labels)[active_loss]
                active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]

                loss_KD = loss_KD_fct(active_probs, active_src_probs)
            else:
                loss_KD = loss_KD_fct(probs, src_probs)

            if tea_cls_en is not None:
                MMD = MMDLoss()
                mmd_loss_en_other = MMD(tea_cls_en, sequence_output[:,0])
                mmd_loss_en_en = MMD(tea_cls_en, cls_other)

                loss_KD += mmd_loss_en_other/1000 + mmd_loss_en_en/1000
                
            outputs = (loss_KD,) + outputs

        return outputs  # (loss_KD), (loss), scores, (hidden_states), (attentions)



class my_BertsoftmaxNER_unit(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    def __init__(self, config):
        super(my_BertsoftmaxNER_unit, self).__init__(config)
        self.num_labels = config.num_labels
    
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.linears = nn.ModuleList([nn.Linear(config.hidden_size, config.num_labels) for i in range(8)])

        self.omega=nn.Parameter(torch.Tensor(8,1))
    
        self.init_weights()

        nn.init.uniform_(self.omega,-1.0,1.0)

    def forward(self, input_ids, src_probs=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, loss_ignore_index=-100,
                input_ids_other=None, attention_mask_other=None,token_type_ids_other=None, tea_cls_en=None,weight=0.1):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        all_hidden_states_train = outputs[2][4:12]
        layer_logits = ()
        for hs, layer_linear in zip(all_hidden_states_train,self.linears):
            layer_logits = layer_logits + (layer_linear(self.dropout(hs)).to(input_ids.device),)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + (layer_logits,) + outputs[2:]  # add hidden states and attention if they are here

        cls_other = None
        if input_ids_other is not None:
            outputs_noen = self.bert(input_ids_other,
                            attention_mask=attention_mask_other,
                            token_type_ids=token_type_ids_other
                            )
            cls_other = self.dropout(outputs_noen[0])[:,0]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss_12 = loss_fct(active_logits, active_labels)
                loss_8 = None
                for i in range(8):
                  loss_layer = loss_fct(layer_logits[i].view(-1, self.num_labels)[active_loss], active_labels).unsqueeze(0)
                  loss_8 = loss_layer if loss_8 is None else torch.cat((loss_8,loss_layer),dim=0)
                loss = loss_12 + torch.matmul(loss_8,F.softmax(self.omega,dim=0)).squeeze(0)*weight
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # if (cls_other is not None) and (tea_cls_en is None):
            #     MMD = MMDLoss()
            #     loss_mmd = MMD(sequence_output[:,0],cls_other)
                
            #     loss += loss_mmd/20
                
            outputs = (loss,) + outputs
        

        if src_probs is not None:


            ## L2 Norm
            loss_KD_fct = MSELoss(reduction="mean")
            probs = torch.nn.functional.softmax(logits, dim=-1)

            probs_layers = []
            for logits_l in layer_logits:
                probs_layers.append(torch.nn.functional.softmax(logits_l, dim=-1))
            
            src_probs = src_probs.transpose(0,1)
            # print(src_probs.shape)
            src_probs_12 = src_probs[8].contiguous()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                inactive_subword = labels.view(-1) == loss_ignore_index
                active_loss[inactive_subword] = 0
                active_probs = probs.view(-1, self.num_labels)[active_loss]
                active_src_probs = src_probs_12.view(-1, self.num_labels)[active_loss]

                loss_KD_12 = loss_KD_fct(active_probs, active_src_probs)

                for i in range(8):
                  loss_layer = loss_KD_fct(probs_layers[i].view(-1, self.num_labels)[active_loss], src_probs[i].contiguous().view(-1, self.num_labels)[active_loss]).unsqueeze(0)
                  loss_KD_8 = loss_layer if loss_8 is None else torch.cat((loss_8,loss_layer),dim=0)
                loss_KD = loss_KD_12 + torch.matmul(loss_8,F.softmax(self.omega,dim=0)).squeeze(0)*weight
            else:
                loss_KD = loss_KD_fct(probs, src_probs)

            if tea_cls_en is not None:
                MMD = MMDLoss()
                mmd_loss_en_other = MMD(tea_cls_en, sequence_output[:,0])
                mmd_loss_en_en = MMD(tea_cls_en, cls_other)

                loss_KD += mmd_loss_en_other/1000 + mmd_loss_en_en/1000
                
            outputs = (loss_KD,) + outputs

        return outputs  # (loss_KD), (loss), scores, (hidden_states), (attentions)




