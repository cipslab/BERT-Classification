import torch
import argparse
import pandas as pd
from dataset import PDataset, collate_fn
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
from tqdm import tqdm
from utils import AverageMeter, calculateMetrics
import os
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

def model_save(model, output_dir):

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()


    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    return

def train(model, optimizer, scheduler, dataloader, args):
    model.train()
    losses = AverageMeter()

    for epoch_i in range(0, args.epochs):
        losses.reset()
        for i, (story_id, chunk_num, input, attn_mask, label) in enumerate(tqdm(dataloader)):
            # print(story_id, chunk_num, input, attn_mask, label)
            input = input.cuda()
            attn_mask = attn_mask.cuda()
            label = label.cuda()
            model.zero_grad()        

            outputs = model(input, 
                        token_type_ids=None, 
                        attention_mask=attn_mask, 
                        labels=label)
            
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0].mean()
            losses.update(loss.item())
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        print(f"Epoch: {epoch_i}\tIteration: {i}\tLoss: {losses.avg}")
    model_save(model, args.output_dir)

def test(rep_model,cls_model, dataloader, save_results=False, fname="test.csv"):
    rep_model.eval()
    cls_model.eval()
    predictions = []
    labels = []
    story_ids = []
    for i, (story_id, input, attn_mask, label, split_lens) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            batch_size, max_split_len, dim = input.shape
            input = input.cuda()
            attn_mask = attn_mask.cuda()
            label = label.cuda()
            split_lens = split_lens.cuda()
            input = input.view(-1,512)
            attn_mask = attn_mask.view(-1,512)
            # print(input[0])
            # print(attn_mask[0])
            mask = (torch.arange(max_split_len)[None, :].cuda() < split_lens[:, None]).unsqueeze(-1)
            representation = rep_model(input, attention_mask=attn_mask)[1].view(batch_size, max_split_len, 768) * mask
            
            representation = torch.div(torch.sum(representation, 1), split_lens.view(-1,1))
            # print(representation.shape)
            # print(split_lens.unsqueeze(1).shape)
            # print(representation.shape)
            # representation = torch.mean(representation,0).unsqueeze(0)
            classification = F.softmax(cls_model(representation),1).cpu()
            classification = torch.argmax(classification,1).view(-1).cpu()
            # print(classification)
            predictions.extend(classification.cpu().tolist())
            # print(classification)
            # print(label)
            labels.extend(label.cpu().tolist())
            story_ids.extend(story_id.view(-1).cpu().tolist())
         
            del representation
            del classification
            del label
            del attn_mask
            del input
    
    if save_results:
        save_df = pd.DataFrame({"story_id": story_ids, "bert_classification": predictions})
        save_df.to_csv(fname)

    # correct = 0
    # for i in range(len(predictions)):
    #     if predictions[i] == labels[i]:
    #         correct+=1
    
    # print(f"Test Accuracy: {correct/float(len(predictions))}")
    # calculateMetrics(labels, predictions,'BERT')

def load_representation_model(model_path):
    state_dict = torch.load(model_path)
    model = BertModel.from_pretrained("bert-base-uncased",add_pooling_layer=True)
    # model_state_dict = representation_model.state_dict()
    new_state_dict = {}
    for key in state_dict:
        if key == "classifier.weight" or key == "classifier.bias":
            continue
        model_key = ".".join(key.split(".")[1:])
        new_state_dict[model_key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    return model

def load_classifier(model_path):
    state_dict = torch.load(model_path)
    classifier_model = nn.Sequential(
        nn.Linear(768,3,bias=True)
    )
    # print(list(state_dict.keys()))
    new_state_dict = {}
    for key in state_dict:
        if key == "classifier.weight" or key == "classifier.bias":
            model_key = "0." + key.split(".")[1]
            new_state_dict[model_key] = state_dict[key]

    classifier_model.load_state_dict(new_state_dict)
    return classifier_model


if __name__ == "__main__":

    device = torch.device("cpu")
    parser = argparse.ArgumentParser(description='Train a Bert Classifier')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--lr', type=int, default=2e-5)
    parser.add_argument('--eps', type=int, default=1e-8)
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--output-dir', type=str, default="./propaganda_model/")
    parser.add_argument('--save_file', type=str, default="new-blog-propaganda-results.csv")
    parser.add_argument('--num-labels', type=int, default=3, help='Number of labels to fine tune BERT on')
    parser.add_argument('--data-path', type='str', default='../data/blog_sample/blog_processed_disinfo.csv', help='Training/Testing DataFrame')
    parser.add_argument('--save-model', action='store_true')
    args = parser.parse_args()
    
    all_df = pd.read_csv(args.data_path)

    train_df, test_df = train_test_split(all_df, test_size=0.2, random_state=2)
    # test_df = pd.read_csv('../data/blog_sample/blog_all.csv')
    # train_df = pd.read_csv('../data/balanced_data/train.tsv', sep = '\t')
    # # test_df = pd.read_csv('../data/balanced_data/test.tsv', sep = '\t')
    # # test_df = pd.read_csv('../data/blog_sample/blog_data.csv')
    # test_df = pd.read_csv('../data/blog_sample/new_processed_ukraine_blog.csv')
    train_df.drop_duplicates(subset=["story_id"], inplace=True)
    test_df.drop_duplicates(subset=["story_id"], inplace=True)
    test_df.dropna(subset=["raw_text"],inplace=True)
    # test_df = test_df[~test_df["story_id"].isin(train_df["story_id"].unique())]
    dataset = PDataset(train_df, test_df, args.mode)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = args.num_labels, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        model.cuda()
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        print('There are %d GPU(s) available.' % num_gpus)
        
        if num_gpus > 1:
            model = nn.DataParallel(model)

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        model.to(device)
    
    if args.mode == "train":
        dataloader = DataLoader(dataset, shuffle=True, batch_size = args.batch_size)
        optimizer = AdamW(model.parameters(),lr = args.lr, eps = args.eps)
    
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = args.warmup, # Default value in run_glue.py
                                            num_training_steps = len(dataloader)*args.epochs)
        train(model,optimizer, scheduler, dataloader, args)
    
    else:
        dataloader = DataLoader(dataset, shuffle=False, batch_size = args.batch_size, collate_fn=collate_fn)
        representation_model = load_representation_model(os.path.join(args.output_dir,'pytorch_model.bin'))
        classifier_model = load_classifier(os.path.join(args.output_dir,'pytorch_model.bin'))
        representation_model.to(device)
        classifier_model.to(device)
        representation_model = nn.DataParallel(representation_model)
        classifier_model = nn.DataParallel(classifier_model)
        test(representation_model, classifier_model, dataloader, save_results=args.save_model, fname=args.save_file)


