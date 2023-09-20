import torch
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import glob
from sklearn.metrics import f1_score

from ast import literal_eval

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
args = parser.parse_args()
output_dir = args.output_dir

VOCAB_PATH = output_dir + r"/vocab.txt"
RAW_TRAIN_DATA = output_dir + r'/outputs/raw_train_data.txt'
NEW_TRAIN_DATA = output_dir + r'/outputs/new_train_data.txt'
RAW_DEV_DATA = output_dir + r'/outputs/raw_dev_data.txt'
NEW_DEV_DATA = output_dir + r'/outputs/new_dev_data.txt'
TRAIN = output_dir + r'/train'
DEV_PATH = output_dir + r'/dev'

TRAIN_NUM = 1


class CBOW(nn.Module):

    def __init__(self, embedding_size, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size, bias=False)
        self.w_to_i = {}
        self.i_to_w = {}
        self.raw_data = []
        self.new_data = []

    def forward(self, inputs):
        embeddings = self.embeddings(inputs).mean(1).squeeze(1)
        out = self.linear(embeddings)
        return out


w_to_i = {}
i_to_w = {}
raw_data = []
new_data = []

raw_data = []
new_data = []
raw_dev_data = []
new_dev_data = []


def get_word2ix(path=VOCAB_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()

        for line in data:
            w_to_i[line.split("\t")[1].strip()] = int(line.split("\t")[0])
            i_to_w[int(line.split("\t")[0])] = line.split("\t")[1].strip()

    return len(w_to_i)


def read_text_file(file_path):
    tokens = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens.append(line.strip())

    for i in range(5):
        tokens.insert(0, '[PAD]')
        tokens.append('[PAD]')

    for i in range(5, len(tokens) - 5):
        context = [
            tokens[i - 5],
            tokens[i - 4],
            tokens[i - 3],
            tokens[i - 2],
            tokens[i - 1],
            tokens[i + 1],
            tokens[i + 2],
            tokens[i + 3],
            tokens[i + 4],
            tokens[i + 5],
        ]

        target = tokens[i]

        # Mapping to indices
        context_to_i = [w_to_i[w] if w in w_to_i else w_to_i['[UNK]'] for w in context]
        target_to_i = w_to_i[target] if target in w_to_i else w_to_i['[UNK]']

        # Appending to arrays
        raw_data.append((context, target))
        new_data.append((context_to_i, target_to_i))

        #Creating txt file
        # with open(RAW_TRAIN_DATA, "a", encoding='utf-8') as rd:
        #     rd.write(f"{context}\t{target}")
        #     rd.write("\n")
        # with open(NEW_TRAIN_DATA, "a", encoding='utf-8') as d:
        #     d.write(f"{context_to_i}\t{target_to_i}")
        #     d.write("\n")

        # print(context, target)
   

def read_saved_data():
    with open(
            RAW_TRAIN_DATA, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            context = literal_eval(line.split("\t")[0])
            target = line.split("\t")[1].strip()
            raw_data.append((context, target))
    with open(
            NEW_TRAIN_DATA,
            "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            context = literal_eval(line.split("\t")[0])
            target = literal_eval(line.split("\t")[1].strip())
            new_data.append((context, target))

    with open(RAW_DEV_DATA,
              "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            context = literal_eval(line.split("\t")[0])
            target = line.split("\t")[1].strip()
            raw_dev_data.append((context, target))

    with open(NEW_DEV_DATA,
              "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            context = literal_eval(line.split("\t")[0])
            target = literal_eval(line.split("\t")[1].strip())
            new_dev_data.append((context, target))


def read_dev_text_file(file_path):
    tokens = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens.append(line.strip())

    for i in range(5, len(tokens) - 5):
        context = [
            tokens[i - 5],
            tokens[i - 4],
            tokens[i - 3],
            tokens[i - 2],
            tokens[i - 1],
            tokens[i + 1],
            tokens[i + 2],
            tokens[i + 3],
            tokens[i + 4],
            tokens[i + 5],
        ]

        target = tokens[i]

        # Mapping to indices
        context_to_i = [w_to_i[w] if w in w_to_i else w_to_i['[UNK]'] for w in context]
        target_to_i = w_to_i[target] if target in w_to_i else w_to_i['[UNK]']

        # Appending to arrays
        raw_dev_data.append((context, target))
        new_dev_data.append((context_to_i, target_to_i))

        #Creating txt file
        # with open(RAW_DEV_DATA, "a", encoding='utf-8') as rd:
        #     rd.write(f"{context}\t{target}")
        #     rd.write("\n")
        # with open(NEW_DEV_DATA, "a", encoding='utf-8') as d:
        #     d.write(f"{context_to_i}\t{target_to_i}")
        #     d.write("\n")

        # print(context, target)


def get_files(path):
    file_list = list(glob.glob(f"{path}/*.txt"))
    return file_list


def create_dataset(train_path, dev_path):
    train_files = get_files(train_path)
    dev_files = get_files(dev_path)
    print("-----------------TRAIN DATA---------------------")
    for file in train_files:
        read_text_file(file)
    print("-----------------DEV DATA---------------------")
    for file in dev_files:
        read_dev_text_file(file)


def main():
    vocab_size = get_word2ix(VOCAB_PATH)
    create_dataset(TRAIN, DEV_PATH)
    #read_saved_data()

    m1 = CBOW(100, vocab_size)
    # optimizer = optim.Adam(m1.parameters(), lr=0.01)

    train_context = torch.tensor([data[0] for data in new_data])
    train_target = torch.tensor([data[1] for data in new_data])

    train_dataset = TensorDataset(train_context, train_target)
    train_dataloader = DataLoader(train_dataset, batch_size=64)

    # DEV DATALOADER

    dev_context = torch.tensor([data[0] for data in new_dev_data])
    dev_target = torch.tensor([data[1] for data in new_dev_data])

    dev_dataset = TensorDataset(dev_context, dev_target)
    dev_dataloader = DataLoader(dev_dataset, batch_size=64)

    # TRAINING
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    min_loss = 10000
    optimized_emb = -1
    loss = nn.CrossEntropyLoss()

    #
    # best_perf_dict = {"metric": 0, "dev_loss": 1000, "epoch": 0, "model_param": {}, "optim_param": {}}
    lrs = [0.01, 0.001, 0.0001]
    best_lr = -1
    best_epoch = -1
    best_dev_loss = 1000
    best_f1 = 0
    for i in range(3):
        optimizer = optim.Adam(m1.parameters(), lr=lrs[i])
        best_perf_dict = {"metric": 0, "dev_loss": 1000, "epoch": 0, "model_param": {}, "optim_param": {}}

        print(f"---TRAINING FOR LEARNING RATE: {lrs[i]}")

        for epoch in range(10):
            train_loss = []
            dev_loss = []
            for context, target in tqdm(train_dataloader):
                m1.train()

                optimizer.zero_grad()
                out = m1(context.to(device))

                model_loss = loss(out, target.long().to(device))

                model_loss.backward()
                optimizer.step()

                train_loss.append(model_loss.cpu().item())

            # print(f"Epoch: {epoch + 1}, loss: {model_loss}")
            print(f"Average training batch loss: {np.mean(train_loss)}")

            true_labels = []
            pred_labels = []

            for context, target in tqdm(dev_dataloader):
                m1.eval()
                with torch.no_grad():
                    out = m1(context.to(device))
                    preds = torch.argmax(out, dim=1)

                    d_loss = loss(out, target.long().to(device))

                    pred_labels.extend(preds.cpu().tolist())
                    true_labels.extend(target.tolist())

                    dev_loss.append(d_loss.cpu().item())

            dev_loss_avg = np.mean(dev_loss)
            print(f"Epoch {epoch + 1}, Dev Loss: {dev_loss_avg} ")

            dev_f1 = f1_score(true_labels, pred_labels, average='macro')
            print(f"Dev F1: {dev_f1}\n")

            # Update the `best_perf_dict` if the best dev performance seen
            # so far is beaten

            if dev_loss_avg < best_perf_dict["dev_loss"]:
                best_perf_dict["dev_loss"] = dev_loss_avg
                best_perf_dict["metric"] = dev_f1
                best_perf_dict["epoch"] = epoch + 1
                best_perf_dict["model_param"] = m1.state_dict()
                best_perf_dict["optim_param"] = optimizer.state_dict()

            if dev_loss_avg < best_dev_loss:
                best_lr = lrs[i]
                best_epoch = epoch + 1
                best_f1 = dev_f1
                best_dev_loss = dev_loss_avg


            # if dev_f1 > best_perf_dict["metric"]:
            #     best_perf_dict["metric"] = dev_f1
            #     best_perf_dict["epoch"] = epoch + 1
            #     best_perf_dict["model_param"] = m1.state_dict()
            #     best_perf_dict["optim_param"] = optimizer.state_dict()

        print(f"""\nBest Dev performance of {best_perf_dict["dev_loss"]} at epoch {best_perf_dict["epoch"]}""")
        print(best_perf_dict["model_param"]["embeddings.weight"].size())
        print(best_perf_dict["model_param"]["embeddings.weight"])

        torch.save(best_perf_dict["model_param"], 'outputs/saved_model3_01/model.pt')
        torch.save(best_perf_dict["optim_param"], 'outputs/saved_model3_01/optimizer.pt')

        torch.save({
            "model_param": best_perf_dict["model_param"],
            "optim_param": best_perf_dict["optim_param"],
            "dev_metric": best_perf_dict["metric"],
            "dev_loss": best_perf_dict["dev_loss"],
            "epoch": best_perf_dict["epoch"],
        }, f"{output_dir}/{lrs[i]}")

    print(f"Least development loss of {best_dev_loss} was encountered with learning rate: {best_lr} at epoch: {best_epoch}")




    model_path1 = output_dir + r"/0.01"
    model_path2 = output_dir + r"/0.001"
    model_path3 = output_dir + r"/0.0001"

    checkpoint1 = torch.load(model_path1)
    checkpoint2 = torch.load(model_path2)
    checkpoint3 = torch.load(model_path3)


    cos = torch.nn.CosineSimilarity(dim=0)

    model1 = CBOW(100, 18061)
    model2 = CBOW(100, 18061)
    model3 = CBOW(100, 18061)

    optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    optimizer3 = optim.Adam(model3.parameters(), lr=0.0001)


    model1.load_state_dict(checkpoint1["model_param"])
    model2.load_state_dict(checkpoint2["model_param"])
    model3.load_state_dict(checkpoint3["model_param"])

    optimizer1.load_state_dict(checkpoint1["optim_param"])
    optimizer2.load_state_dict(checkpoint2["optim_param"])
    optimizer3.load_state_dict(checkpoint3["optim_param"])

    print(f"""Dev F1 of loaded model with learning rate 0.01: {checkpoint1["dev_metric"]} and Dev Loss: {checkpoint1["dev_loss"]}at epoch {checkpoint1["epoch"]}""")
    print(f"""Dev F1 of loaded model with learning rate 0.001: {checkpoint2["dev_metric"]} and Dev Loss: {checkpoint2["dev_loss"]}at epoch {checkpoint2["epoch"]}""")
    print(f"""Dev F1 of loaded model with learning rate 0.0001: {checkpoint3["dev_metric"]} and Dev Loss: {checkpoint3["dev_loss"]}at epoch {checkpoint3["epoch"]}""")

    best_dev_loss = min(checkpoint1["dev_loss"], checkpoint2["dev_loss"], checkpoint3["dev_loss"])
    best_lr = 0
    best_model = -1
    if best_dev_loss == checkpoint1["dev_loss"]:
        best_lr = 0.01
        best_model = model1
    elif best_dev_loss == checkpoint2["dev_loss"]:
        best_lr = 0.001
        best_model = model2
    else:
        best_lr = 0.0001
        best_model = model3

    print(f"\nBest model learning rate: {best_lr} and dev loss: {best_dev_loss}")

    model_embeddings = best_model.state_dict()['embeddings.weight']

    w_to_i = {}
    i_to_w = {}
    vocab = []
    def get_word2ix2(path= output_dir + r"/vocab.txt"):
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()

            for line in data:
                w_to_i[line.split("\t")[1].strip()] = int(line.split("\t")[0])
                i_to_w[int(line.split("\t")[0])] = line.split("\t")[1].strip()
                vocab.append(line.split("\t")[1].strip())

    get_word2ix2(output_dir + r"/vocab.txt")

    def similarity(arr):
        w1 = model_embeddings[w_to_i[arr[0]]]
        w2 = model_embeddings[w_to_i[arr[1]]]
        w3 = model_embeddings[w_to_i[arr[2]]]
        w4 = model_embeddings[w_to_i[arr[3]]]

        print(f"Similarity score for {arr[0]} and {arr[1]}: {cos(w1, w2)}")
        print(f"Similarity score for {arr[2]} and {arr[3]}: {cos(w3, w4)}")
        print("\n\n")


    def analogy(arr):
        wa = model_embeddings[w_to_i[arr[0]]]
        wb = model_embeddings[w_to_i[arr[1]]]
        wc = model_embeddings[w_to_i[arr[2]]]
        wd = wa - wc + wb

        min = 10
        ans = -1
        for i in range(len(model_embeddings)):
            s = cos(wd, model_embeddings[i])
            if s < min:
                min = s
                ans = i
        print(f"Analogy, {arr[0]}:{arr[1]}::{arr[2]}:{i_to_w[ans]} ")




    calc_sim = [["cat", "tiger", "plane", "human"], ["my", "mine", "happy", "human"], ["happy", "cat", "king", "princess"], ["ball", "racket", "good", "ugly"],
                ["cat", "racket", "good", "bad"]]

    calc_ana = [["king", "queen", "man"], ["king", "queen", "prince"], ["king", "man", "queen"], ["woman", "man", "princess"], ["prince", "princess", "man"]]

    print("\nSimilarity scores considering embedding weights: ")
    for i in range(len(calc_sim)):
        similarity(calc_sim[i])

    print("\nAnalogy:")
    for j in range(len(calc_ana)):
        analogy(calc_ana[j])

    with open(output_dir + r"/embeddings.txt", "a", encoding="utf-8") as e:
        e.write(f"{len(vocab)} 100")
        e.write("\n")

        for i in range(len(vocab)):
            v_w = vocab[i]
            emb = model_embeddings[i].tolist()
            s = ''
            for i in range(len(emb)):
                if i == len(emb) - 1:
                    s += str(emb[i])
                else:
                    s += str(emb[i]) + " "
            e.write(f"{v_w} {s}")
            e.write("\n")
if __name__ == "__main__":
    main()











