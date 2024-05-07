import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from EC_contrastive import *

Model_Path = "./Model/"


class EnzDataset(Dataset):
    def __init__(self, dataframe,data_path):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.data_path = data_path

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        data_path = self.data_path

        pssm_feature,hmm_feature,evo_feature = embedding(sequence_name,data_path)
        atom_features,seq_feature = get_atom_features(sequence_name, data_path)
        node_features = np.concatenate([pssm_feature, hmm_feature, atom_features, seq_feature], axis=1)
        graph = load_graph(sequence_name,data_path)
        return sequence_name, sequence, label, node_features, graph, evo_feature, atom_features

    def __len__(self):
        return len(self.labels)

def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    every_valid_pred = []
    every_valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_names, sequence, labels, node_features, graphs, evo_feature, atom_features = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda())
                graphs = Variable(graphs.cuda())
                evo_feature = Variable(evo_feature.cuda())
                y_true = Variable(labels.cuda())
            else:
                node_features = Variable(node_features)
                graphs = Variable(graphs)
                evo_feature = Variable(evo_feature)
                y_true = Variable(labels)

            node_features = torch.squeeze(node_features)
            graphs = torch.squeeze(graphs)
            evo_feature = torch.squeeze(evo_feature)
            y_true = torch.squeeze(y_true)

            y_pred, enzfeas = model(node_features, graphs, evo_feature)

            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred/8)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy().tolist()

            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)

            every_valid_pred.append([pred[1] for pred in y_pred])
            every_valid_true.append(list(y_true))

            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]
            epoch_loss += loss.item()
            n += 1

    epoch_loss_avg = epoch_loss / n
    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred,best_threshold =None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    #print(len(y_pred))
    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)

    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold }
    return results


def test(test_dataframe,data_path):

    test_loader = DataLoader(dataset=EnzDataset(test_dataframe,data_path), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    model_name ='EC_contrastive.pkl'
    model = SCREEN(NLAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT)

    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:0'))
    epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, test_loader)

    result_test = analysis(test_true, test_pred)
    print("========== Evaluate Test set ==========")
    print("Test recall: ", result_test['recall'])
    print("Test precision:", result_test['precision'])
    print("Test f1: ", result_test['f1'])
    print("Test mcc: ", result_test['mcc'])
    print("Test AUC: ", result_test['AUC'])
    print("Test AUPRC: ", result_test['AUPRC'])
    print("Threshold: ", result_test['threshold'])




def main():
    dataset = ["NN","HA_superfamily", "EF_superfamily", "PC", "EF_fold"]
    index = 0

    # loading the PDBid for testing
    data_dir = "./Dataset/" + dataset[index] + "/"
    f = open(data_dir + "test-" + dataset[index] + "_id.txt", "r")
    protein_list ,sequences, labels = [], [], []
    filedata = f.readlines()
    for line in filedata:
        protein = line.strip()
        protein_list.append(protein)
    f.close()

    # loading the testing label
    prot_seq = {}
    prot_anno = {}
    f = open(data_dir + dataset[index] +"_enzyme_label.txt", "r")
    data = f.readlines()
    for line in range(0, len(data)):
        if data[line].startswith('>'):
            protein = data[line].lstrip('>').strip()
            PDBID, Chain = protein.split('-') if '-' in protein else protein.split('_')
            pro = PDBID.lower() + "-" + Chain.upper()
            seq_p = data[line + 1].strip()
            query_anno = data[line + 2].strip()
            prot_seq[pro] = seq_p
            prot_anno[pro] = query_anno

    for prot in protein_list:
        label_list = []
        seq = prot_seq[prot]
        label = prot_anno[prot]
        sequences.append(seq)
        for i in range(len(label)):
            label_list.append(int(label[i]))
        labels.append(label_list)

    test_dic = {"ID": protein_list, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    test(test_dataframe, data_dir)


if __name__ == "__main__":
    main()



