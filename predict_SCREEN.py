import pandas as pd
from torch.autograd import Variable
from EC_contrastive import *
from feature_extract import *
import argparse

Model_Path = "./Model/"

class EnzCatDataset(Dataset):
    def __init__(self, dataframe,data_path):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.data_path = data_path

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        data_path = self.data_path

        pssm_feature,hmm_feature,evo_feature = embedding(sequence_name,data_path)
        atom_features,seq_feature = get_atom_features(sequence_name, data_path)
        node_features = np.concatenate([pssm_feature, hmm_feature, atom_features, seq_feature], axis=1)
        graph = load_graph(sequence_name,data_path)
        return sequence_name, sequence, node_features, graph, evo_feature, atom_features

    def __len__(self):
        return len(self.names)

def evaluate(model, data_loader):
    model.eval()
    every_valid_pred = []
    pred_dict = {}
    Enz_names = []
    Sequences = []
    binary_pred = []

    for data in data_loader:
        with torch.no_grad():
            sequence_names, sequence, node_features, graphs, evo_feature, atom_features = data
            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda())
                graphs = Variable(graphs.cuda())
                evo_feature = Variable(evo_feature.cuda())
            else:
                node_features = Variable(node_features)
                graphs = Variable(graphs)
                evo_feature = Variable(evo_feature)

            node_features = torch.squeeze(node_features)
            graphs = torch.squeeze(graphs)
            evo_feature = torch.squeeze(evo_feature)

            y_pred, _ = model(node_features, graphs, evo_feature)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred/8)
            y_pred = y_pred.cpu().detach().numpy()

            every_valid_pred.append([pred[1] for pred in y_pred])
            binary_pred.append( [1 if pred[1] >= 0.5 else 0 for pred in y_pred])
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]
            Sequences.append(sequence[0])
            Enz_names.append(sequence_names[0])

    return pred_dict,Sequences,binary_pred,Enz_names


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--PDBfile', type=str, default='./Example/PDB_id.txt', help='file containing PDBids')
    args = parser.parse_args()

    protein_list = []
    f = open(args.PDBfile, "r")
    filedata = f.readlines()
    for line in filedata:
        protein = line.strip()
        PDBID, Chain = protein.split('-') if '-' in protein else protein.split('_')
        pdb_id = PDBID.lower() + "-" + Chain.upper()
        protein_list.append(pdb_id)
    f.close()

    print("starting to extract features")
    prot_seq = feature_extraction(protein_list,'./Example/')

    print("starting to predict the catalytic residue")
    sequences = []
    for prot in protein_list:
        seq = prot_seq[prot]
        sequences.append(seq)

    test_dic = {"ID": protein_list, "sequence": sequences}
    test_dataframe = pd.DataFrame(test_dic)
    test_loader = DataLoader(dataset=EnzCatDataset(test_dataframe, './Example/'), batch_size=BATCH_SIZE, shuffle=True,num_workers=2)

    model_name = 'EC_contrastive.pkl'
    model = SCREEN(NLAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT)

    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:0'))
    pred_dict,sequences,binary_preds,enz_names = evaluate(model, test_loader)
    #print the predicted catalytic residue for every enzyme
    for i in range(len(enz_names)):
        PDB_id = enz_names[i]
        seq = sequences[i]
        print("For enzyme",PDB_id)
        for i in range(len(pred_dict[PDB_id])):
            if pred_dict[PDB_id][i]>0.5:
                print("The predicted catlytic residue:", seq[i] + str(i+1))

if __name__ == "__main__":
    main()



