from biotoolbox.structure_file_reader import build_structure_container_for_pdb
from biotoolbox.contact_map_builder import DistanceMapBuilder
from functools import partial
import gzip
import secrets
import math
from Bio import pairwise2
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from evofea_embedding import *


torch.set_grad_enabled(False)
############replace with your own path
PSIBLAST = "/home/tongpan/ncbi-blast-2.13.0+/bin/psiblast"
HHBLITS = "/Workspace/tongpan/anaconda3/pkgs/hhsuite-3.3.0-py38pl526h6ed170a_1/bin/hhblits"  # HH-SUITE software folder path for features extraction
UR90 = "/Workspace/tongpan/BlastDB/uniref90.fasta"  # database for pssm path #PSIBLAST DATABASE folder path for features extraction
HHDB = "/Workspace/tongpan/uniclust30_2018_08/uniclust30_2018_08"
dssp = "/Workspace/tongpan/anaconda3/pkgs/dssp-2.2.1-1/bin/mkdssp"


res_dict = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M', 'TRP': 'W',
            'CYS': 'C', 'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K', 'ARG': 'R'}
restype_1to3 = {'A': 'ALA','R': 'ARG','N': 'ASN','D': 'ASP','C': 'CYS','Q': 'GLN','E': 'GLU','G': 'GLY','H': 'HIS','I': 'ILE','L': 'LEU','K': 'LYS','M': 'MET','F': 'PHE','P': 'PRO','S': 'SER','T': 'THR','W': 'TRP','Y': 'TYR','V': 'VAL',}

def def_atom_features():
    A = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,3,0]}
    V = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,1,0], 'CG1':[0,3,0], 'CG2':[0,3,0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,1,1] }
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,1], 'CG':[0,2,1], 'CD':[0,2,1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,1,0], 'CD1':[0,3,0], 'CD2':[0,3,0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'CG1':[0,2,0], 'CG2':[0,3,0], 'CD1':[0,3,0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,2,0], 'CD':[0,2,0], 'NE':[0,1,0], 'CZ':[1,0,0], 'NH1':[0,2,0], 'NH2':[0,2,0] }
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[-1,0,0], 'OD1':[-1,0,0], 'OD2':[-1,0,0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[-1,0,0], 'OE1':[-1,0,0], 'OE2':[-1,0,0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'OG':[0,1,0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'OG1':[0,1,0], 'CG2':[0,3,0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'SG':[-1,1,0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,0,0], 'OD1':[0,0,0], 'ND2':[0,2,0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,0,0], 'OE1':[0,0,0], 'NE2':[0,2,0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'ND1':[-1,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'NE2':[-1,1,1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,2,0], 'CE':[0,2,0], 'NZ':[0,3,1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,0,1], 'OH':[-1,1,0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'SD':[0,0,0], 'CE':[0,3,0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,0,1], 'NE1':[0,1,1], 'CE2':[0,0,1], 'CE3':[0,1,1], 'CZ2':[0,1,1], 'CZ3':[0,1,1], 'CH2':[0,1,1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S, 'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}

    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0]/2+0.5,i_fea[1]/3,i_fea[2]]

    return atom_features


def write_all_fasta(data_dir,protein_list,prot_seq):
    # write fasta
    with open(data_dir + "fasta/allseq.fa", "w") as f:
        for pro in protein_list:
            f.write(">" + pro + "\n" + prot_seq[pro]+ "\n")
    f.close()


def make_distance_maps(pdbfile, chain=None, sequence=None):

    pdb_handle = open(pdbfile, 'r')
    structure_container = build_structure_container_for_pdb(pdb_handle.read(), chain).with_seqres(sequence)
    mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)  # start with CA distances
    ca = mapper.generate_map_for_pdb(structure_container)
    pdb_handle.close()

    return ca.chains

def Write_single_fasta(input_dir,output_dir):
    f = open(input_dir, "r")
    data = f.readlines()
    for line in range(0, len(data)):
        if data[line].startswith('>'):
            protein = data[line].lstrip('>').strip()
            PDBID, Chain= protein.split("-")if '-' in protein else protein.split('_')
            pro = PDBID.lower() + "-" + Chain.upper()
            seq_p = data[line + 1].strip()
            with open(output_dir + "fasta/" + pro + ".fa", "w") as f:
                f.write(">" + pro + "\n" + seq_p)
    f.close()

def write_contact_map(prot, prot2seq, data_path):
    if not os.path.exists(data_path + 'contact_map'):
        os.makedirs(data_path + 'contact_map')
    if not os.path.exists(data_path + 'contact_map/{}.npy'.format(prot)):
        pdb, chain = prot.split('-')
        if chain == "_":
            chain = "A"
        contact_dir = data_path +'PDB/SavechainPDB/'
        try:
            ca = make_distance_maps(contact_dir + prot + '.pdb', chain=chain, sequence=prot2seq[prot])
            A_ca = ca[chain]['contact-map']
            np.save(data_path + 'contact_map/{}.npy'.format(prot), A_ca)
        except Exception as e:
            print(e)

def cal_PSSM(seq_list,pssm_dir):
    if not os.path.exists(pssm_dir + 'processed_pssm/'):
        os.makedirs(pssm_dir + 'processed_pssm/')

    for seqid in seq_list:
        print(seqid)
        PDBID, Chain = seqid.split('-')
        pro1 = PDBID + "_" + Chain
        file = seqid+'.pssm'
        file1 = pro1 + '.pssm'
        if os.path.exists(pssm_dir+file):
            pssm_file = pssm_dir+file
        else:
            pssm_file = pssm_dir + file1

        with open(pssm_file,'r') as fin:
            fin_data = fin.readlines()
            pssm_begin_line = 3
            pssm_end_line = 0
            for i in range(1,len(fin_data)):
                if fin_data[i] == '\n':
                    pssm_end_line = i
                    break
            feature = np.zeros([(pssm_end_line-pssm_begin_line),20])
            axis_x = 0
            for i in range(pssm_begin_line,pssm_end_line):
                raw_pssm = fin_data[i].split()[2:22]
                axis_y = 0
                for j in raw_pssm:
                    feature[axis_x][axis_y]= (1 / (1 + math.exp(-float(j))))
                    axis_y+=1
                axis_x+=1

        np.save(pssm_dir + 'processed_pssm/{}.npy'.format(seqid), feature)
    return

def cal_HMM(seq_list,hmm_dir):
    if not os.path.exists(hmm_dir + 'processed_hhm'):
        os.makedirs(hmm_dir + 'processed_hhm')
    for seqid in seq_list:
        PDBID, Chain = seqid.split('-')
        pro1 = PDBID + "_" + Chain
        file = seqid + '.hhm'
        file1 = pro1 + '.hhm'
        if os.path.exists(hmm_dir+file):
            hmm_file = hmm_dir + file
        else:
            hmm_file = hmm_dir + file1

        with open(hmm_file,'r') as fin:
            fin_data = fin.readlines()
            hhm_begin_line = 0
            hhm_end_line = 0
            for i in range(len(fin_data)):
                if '#' in fin_data[i]:
                    hhm_begin_line = i+5
                elif '//' in fin_data[i]:
                    hhm_end_line = i
            feature = np.zeros([int((hhm_end_line-hhm_begin_line)/3),30])
            axis_x = 0
            for i in range(hhm_begin_line,hhm_end_line,3):
                line1 = fin_data[i].split()[2:-1]
                line2 = fin_data[i+1].split()
                axis_y = 0
                for j in line1:
                    if j == '*':
                        feature[axis_x][axis_y]=9999/10000.0
                    else:
                        feature[axis_x][axis_y]=float(j)/10000.0
                    axis_y+=1
                for j in line2:
                    if j == '*':
                        feature[axis_x][axis_y]=9999/10000.0
                    else:
                        feature[axis_x][axis_y]=float(j)/10000.0
                    axis_y+=1
                axis_x+=1
            feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

        np.save(hmm_dir + 'processed_hhm/{}.npy'.format(seqid), feature)
    return

def cal_DSSP(seq_list,dssp_dir):
    maxASA = {'G':188,'A':198,'V':220,'I':233,'L':304,'F':272,'P':203,'M':262,'W':317,'C':201, 'S':234,'T':215,'N':254,'Q':259,'Y':304,'H':258,'D':236,'E':262,'K':317,'R':319}
    map_ss_8 = {' ':[1,0,0,0,0,0,0,0],'S':[0,1,0,0,0,0,0,0],'T':[0,0,1,0,0,0,0,0],'H':[0,0,0,1,0,0,0,0], 'G':[0,0,0,0,1,0,0,0],'I':[0,0,0,0,0,1,0,0],'E':[0,0,0,0,0,0,1,0],'B':[0,0,0,0,0,0,0,1]}
    if not os.path.exists(dssp_dir + 'processed_dssp'):
        os.makedirs(dssp_dir + 'processed_dssp')
    for seqid in seq_list:
        file = seqid + '.dssp'
        if os.path.exists(dssp_dir + file):
            with open(dssp_dir + file, 'r') as fin:
                fin_data = fin.readlines()
            seq_feature = []
            p = 0
            while fin_data[p].strip()[0] != "#":
                p += 1
            for i in range(p + 1, len(fin_data)):
                line = fin_data[i]
                if line[13] not in maxASA.keys() or line[9]==' ':
                    continue
                feature = np.zeros([14])
                feature[:8] = map_ss_8[line[16]] #ss
                feature[8] = min(float(line[35:38]) / maxASA[line[13]], 1)#ACC,ASA
                feature[9] = (float(line[85:91]) + 1) / 2
                feature[10] = min(1, float(line[91:97]) / 180)
                feature[11] = min(1, (float(line[97:103]) + 180) / 360)
                feature[12] = min(1, (float(line[103:109]) + 180) / 360) #PHI
                feature[13] = min(1, (float(line[109:115]) + 180) / 360) #PSI
                seq_feature.append(feature.reshape((1, -1)))

            np.save(dssp_dir + 'processed_dssp/{}.npy'.format(seqid), seq_feature)
    return


def matched_dssp(sequence_name,ref_seq,PDB_dir,dssp_dir):

    file_path = PDB_dir + sequence_name + '.pdb'
    pdb_handle = open(file_path, 'r')
    sequence = ''
    res_id_list = []

    while True:
        line = pdb_handle.readline()
        if line.startswith('ATOM'):
            res = res_dict[line[17:20]]
            res_pdb_id = int(line[22:26])  # 第几个残基

            if len(res_id_list) == 0:
                res_id_list.append(res_pdb_id)
                sequence = str(res)
            elif res_id_list[-1] != res_pdb_id:
                res_id_list.append(res_pdb_id)
                sequence = sequence + str(res)
        if line.startswith('TER'):
            break

    dssp_feature = np.load(dssp_dir + 'processed_dssp/' + sequence_name + '.npy', allow_pickle=True)
    data_shape = dssp_feature[0].shape

    alignments = pairwise2.align.globalxx(ref_seq, sequence)
    align_ref_seq = alignments[0].seqA
    align_seq = alignments[0].seqB

    new_dssp = []
    for aa in align_seq:
        if aa == "-":
            new_dssp.append(np.zeros(data_shape))
        else:
            new_dssp.append(dssp_feature[0])
            dssp_feature = dssp_feature[1:]
            #new_dssp.append(list(dssp_feature).pop(0))

    matched_dssp = []
    for i in range(len(align_ref_seq)):
        if align_ref_seq[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])
    matched_dssp = np.concatenate(matched_dssp, axis=0)

    if not os.path.exists(dssp_dir + 'matched_dssp'):
        os.makedirs(dssp_dir + 'matched_dssp')
    np.save(dssp_dir + 'matched_dssp/{}.npy'.format(sequence_name), matched_dssp)
    return

def Multi_process_PSSM_generation(PDB_id,data_path):
    if not os.path.exists(data_path + 'pssm/'):
        os.makedirs(data_path + 'pssm/')
    if not os.path.exists(data_path + 'hhm/'):
        os.makedirs(data_path + 'hhm/')

    if os.path.exists(data_path + 'pssm/{}.pssm'.format(PDB_id)) == False:
        print("starting to generate pssm", PDB_id)
        os.system("{0} -db {1} -num_iterations 3 -num_alignments 1 -num_threads 2 -query {3}fasta/{2}.fa -out {3}fasta/{2}.bla -out_ascii_pssm {3}pssm/{2}.pssm".format(PSIBLAST, UR90, PDB_id, data_path))
    if os.path.exists(data_path + 'hhm/{}.hhm'.format(PDB_id)) == False:
        print("starting to generate hhm", PDB_id)
        os.system("{0} -i {2}fasta/{1}.fa -ohhm {2}hhm/{1}.hhm -oa3m {2}fasta/{1}.a3m -d {3} -v 0 -maxres 40000 -cpu 6 -Z 0 -o {2}fasta/{1}.hhr".format(HHBLITS, PDB_id, data_path, HHDB))
    """if os.path.exists(data_path + 'dssp/{}.dssp'.format(PDB_id)) == False:
        print("starting to generate dssp", PDB_id)
        os.system("{} -i {}.pdb -o {}.dssp".format(dssp, data_path + "PDB/SavechainPDB/" + PDB_id, data_path + "dssp/" + PDB_id))"""

def get_pdb_DF(prot, data_path):

    atom_fea_dict = def_atom_features()
    atom_count = -1
    res_count = -1

    file_path = data_path + "PDB/SavechainPDB/" + prot + '.pdb'
    pdb_file = open(file_path, 'r')

    pdb_res = pd.DataFrame(columns=['ID','atom','res','res_id','xyz','B_factor'])
    res_id_list = []
    sequence = ''
    before_res_pdb_id = None
    Relative_atomic_mass = {'H':1,'C':12,'O':16,'N':14,'S':32,'FE':56,'P':31,'BR':80,'F':19,'CO':59,'V':51, 'I':127,'CL':35.5,'CA':40,'B':10.8,'ZN':65.5,'MG':24.3,'NA':23,'HG':200.6,'MN':55, 'K':39.1,'AP':31,'AC':227,'AL':27,'W':183.9,'SE':79,'NI':58.7}

    while True:
        line = pdb_file.readline()
        if line.startswith('ATOM') :
            atom_type = line[76:78].strip()
            if atom_type not in Relative_atomic_mass.keys():
                continue
            atom_count+=1
            res_pdb_id = int(line[22:26]) #第几个残基
            if res_pdb_id != before_res_pdb_id:
                res_count +=1
            before_res_pdb_id = res_pdb_id
            if line[12:16].strip() not in ['N','CA','C','O','H']:
                is_sidechain = 1
            else:
                is_sidechain = 0
            res = res_dict[line[17:20]]
            atom = line[12:16].strip()
            try:
                atom_fea = atom_fea_dict[res][atom]
            except KeyError:
                atom_fea = [0.5,0.5,0.5]
            tmps = pd.Series( {'ID': atom_count, 'atom':line[12:16].strip(),'atom_type':atom_type, 'res': res, 'res_id': int(line[22:26]), 'xyz': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),'occupancy':float(line[54:60]),
                 'B_factor': float(line[60:66]),'mass':Relative_atomic_mass[atom_type],'is_sidechain':is_sidechain, 'charge':atom_fea[0],'num_H':atom_fea[1],'ring':atom_fea[2]})
            if len(res_id_list) == 0:
                res_id_list.append(res_pdb_id)
                sequence = str(res)
            elif res_id_list[-1] != res_pdb_id:
                res_id_list.append(res_pdb_id)
                sequence = sequence + str(res)
            pdb_res = pdb_res.append(tmps,ignore_index=True)
        if line.startswith('TER'):
            break

    if not os.path.exists(data_path + 'Atom_feas/'):
        os.makedirs(data_path + 'Atom_feas/')

    with open(data_path + 'Atom_feas' + '/{}.csv.pkl'.format(prot), 'wb') as f:
        pickle.dump({'pdb_DF': pdb_res, 'res_id_list': res_id_list, 'sequence': sequence}, f)
    return

def cal_atomFea(seqlist, data_path):
    atom_vander_dict = {'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.85,'H':1.2,'D':1.2,'SE':1.9,'P':1.8,'FE':2.23,'BR':1.95, 'F':1.47,'CO':2.23,'V':2.29,'I':1.98,'CL':1.75,'CA':2.81,'B':2.13,'ZN':2.29,'MG':1.73,'NA':2.27,
                        'HG':1.7,'MN':2.24,'K':2.75,'AC':3.08,'AL':2.51,'W':2.39,'NI':2.22}
    for key in atom_vander_dict.keys():
        atom_vander_dict[key] = (atom_vander_dict[key] - 1.52) / (1.85 - 1.52)

    for seq_id in tqdm(seqlist):
        with open(data_path +'Atom_feas' + '/{}.csv.pkl'.format(seq_id), 'rb') as f:
            tmp = pickle.load(f)

        pdb_res, res_id_list = tmp['pdb_DF'], tmp['res_id_list']
        pdb_res = pdb_res[pdb_res['atom_type']!='H']

        mass = np.array(pdb_res['mass'].tolist()).reshape(-1, 1)
        mass = mass / 32

        B_factor = np.array(pdb_res['B_factor'].tolist()).reshape(-1, 1)
        if (max(B_factor) - min(B_factor)) == 0:
            B_factor = np.zeros(B_factor.shape) + 0.5
        else:
            B_factor = (B_factor - min(B_factor)) / (max(B_factor) - min(B_factor))

        is_sidechain = np.array(pdb_res['is_sidechain'].tolist()).reshape(-1, 1)
        charge = np.array(pdb_res['charge'].tolist()).reshape(-1, 1)
        num_H = np.array(pdb_res['num_H'].tolist()).reshape(-1, 1)
        ring = np.array(pdb_res['ring'].tolist()).reshape(-1, 1)

        atom_type = pdb_res['atom_type'].tolist()
        atom_vander = np.zeros((len(atom_type), 1))
        for i, type in enumerate(atom_type):
            try:
                atom_vander[i] = atom_vander_dict[type]
            except:
                atom_vander[i] = atom_vander_dict['C']

        atom_feas = [mass, B_factor, is_sidechain, charge, num_H, ring, atom_vander]
        atom_feas = np.concatenate(atom_feas,axis=1)

        res_atom_feas = []
        atom_begin = 0
        for i, res_id in enumerate(res_id_list):
            res_atom_df = pdb_res[pdb_res['res_id'] == res_id]
            atom_num = len(res_atom_df)
            res_atom_feas_i = atom_feas[atom_begin:atom_begin+atom_num]
            #在残基水平上做平均
            res_atom_feas_i = np.average(res_atom_feas_i,axis=0).reshape(1,-1)
            res_atom_feas.append(res_atom_feas_i)
            atom_begin += atom_num

        if not os.path.exists(data_path + 'Atom_feas/processed_atomfea'):
            os.makedirs(data_path + 'Atom_feas/processed_atomfea')

        np.save(data_path +'Atom_feas' + '/processed_atomfea/{}.npy'.format(seq_id), res_atom_feas)
    return

def matched_atomfea(seq_id, ref_seq, data_path):

    with open(data_path + 'Atom_feas' + '/{}.csv.pkl'.format(seq_id), 'rb') as f:
        tmp = pickle.load(f)
    pdb_sequence = tmp['sequence']

    atom_feature = np.load(data_path +'Atom_feas/processed_atomfea/' + seq_id + '.npy', allow_pickle=True)
    data_shape = atom_feature[0].shape

    alignments = pairwise2.align.globalxx(ref_seq, pdb_sequence)
    align_ref_seq = alignments[0].seqA
    align_seq = alignments[0].seqB

    new_atomfea = []
    for aa in align_seq:
        if aa == "-":
            new_atomfea.append(np.zeros(data_shape))
        else:
            new_atomfea.append(atom_feature[0])
            atom_feature = atom_feature[1:]

    matched_atomfea = []
    for i in range(len(align_ref_seq)):
        if align_ref_seq[i] == "-":
            continue
        matched_atomfea.append(new_atomfea[i])
    matched_atomfea = np.concatenate(matched_atomfea, axis=0)

    if not os.path.exists(data_path + 'Atom_feas/matched_atomfea'):
        os.makedirs(data_path + 'Atom_feas/matched_atomfea')
    np.save(data_path + 'Atom_feas/matched_atomfea/{}.npy'.format(seq_id), matched_atomfea)
    return

def seq_fea_generate(pro, seqence, data_path):
    all_for_assign = np.loadtxt('./Dataset/training_data/seqfea/all_assign.txt')
    xx = seqence
    x_p = np.zeros((len(xx), 7))
    for j in range(len(xx)):
        if restype_1to3[xx[j]] == 'ALA':
            x_p[j] = all_for_assign[0,:]
        elif restype_1to3[xx[j]] == 'CYS':
            x_p[j] = all_for_assign[1,:]
        elif restype_1to3[xx[j]] == 'ASP':
            x_p[j] = all_for_assign[2,:]
        elif restype_1to3[xx[j]] == 'GLU':
            x_p[j] = all_for_assign[3,:]
        elif restype_1to3[xx[j]] == 'PHE':
            x_p[j] = all_for_assign[4,:]
        elif restype_1to3[xx[j]] == 'GLY':
            x_p[j] = all_for_assign[5,:]
        elif restype_1to3[xx[j]] == 'HIS':
            x_p[j] = all_for_assign[6,:]
        elif restype_1to3[xx[j]] == 'ILE':
            x_p[j] = all_for_assign[7,:]
        elif restype_1to3[xx[j]] == 'LYS':
            x_p[j] = all_for_assign[8,:]
        elif restype_1to3[xx[j]] == 'LEU':
            x_p[j] = all_for_assign[9,:]
        elif restype_1to3[xx[j]] == 'MET':
            x_p[j] = all_for_assign[10,:]
        elif restype_1to3[xx[j]] == 'ASN':
            x_p[j] = all_for_assign[11,:]
        elif restype_1to3[xx[j]] == 'PRO':
            x_p[j] = all_for_assign[12,:]
        elif restype_1to3[xx[j]] == 'GLN':
            x_p[j] = all_for_assign[13,:]
        elif restype_1to3[xx[j]] == 'ARG':
            x_p[j] = all_for_assign[14,:]
        elif restype_1to3[xx[j]] == 'SER':
            x_p[j] = all_for_assign[15,:]
        elif restype_1to3[xx[j]] == 'THR':
            x_p[j] = all_for_assign[16,:]
        elif restype_1to3[xx[j]] == 'VAL':
            x_p[j] = all_for_assign[17,:]
        elif restype_1to3[xx[j]] == 'TRP':
            x_p[j] = all_for_assign[18,:]
        elif restype_1to3[xx[j]] == 'TYR':
            x_p[j] = all_for_assign[19,:]
    if not os.path.exists(data_path + 'seqfea/'):
        os.makedirs(data_path + 'seqfea/')
    np.save(data_path + 'seqfea/{}.npy'.format(pro), x_p)

def SaveChainPDB(protein_list,data_path):
    if not os.path.exists(data_path + 'SavechainPDB'):
        os.makedirs(data_path + 'SavechainPDB')

    for prot in protein_list:
        if not os.path.exists(data_path +'SavechainPDB/{}.pdb'.format(prot)):
            PDBID = prot.split("-")[0]
            chain_id = prot.split("-")[1]

            pdbgz_file = data_path + "{}.pdb.gz".format(PDBID)
            rnd_fn = "".join([secrets.token_hex(10), '.pdb'])
            with gzip.open(pdbgz_file, 'rb') as f, open(rnd_fn, 'w') as out:
                out.write(f.read().decode())

            with open(rnd_fn,'r') as f:
                pdb_text = f.readlines()
            text = []

            if chain_id == '_':
                chainid_list = set()
                for line in pdb_text:
                    if line.startswith('ATOM'):
                        chainid_list.add(line[21])
                chainid_list = list(chainid_list)
                if len(chainid_list) == 1:
                    chain_id = chainid_list[0]
                    print('Chain: Your query structure has specific chain',prot,chain_id)
                else:
                    print('ERROR: Your query structure has multiple chains, please input the chain ID!',prot)
                    continue

            for line in pdb_text:
                if line.startswith('ATOM') and line[21] == chain_id:
                    text.append(line)
                if line.startswith('TER') and line[21] == chain_id:
                    break
            text.append('\nTER\n')
            text.append('END\n')

            with open(data_path +'SavechainPDB/{}.pdb'.format(prot), 'w') as f:
                f.writelines(text)
    return

def download_pdb(protein_list,data_path):
    for prot in protein_list:
        PDBID = prot.split("-")[0]
        if os.path.exists(data_path + "{}.pdb.gz".format(PDBID))== False:
            print("downloading PDB", PDBID)
            os.system("wget -P {} http://www.rcsb.org/pdb/files/{}.pdb.gz".format(data_path, PDBID))
            if os.path.exists(data_path + "{}.pdb.gz".format(PDBID)) == False:
                print("PDB not exist")



def evo_embedding(data_dir):

    if not os.path.exists(data_dir + 'Prot5'):
        os.makedirs(data_dir + 'Prot5')
    model, tokenizer = get_T5_model()

    # Load example fasta.
    seqs = read_fasta(data_dir +'fasta/allseq.fa')
    # Compute embeddings and/or secondary structure predictions
    results = get_embeddings(model, tokenizer, seqs, True, True)
    # Store per-residue embeddings
    save_embeddings(results["residue_embs"], data_dir + 'Prot5/train_per_residue_embeddings.h5')
    save_embeddings(results["protein_embs"], data_dir +'Prot5/train_per_protein_embeddings.h5')
    print("done")



if __name__ == '__main__':

    data_dir = "./Dataset/training_data/"
    protein_list = []
    f = open(data_dir + "training_id_withEC.txt", "r")
    filedata = f.readlines()
    for line in filedata:
        protein = line.strip()
        pdb_id = protein[:4].lower() + "-" + protein[5].upper()
        protein_list.append(pdb_id)
    f.close()

    prot_seq = {}
    prot_anno = {}
    f = open(data_dir + "training_label.txt", "r")
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

    #---------write enzyme fasta--------------
    write_all_fasta(data_dir,protein_list,prot_seq)
    Write_single_fasta(data_dir + "training_label.txt", data_dir)

    #---------write specific chain PDB
    download_pdb(protein_list, data_dir + "PDB/")
    SaveChainPDB(protein_list,data_dir + "PDB/")

    #---------process CA contact map
    nprocs = 20
    import multiprocessing
    nprocs = np.minimum(nprocs, multiprocessing.cpu_count())
    if nprocs > 4:
        pool = multiprocessing.Pool(processes=nprocs)
        pool.map(partial(write_contact_map, prot2seq=prot_seq, data_path =data_dir),  protein_list)  # partial将函数和参数封装到一个指定变量名中,下次执行直接调用
    else:
        for prot in protein_list:
            write_contact_map(prot, prot2seq=prot_seq, data_path =data_dir)

    #---------generate PSSM/HHM/DSSP file
    import multiprocessing
    nprocs = 20
    nprocs = np.minimum(nprocs, multiprocessing.cpu_count())
    if nprocs > 4:
        pool = multiprocessing.Pool(processes=nprocs)
        pool.map(partial(Multi_process_PSSM_generation,data_path =data_dir), protein_list)
    else:
        for prot in protein_list:
            Multi_process_PSSM_generation(prot,data_path =data_dir)

    #---------extract pssm/hhm/dssp feature
    cal_PSSM(protein_list,  data_dir + 'pssm/')
    cal_HMM(protein_list,  data_dir + 'hhm/')
    cal_DSSP(protein_list, data_dir + 'dssp/')
    for seqid in protein_list:
        matched_dssp(seqid, prot_seq[seqid], data_dir + "PDB/SavechainPDB/", data_dir + 'dssp/')

    # ---------extract atom feature
    if nprocs > 4:
        pool = multiprocessing.Pool(processes=nprocs)
        pool.map(partial(get_pdb_DF, data_path=data_dir), protein_list)
    else:
        for prot in protein_list:
            get_pdb_DF(prot, data_path = data_dir)

    cal_atomFea(protein_list, data_dir)

    for seqid in protein_list:
        matched_atomfea(seqid, prot_seq[seqid], data_dir)
        seq_fea_generate(seqid,prot_seq[seqid],data_dir)

    # ---------extract protein residual level embedding using ProtT5
    evo_embedding(data_dir)

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!finish!!!!!!!!!!!!!")

