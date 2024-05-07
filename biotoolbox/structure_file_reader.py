import json
import tempfile
import os

import Bio
from Bio import SeqIO
import secrets
import gzip
from Bio.Data.SCOPData import protein_letters_3to1


class PdbSeqResDataParser:
    def __init__(self, handle, parser_mode, chain_name, verbose=False):
        self.seq_res_seqs = []
        self.idx_to_chain = {}
        self.chain_count = 0
        # self.chain_ids = []
        for record in SeqIO.parse(handle, f"{parser_mode}-seqres"):
            if verbose:
                print("Record id %s, chain %s, len %s" % (record.id, record.annotations["chain"], len(record.seq)))
                #print(record.dbxrefs)
                print(record.seq)
            #if record.annotations['chain'] == chain_name:
            self.seq_res_seqs.append(record.seq)
            self.idx_to_chain[self.chain_count] = record.annotations['chain']
                # self.chain_ids.append(record.name)
            self.chain_count += 1
                #break

    def has_seq_res_data(self):
        return self.chain_count > 0


class PdbAtomDataParser:
    def __init__(self, handle, parser_mode, chain_name, verbose=False):
        self.idx_to_chain = {}
        self.chain_to_idx = {}
        self.atom_seqs = []
        self.chain_count = 0

        for record in SeqIO.parse(handle, f"{parser_mode}-atom"):
            if verbose:
                print("Record id %s, chain %s len %s" % (record.id, record.annotations["chain"], len(record.seq)))
                print(record.seq)
            #if record.annotations['chain'] == chain_name:
            self.atom_seqs.append(record.seq)
            self.idx_to_chain[self.chain_count] = record.annotations['chain']
            self.chain_to_idx[record.annotations['chain']] = self.chain_count
            self.chain_count += 1
                #break



class StructureContainer:
    def __init__(self):
        self.structure = None
        self.chains = {}
        self.id_code = None

    def with_id_code(self, id_code): #eg:5O9Z
        self.id_code = id_code
        return self

    def with_structure(self, structure):
        self.structure = structure
        return self

    def with_chain(self, chain_name, seqres_seq, atom_seq):
        chain_info = {'seqres-seq': seqres_seq, 'atom-seq': atom_seq}
        if seqres_seq is not None:
            chain_info['seq'] = seqres_seq
        else:
            chain_info['seq'] = atom_seq
        self.chains[chain_name] = chain_info
        return self

    def with_seqres(self,seqres_seq):
        for chain_name in self.chains:
            self.chains[chain_name]['seqres-seq'] = seqres_seq
        return self

    def toJSON(self):
        result = {'chain_info': self.chains, 'id_code': self.id_code}
        return json.dumps(result, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4, skipkeys=True)

def build_structure_container_for_pdb(structure_data, chain_name):
    # Test the data to see if this looks like a PDB or an mmCIF
    #0615 pan
    """rnd_fn = "".join([secrets.token_hex(10), '.pdb'])
    with gzip.open(pdbfile, 'rb') as f, open(rnd_fn, 'w') as out:
        out.write(f.read().decode())"""
        # os.remove(rnd_fn)
    parser_mode = 'pdb'

    """tester = re.compile('^_', re.MULTILINE) #正则表达式 多行匹配 ^限定开头
    if len(tester.findall(structure_data)) == 0:
        parser_mode = 'pdb'
    else:
        parser_mode = 'cif'
    #print("parser_mode is",parser_mode)"""

    temp = tempfile.TemporaryFile(mode='w+') #创建临时文件
    #with gzip.open(pdbfile, 'rb') as f, open(temp, 'w') as out:
        #out.write(f.read().decode())
    temp.write(str(structure_data))
    temp.flush()
    temp.seek(0, 0)#从头读取

    container_builder = StructureContainer()
    #{chain_count:1,idx_to_chain:{0:'R'},seq_res_seqs:['MPERDSE...']}
    seq_res_info = PdbSeqResDataParser(temp, parser_mode, chain_name)
    temp.seek(0, 0)
    try:
        #{atom_seqs:[EFKTRLGRNVYRMLFK],chain_count:1,chain_to_index:{R:0},index_to_chain:{0:R}}
        atom_info = PdbAtomDataParser(temp, parser_mode, chain_name)
    except ValueError:
        # For some reason we literally just can't parse it...
        raise ValueError('Biopython doesn\'t know how to parse this PDB')
    """if len(seq_res_info.idx_to_chain) == len(atom_info.idx_to_chain) \
            and set(seq_res_info.idx_to_chain.values()) != set(atom_info.idx_to_chain.values()):
        print(f"WARNING: The IDs from the seqres lines don't match the IDs from the ATOM lines. This might not work.")
        raise Exception"""

    temp.seek(0, 0)
    if parser_mode == 'pdb':
        structure = Bio.PDB.PDBParser().get_structure('input', temp)
        #id_code = structure.header['idcode']
        #container_builder.with_id_code(id_code)
    elif parser_mode == 'cif':
        structure = Bio.PDB.MMCIFParser().get_structure('input', temp)
        # TODO(cchandler): See if there's something I can use in biopython to actually get this.
        # the default parser appears to do it the wrong way.
        id_code = None

    # model = structure[0]
    container_builder.with_structure(structure)

    if seq_res_info.has_seq_res_data():
        for i, seqres_seq in enumerate(seq_res_info.seq_res_seqs):
            chain_name_from_seqres = seq_res_info.idx_to_chain[i]
            try:
                chain_idx = atom_info.chain_to_idx[chain_name_from_seqres]
                atom_seq = atom_info.atom_seqs[chain_idx]
            except (IndexError, KeyError) as e:
                # This is the case where the number of SEQRES chain entries didn't match
                # the number of ATOM line entries. Usually len(seqres) > len(atom). Since we _did_
                # have SEQRES lines, they are being treated canonically.
                continue

            container_builder.with_chain(chain_name_from_seqres, seqres_seq, atom_seq)
    else:
        for i, atom_seq in enumerate(atom_info.atom_seqs):
            chain_name_from_seqres = atom_info.idx_to_chain[i]
            container_builder.with_chain(chain_name_from_seqres, None, atom_seq)

    temp.close()
    return container_builder



"""def pdbstartpos(structure_data,chain_name):
    start_pos = 1
    #pdbTraj = open(structure_data, 'r')
    pdbTraj_line = structure_data.readlines()
    for line in pdbTraj_line:
        if line.startswith("SEQADV"):
            chainType = line.strip().split()[3]
            pos = int(line.strip().split()[4])
            if chainType == chain_name:
                start_pos = pos
                break
    return(start_pos)"""