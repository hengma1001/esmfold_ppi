import glob

import pandas as pd
import torch
import yaml
from tqdm import tqdm


def batcher(iterable, n=1):
    len_iter = len(iterable)
    for ndx in range(0, len_iter, n):
        yield iterable[ndx : min(ndx + n, len_iter)]


model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

batch_converter = alphabet.get_batch_converter()
model.eval().cuda()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
yml_files = glob.glob(
    "/homes/heng.ma/Research/md_pkgs/esmfold_ppi/examples/run_r1/esmfold/data/*.yml"
)
batch_size = 50

df = []
for yml_batch in tqdm(
    batcher(yml_files, batch_size), total=len(yml_files) // batch_size + 1
):
    yml_dicts = [yaml.safe_load(open(yml, "r")) for yml in yml_batch]
    data = [(protein["id"], protein["sequence"]) for protein in yml_dicts]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    for i, tokens_len in enumerate(batch_lens):
        seq_rep = token_representations[i, 1 : tokens_len - 1].mean(0)
        local_dict = yml_dicts[i].copy()
        local_dict["embedding"] = list(seq_rep.cpu().numpy())
        df.append(local_dict)

df = pd.DataFrame(df)
df.to_pickle("esm_emb.pkl")
