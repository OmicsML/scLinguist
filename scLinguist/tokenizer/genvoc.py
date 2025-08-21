import json

from gene_tokenizer import GeneVocab

# get gene list

# adata = sc.read('/Users/ericmac/scGPT/data/cellxgene/datasets/partition_16.h5ad')
# print(adata.obs)
# print(adata.layers)
# print(adata.var)
# adata.var_names = adata.var.feature_id
# print(adata[0].X.tolil())
# print(type(adata.obs['assay']))
# tmp = adata.X.tolil()
# print(tmp)
# ing = list(adata.var['feature_id'])
with open('../../allcol.txt', 'r')as f:
    lines = f.readlines()
lines = lines[0]
import ast
after7 = list(ast.literal_eval(lines))
after7.remove('tech')
after7.append('tech')
GV = GeneVocab(gene_list_or_vocab=after7)
print(GV.save_json('./vocab.json'))
with open('vocab.json', "r") as f:
    tmp = json.load(f)
inv_map = {int(v): k for k, v in tmp.items()}
print(inv_map)
with open('reverse_vocab.json', "w") as f:
    json.dump(inv_map, f, indent=2)
# with open('./fully_vocab.json', "w") as f:
#     json.dump(names, f, indent=2)
# gene_order_path = '/Users/ericmac/scGPT/data/cellxgene/datasets/gene_order.csv'
# df = pd.read_csv(gene_order_path, index_col=0)
# ing = df['0'].tolist()
# GV = GeneVocab(gene_list_or_vocab=ing)
# print(GV.save_json('./order_vocab.json'))
# print(adata.var.feature_id[3])
#
# pdata = pd.read_parquet('/Users/ericmac/scGPT/data/cellxgene/databanks/partition_0.scb/X.datatable.parquet',
#                      engine='pyarrow')
# pdata = pq.ParquetFile('/Users/ericmac/scGPT/data/cellxgene/databanks/partition_0.scb/X.datatable.parquet')
# a = pdata.read_row_groups([0])
# pgenes = pdata.loc[12, 'genes']
# print(pgenes)
# pexps = pdata.loc[file_index, 'expressions']
# # print(pexps[pexps!=0].tolist())
# # print(list(pexps))
# data = np.zeros(60664)
# data[pgenes] = pexps
# # print(self.gene_order.tolist())
# # data = data[self.gene_order]
# for idx, item in enumerate(pdata.iter_row_groups()):
#     a = pd.DataFrame(item)
#     print(type(item))
#     print(a.info)
#     a = pd.DataFrame(item)
# #     print(a.info)
# print(pdata[0])
# print(pdata[1].to_pandas(filters=[[('id', '==', 1001)]]))
# print(a['genes'])
# print(a['genes'][0].as_py())
# b = a['expressions'][0].as_py()
# print(type(b))