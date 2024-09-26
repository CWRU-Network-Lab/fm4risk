# main neural search build -- nathaniel hahn

import build_ds
import torch
from unixcoder import UniXcoder
from docarray import BaseDoc
import os


# Assuming UniXcoder is your model class
model = UniXcoder("microsoft/unixcoder-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to GPU if available
print(f"Using device: {device}")


# codedoc class for each snippet
class CodeDoc(BaseDoc):
    code: str
    docstring: str
    language: str
    partition: str

#build each snippet and store data
def makeCodeDoc(code, comment, query):
    newDoc = CodeDoc(code = code, docstring = comment, language = "python", partition="custom")
    embed = compute(newDoc)
    embedL = [embed]
    score = search(embedL, query)

    return score

#pull pickle files
def get_data():
    if os.path.exists("full_py_unix.pkl"):
        return "placeholder"
    else:
        codesearchnet = build_ds.filter_dataset("full")
        docs = [CodeDoc(code=data['func_code_string'], docstring=data['func_documentation_string'], language=data['language'], partition=data['split_name']) for data in codesearchnet]

        return docs

## embed the tokens and manage cuda memory
def compute(code):
    token_ids = model.tokenize([code.docstring + " " + code.code], max_length=512, mode="<encoder-only>")
    source_ids = torch.tensor(token_ids).to(device)  # Move tensor to GPU
    with torch.no_grad():  
        tokens_embeddings, func_embedding = model(source_ids)
    del token_ids  # Delete token_ids to release memory
    del source_ids  # Delete source_ids to release memory
    return func_embedding

## embed the entire ds 
def tokenize_ds(dataset):
    embedded = []
    for i, code in enumerate(dataset):
        func_embedding = compute(code)
        embedded.append(func_embedding)
    return embedded


#Linear search for the best embeds
def search(docs, query):
    print("searching")
    tokens_ids = model.tokenize([query], max_length=512, mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)  # Move tensor to GPU
    tokens_embeddings, nl_embedding = model(source_ids)
    # Normalize embedding
    query_embedding = torch.nn.functional.normalize(nl_embedding, p=2, dim=1)
    scores = []
    for i, doc in enumerate(docs):
        docs_embedding = torch.nn.functional.normalize(doc, p=2, dim=1)
        max_func_nl_similarity = torch.nn.functional.cosine_similarity(query_embedding, docs_embedding)
        scores.append([max_func_nl_similarity, i])
    return scores

# return top n results 
def results(tensors, query, docs, top_n=1):
    print("Query: ", query)
    first_tensors = [item[0] for item in tensors]
    concatenated_tensor = torch.stack(first_tensors, dim=0)
    top_values, top_indices = torch.topk(concatenated_tensor, k=top_n, dim=0)
    result = []
    for i in range(top_n):
        tupleElem = (docs[top_indices[i]].code, docs[top_indices[i]].docstring, str(top_values[i].cpu().detach().numpy()[0]))
        result.append(tupleElem)

    return result

# determine the rank of tensors
def getRank(tensors, sim_score):
    first_tensors = [item[0] for item in tensors]
    tensor_stack = torch.stack(first_tensors, dim=0)
    tensor_list = []
    for item in tensor_stack:
        tensor_list.append(item.cpu().detach().numpy()[0])
    rev = sorted(tensor_list)
    forward = sorted(tensor_list, reverse=True)
    tval = sim_score[0][0].cpu().detach().numpy()[0]
    i = 0
    while tval < forward[i]:
        i+=1
    return i+ 1
