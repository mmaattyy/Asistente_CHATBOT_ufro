import os, re, json, argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
from pypdf import PdfReader

def read_txt(path): 
    return open(path, "r", encoding="utf-8", errors="ignore").read()

def read_pdf(path):
    try:
        return "\n".join([p.extract_text() or "" for p in PdfReader(path).pages])
    except:
        return ""

def load_docs(data_dir):
    docs = []
    for root,_,files in os.walk(data_dir):
        for fn in files:
            path = os.path.join(root, fn)
            if fn.lower().endswith(".txt"):
                txt = read_txt(path)
            elif fn.lower().endswith(".pdf"):
                txt = read_pdf(path)
            else:
                continue
            # limpieza ligera
            txt = re.sub(r'\u00AD', '', txt)         # soft hyphen
            txt = re.sub(r'\s+\n', '\n', txt)
            txt = re.sub(r'\n{3,}', '\n\n', txt).strip()
            if txt:
                docs.append((path, txt))
    return docs

def chunk_by_tokens(text, tokenizer, chunk_size=400, overlap=70):
    ids = tokenizer.encode(text)
    out=[]; start=0; cid=0
    while start < len(ids):
        end=min(start+chunk_size,len(ids))
        sub=tokenizer.decode(ids[start:end])
        out.append((cid,start,end,sub))
        if end==len(ids): break
        start=end-overlap; cid+=1
    return out

def embed_texts(model, texts):
    return model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

def cmd_ingest(args):
    tok = tiktoken.get_encoding("cl100k_base")
    model = SentenceTransformer(args.model)
    docs = load_docs(args.data_dir)

    metas,texts=[],[]
    doc_id=0
    for path,txt in docs:
        for cid,s,e,sub in chunk_by_tokens(txt,tok,args.chunk_size,args.overlap):
            metas.append({"doc_id":doc_id,"path":path,"chunk_id":cid,"text":sub})
            texts.append(sub)
        doc_id+=1

    E = embed_texts(model,texts)
    os.makedirs(args.out_dir,exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir,"embeddings.npz"),E=E)
    with open(os.path.join(args.out_dir,"chunks.jsonl"),"w",encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m,ensure_ascii=False)+"\n")
    print("✅ Índice creado en",args.out_dir,
          f"({len(metas)} chunks, modelo {args.model})")

def cmd_query(args):
    arr=np.load(os.path.join(args.index_dir,"embeddings.npz"))
    E=arr["E"]
    metas=[json.loads(l) for l in open(os.path.join(args.index_dir,"chunks.jsonl"),encoding="utf-8")]
    model=SentenceTransformer(args.model)
    qv=embed_texts(model,[args.query])[0]
    sims=E@qv
    idx=np.argsort(-sims)[:args.k]
    for r,i in enumerate(idx,1):
        m=metas[int(i)]
        print(f"#{r} score={float(sims[i]):.4f} | {m['path']}")
        print(m['text'][:300].replace("\n"," "))
        print("-"*60)

def build_argparser():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers()

    pi=sub.add_parser("ingest")
    pi.add_argument("--data_dir",required=True)
    pi.add_argument("--out_dir",required=True)
    pi.add_argument("--chunk_size",type=int,default=400)   # 👈 nuevo default
    pi.add_argument("--overlap",type=int,default=70)       # 👈 nuevo default
    pi.add_argument("--model",type=str,default="sentence-transformers/all-MiniLM-L6-v2")
    pi.set_defaults(func=cmd_ingest)

    pq=sub.add_parser("query")
    pq.add_argument("--index_dir",required=True)
    pq.add_argument("--k",type=int,default=5)
    pq.add_argument("--query",required=True)
    pq.add_argument("--model",type=str,default="sentence-transformers/all-MiniLM-L6-v2")
    pq.set_defaults(func=cmd_query)
    return p

if __name__=="__main__":
    parser=build_argparser()
    args=parser.parse_args()
    if hasattr(args,"func"):
        args.func(args)
    else:
        parser.print_help()
