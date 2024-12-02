import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from huggingface_hub import notebook_login
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
import textwrap
import unstructured
import sys
import torch
from torch.utils._pytree import register_pytree_node
import subprocess
import pyterrier as pt
import pandas as pd
from transformers import pipeline

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

app = FastAPI()

pt.init()

#from huggingface_hub import notebook_login 
#os.environ['HuggingFaceHub_API_Token']= 'Your Huggingface User Access Token'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

chat_history = []

new_model_path = "Llama-2-13b-chat-hf"

quantization_config=BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(new_model_path,
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             quantization_config=quantization_config
                                             )

tokenizer = AutoTokenizer.from_pretrained(new_model_path)

print("Model are created!")

pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 1024,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

text_data = []
Doc_Names = ["Fengyun-2D", "Fengyun-2E", "Fengyun-2F", "Fengyun-2H", "Fengyun-4A",
    "Sentinel-3A", "Sentinel-3B", "Sentinel-6A", "Jason-1", "Jason-2", "Jason-3",
    "CryoSat-2", "Haiyang-2A", "TOPEX", "Summary", "SARAL"
]

i = 0
for sat in Doc_Names:
    file_name = f'{sat}.txt'
    with open(file_name, 'r') as file:
      file_text = file.read()
      text_data.append({'docno': i, 'title':sat, 'text': file_text})
    i = i+1

df1 = pd.DataFrame(text_data)




directory_path = './pd_index'

subprocess.run(['rm', '-rf', directory_path])


df1["text"] = df1["text"].astype(str)
df1["title"] = df1["title"].astype(str)
df1["docno"] = df1["docno"].astype(str)




pd_indexer = pt.DFIndexer(index_path="./pd_index")


indexref = pd_indexer.index(df1["title"], df1["text"], df1["docno"])

index = pt.IndexFactory.of(indexref)

index = pt.IndexFactory.of('./pd_index/data.properties')

pd_indexer.setProperty("tokeniser", "UTFTokeniser")
pd_indexer.setProperty( "termpipelines", "Stopwords,PorterStemmer")

data1 = []
for row in df1.itertuples():
    data1.append({'docno':row.docno, 'title': row.title, 'body': row.text})
iter_indexer = pt.IterDictIndexer("./pd_index", meta={'docno': 20, 'title': 10000, 'body':100000},
overwrite=True, blocks=True)
RETRIEVAL_FIELDS = ['title','body']
indexref1 = iter_indexer.index(data1, fields=RETRIEVAL_FIELDS)


import warnings

warnings.filterwarnings("ignore", message="Created a chunk of size .*")

default_sat_name = 'Summary'
def QAbot(query, chat_history):
        query = query.replace('?','')
        qe = (pt.rewrite.SequentialDependence() >> 
        pt.BatchRetrieve(indexref1, wmodel="BM25"))     
        list = ["Fengyun", "FENGYUN", "fengyun", "Sentinel", "SENTINEL", "sentinel", "Jason", "JASON", "jason", "CryoSat", "Cryosat",  "CRYOSAT", "cryosat", "Haiyang", "HAIYANG", "haiyang", "Topex", "TOPEX",
        "topex", "Saral", "SARAL", "saral"]
        res = [ele for ele in list if(ele in query)]
        if len(chat_history) > 0:
                sat_name = chat_history[-1][2]
        else:
                sat_name = default_sat_name
        if len(res) == 0:
                query = f"{query} {sat_name}"
        result = qe.search(query)
        try:
                title = df1["title"][int(result["docno"][0])]
        except KeyError:
                title = "Title not found"
    
        print(title)
        
        loader = UnstructuredFileLoader(str(title)+".txt")
        documents = loader.load()

        text_splitter=CharacterTextSplitter(separator='\n',
                                    chunk_size=1500,
                                    chunk_overlap=50)
        text_chunks=text_splitter.split_documents(documents)


        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cuda'})
        
        llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0.5})
        vectorstore = FAISS.from_documents(text_chunks, embeddings)
        chain =  ConversationalRetrievalChain.from_llm(llm=llm, chain_type = "stuff",return_source_documents=True, retriever=vectorstore.as_retriever(), get_chat_history = None)
        result=chain.invoke({"question": query, "chat_history": chat_history}, return_only_outputs=True)
        wrapped_text = textwrap.fill(result['answer'], width=500)
        chat_history.append((query, wrapped_text.split('Helpful Answer:')[-1], title))
        if len(chat_history) > 4:
                chat_history = chat_history[-4:]
        return wrapped_text.split('Helpful Answer:')[-1]
        
print("QAbot created!")

@app.post("/input_query/")
async def input_query(request: Request):
        data = await request.json()
        query = data.get("query")
        answer = QAbot(query, chat_history)
        
        return {"answer": answer}

@app.get("/", response_class=HTMLResponse)
async def main():
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
                <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=2.0">
        <title>Input query</title>
    </head>
    <body>
        <form id="queryForm" method="post">
            <label for="query">Enter your query:</label>
            <input type="query" id="query" name="query" required>
            <button type="submit">Submit</button>
        </form>

        <div id="response"></div>

        <script>
            document.getElementById("queryForm").addEventListener("submit", async function(event) {
                event.preventDefault();
                const formData = new FormData(event.target);
                const jsonData = {};
                formData.forEach((value, key) => {jsonData[key] = value});
                
                const response = await fetch('/input_query/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(jsonData)
                });
                
                const responseData = await response.json();
                document.getElementById("response").innerText = `\n Answer: ${responseData.answer}`;
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="Your IP address", port=your host)