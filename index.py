import os
from typing import List
import olefile
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline as hf_pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
import torch
import uvicorn

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# 모델 및 벡터 스토어 초기화
MODEL_NAME = "sh2orc/Llama-3.1-Korean-8B-Instruct"

# 모델 설정
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
hf = HuggingFacePipeline(pipeline=pipe, model_id=MODEL_NAME, batch_size=8)

# 임베딩 모델 초기화
local_embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})

# 데이터 로딩 함수
def load_csv_files_from_folder(folder_path: str) -> List[str]:
    texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                # EUC-KR 인코딩으로 CSV 파일 읽기 시도\
                df = pd.read_csv(file_path, encoding='euc-kr')
                for column in df.columns:
                    texts.extend(df[column].dropna().astype(str).tolist())
            except Exception as e:
                try:
                    # EUC-KR로 실패하면 CP949로 시도
                    df = pd.read_csv(file_path, encoding='cp949')
                    for column in df.columns:
                        texts.extend(df[column].dropna().astype(str).tolist())
                except Exception as e:
                    print(f"CSV 파일 처리 중 오류 발생: {file_path}, 오류: {e}")
    return texts

# 문서 로딩
data_folder = "data"  # CSV 파일이 저장된 폴더
csv_texts = load_csv_files_from_folder(data_folder)

texts = csv_texts

# 텍스트 분할
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(separator="\n", chunk_size=600, chunk_overlap=100)
docs = text_splitter.create_documents(texts)

# 벡터 스토어 생성 - 배치 크기 제한으로 인해 청크로 나누어 처리
BATCH_SIZE = 5000  # 최대 배치 크기보다 작게 설정
for i in range(0, len(docs), BATCH_SIZE):
    batch_docs = docs[i:i + BATCH_SIZE]
    if i == 0:
        vectorstore = Chroma.from_documents(documents=batch_docs, embedding=local_embeddings)
    else:
        vectorstore.add_documents(documents=batch_docs)

# API 라우팅
@app.get("/")
async def root():
    return {"message": "API 서버가 실행 중입니다."}

@app.post("/query")
async def query(question: str):
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1})
    context = retriever.get_relevant_documents(question)

    if not context:
        raise HTTPException(status_code=404, detail="대답을 찾을 수 없습니다.")

    # 컨텍스트 기반 응답 생성
    context_text = "\n".join([doc.page_content for doc in context])
    input_text = f"컨텍스트: {context_text}\n질문: {question}"
    response = pipe(input_text)

    return {"question": question, "answer": response[0]["generated_text"]}

# 실행 명령
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
