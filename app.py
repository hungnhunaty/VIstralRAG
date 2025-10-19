import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from  langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# --- 1. CẤU HÌNH ---
# (Đây là những thứ bạn có thể thay đổi)

# Tên file PDF chứa kiến thức HUTECH
pdf_file_path = "data/pdfdb.pdf"  

# Thư mục để lưu "bộ nhớ" (VectorDB)
persist_directory = "./chroma_db"  

# Tên "máy số hóa" (Embedding model)
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Tên "bộ não" (LLM model) bạn đã tải
ollama_model_name = "ontocord/vistral" 

# --- 2. KHỞI TẠO "MÁY SỐ HÓA" (Embedding) ---
# Tải "máy số hóa" về. Nó sẽ tự động tải lần đầu.
print("Đang khởi tạo 'máy số hóa' (Embedding model)...")
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

# --- 3. NẠP HOẶC TẠO "BỘ NHỚ" (Vector Database) ---
# (Đây là phần RAG: Retrieval)

vectorstore = None
if os.path.exists(persist_directory):
    # NẾU ĐÃ CÓ "bộ nhớ" (từ lần chạy trước) -> Tải lên cho nhanh
    print(f"Đang tải 'bộ nhớ' (VectorDB) từ thư mục: {persist_directory}...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    print("Tải 'bộ nhớ' thành công!")

else:
    # NẾU LÀ LẦN ĐẦU CHẠY -> Xử lý file PDF để tạo "bộ nhớ"
    print(f"Không tìm thấy 'bộ nhớ'. Bắt đầu xử lý file: {pdf_file_path}...")
    
    if not os.path.exists(pdf_file_path):
        print(f"LỖI: Không tìm thấy file '{pdf_file_path}'.")
        print("Vui lòng tải file PDF của HUTECH và đặt vào thư mục D:\\Vistral")
        exit()

    # Bước 3.1: Đọc file PDF
        
    print("Đang đọc file tài liệu...")

    if pdf_file_path.endswith(".pdf"):
        loader = PyPDFLoader(pdf_file_path)
    elif pdf_file_path.endswith(".docx"):
        loader = Docx2txtLoader(pdf_file_path)
    else:
        print("Lỗi: chỉ hỗ trợ file .pdf hoặc .docx")
        exit()

    documents = loader.load()


    # Bước 3.2: Cắt nhỏ tài liệu
    # (Giống như xé 1 cuốn sách thành từng đoạn văn nhỏ cho dễ tìm)
    print("Đang cắt nhỏ tài liệu...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Bước 3.3: "Số hóa" và Lưu vào "bộ nhớ" (ChromaDB)
    # Đây là bước tốn thời gian nhất (chỉ trong lần chạy đầu)
    print(f"Đang 'số hóa' {len(texts)} đoạn văn bản (có thể mất vài phút)...")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embedding_function,
        persist_directory=persist_directory  # Tự động lưu vào thư mục này
    )
    print(f"Đã tạo và lưu 'bộ nhớ' vào thư mục: {persist_directory}")

print("--- 'Bộ nhớ' Chatbot đã sẵn sàng ---")

# --- 4. KẾT NỐI "BỘ NÃO" (LLM) ---
# (Đây là phần RAG: Generation)
print(f"Đang kết nối tới 'bộ não' Ollama (model: {ollama_model_name})...")
try:
    llm = OllamaLLM(model=ollama_model_name)
    # Gửi một tin nhắn test nhỏ để đảm bảo Ollama hoạt động
    llm.invoke("test") 
    print("Kết nối 'bộ não' thành công!")
except Exception as e:
    print("\n*** LỖI KẾT NỐI OLLAMA ***")
    print("Hãy đảm bảo bạn đã làm 2 việc sau:")
    print(f"  1. Ứng dụng Ollama (biểu tượng Llama) đang chạy ở Taskbar.")
    print(f"  2. Bạn đã chạy lệnh: ollama run {ollama_model_name}")
    exit()

# --- 5. TẠO "PROMPT TEMPLATE" (Kịch bản cho Bot) ---
# Đây là "linh hồn" của con bot, ra lệnh cho nó phải làm gì.
prompt_template = """
Bạn là trợ lý AI của trường Đại học HUTECH, tên là 'HUTECH AI Helper'.
Nhiệm vụ của bạn là trả lời các câu hỏi của sinh viên DỰA TRÊN NGỮ CẢNH (context) được cung cấp.
Hãy trả lời một cách thân thiện, chuyên nghiệp, chính xác và chỉ sử dụng thông tin từ ngữ cảnh.
Nếu ngữ cảnh không chứa thông tin để trả lời, hãy trả lời: "Xin lỗi, mình không tìm thấy thông tin này trong tài liệu của trường."
Tuyệt đối không tự bịa đặt thông tin.

Ngữ cảnh (Context - Dữ liệu tìm thấy từ file PDF):
{context}

Câu hỏi của sinh viên:
{input}

Trả lời (chỉ bằng Tiếng Việt):
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


# --- 6. TẠO "CHUỖI RAG" (Kết hợp Tìm kiếm & Trả lời) ---
print("Đang tạo chuỗi RAG (Retrieval-Augmented Generation)...")

# Tạo "Công cụ tìm kiếm" (Retriever)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Lấy 5 đoạn văn liên quan nhất

# Tạo chuỗi "Stuff Documents": Nhồi các đoạn văn tìm được vào prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Tạo chuỗi RAG cuối cùng:
# 1. Lấy câu hỏi -> 2. Dùng retriever tìm kiếm -> 3. Nhồi vào prompt -> 4. LLM trả lời
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("\n=============================================")
print("  Chatbot HUTECH đã sẵn sàng! (gõ 'thoát' để dừng)  ")
print("=============================================\n")


# --- 7. VÒNG LẶP HỎI ĐÁP ---
while True:
    try:
        query = input("Bạn hỏi: ")
        if query.lower().strip() == "thoát":
            break
        if not query.strip():
            continue

        print("\nChatbot HUTECH:")
        
        # Chạy chuỗi RAG
        response_stream = rag_chain.stream({"input": query})

        # In câu trả lời ra màn hình (kiểu streaming như ChatGPT)
        for chunk in response_stream:
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
        
        print("\n") 

    except KeyboardInterrupt:
        print("\nĐã dừng chatbot.")
        break
    except Exception as e:
        print(f"\nĐã xảy ra lỗi: {e}")
        break