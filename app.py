import streamlit as st
import os 
#from PyPDF2 import PdfReader
#import pypdf
import chromadb
from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain import document_loaders
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain import HuggingFaceHub
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
import tempfile

st.set_page_config(page_title="✨ PDF AI Chat Assistant - Open Source Version", layout="wide")
st.subheader("✨ Welcome to PDF AI Chat Assistant - Life Enhancing with AI.")
st.write("Important notice: This Open PDF AI Chat Assistant is offered for information and study purpose only and by no means for any other use. Any user should never interact with the AI Assistant in any way that is against any related promulgated regulations. The user is the only entity responsible for interactions taken between the user and the AI Chat Assistant.")

css_file = "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)   

print(f"定义处理多余的Context文本的函数")
def remove_context(text):
    # 检查 'Context:' 是否存在
    if 'Context:' in text:
        # 找到第一个 '\n\n' 的位置
        end_of_context = text.find('\n\n')
        # 删除 'Context:' 到第一个 '\n\n' 之间的部分
        return text[end_of_context + 2:]  # '+2' 是为了跳过两个换行符
    else:
        # 如果 'Context:' 不存在，返回原始文本
        return text
print(f"处理多余的Context文本函数定义结束")        

#model_id = "gpt-3.5-turbo"
#llm=ChatOpenAI(model_name = model_id, temperature=0.1)

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#model_id = os.getenv('model_id')
#hf_token = os.getenv('hf_token')
repo_id = os.getenv('repo_id')

llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":512,
                                   "max_new_tokens":1024, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})

#prompt_template = """
#You are a very helpful AI assistant who is expert in intellectual property industry. Please ONLY use {context} to answer the user's question. If you don't know the answer, just say that you don't know. DON'T try to make up an answer.
#Your response should be full and detailed.
#Question: {question}
#Helpful AI Repsonse:
#"""
#PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT)

loaders = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

if "index" not in st.session_state:
    st.session_state.index = ""        

with st.sidebar:
    st.subheader("Upload your Documents Here: ")
    pdf_files = st.file_uploader("Choose your PDF Files and Press OK", type=['pdf'], accept_multiple_files=True)
    print("文件上传完毕")   #Streamlit程序中有任何操作，都会执行到这一步，但是下面由于有st.button需要用户触发才会执行，所以其中的代码就不会自动执行
    if st.button('Process to AI Chat'):
        with st.spinner("Processing your PDF file..."):    
            for pdf_file in pdf_files:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(pdf_file.read())
                    temp_file_path = temp_file.name
                loaders.append(PyPDFLoader(temp_file_path))   #这个代码决定了只可以处理pdf文件
                print("文件load完毕")
            try:
                print("开始制作index")
                #index = VectorstoreIndexCreator().from_loaders(loaders)        
                #index = VectorstoreIndexCreator(embedding = HuggingFaceEmbeddings()).from_loaders(loaders)
                #index = VectorstoreIndexCreator(embedding = embeddings).from_loaders(loaders)
        
                #https://medium.com/@ynotfriedman2/automatic-loader-for-any-document-in-langchain-f9f8f798da0e
                #index = VectorstoreIndexCreator(embedding = HuggingFaceEmbeddings()).from_loaders([loaders])
                st.session_state.index = VectorstoreIndexCreator(embedding=st.session_state.embeddings).from_loaders(loaders)
        #ImportError: Could not import chromadb python package. Please install it with `pip install chromadb`.        
                #https://www.akshaymakes.com/blogs/youtube-gpt        
                #print("index制作完毕")
                #index.save("./docchatbotindex")
                #print("index本地保存")
                os.remove(temp_file_path)
                print("删除临时文件")
            except Exception as e:
                st.write("Unknown error. Please try again.")  

# Handle the error, e.g., print an error message or return a default text
#    st.write("Documents not uploaded. Please upload your docs first and then enter your question.")
    #os.environ["OPENAI_API_KEY"] = str(openai.api_key)

#loaders = PyPDFLoader(pdf_files)
#index = VectorstoreIndexCreator().from_loaders([loaders])
prompt = st.text_input("Enter your question & query your PDF file:")
print("用户问题输入完毕")
#if st.session_state.user_question !="" and not st.session_state.user_question.strip().isspace() and not st.session_state.user_question == "" and not st.session_state.user_question.strip() == "" and not st.session_state.user_question.isspace():
if prompt !="" and not prompt.strip().isspace() and not prompt == "" and not prompt.strip() == "" and not prompt.isspace():
    with st.spinner("AI Working...Please wait a while to Cheers!"):   
        try:
            response = st.session_state.index.query(llm=llm, question = prompt, chain_type = 'stuff')   
            # stuff chain type sends all the relevant text chunks from the document to LLM    
            print("index问答完毕")
            cleaned_initial_ai_response = remove_context(response)
            print("调用remove_context函数对['output_text']进行处理之后的输出结果: ")
            print(cleaned_initial_ai_response)
            print()             
            final_ai_response = cleaned_initial_ai_response.partition('<|end|>')[0].strip().replace('\n\n', '\n').replace('<|end|>', '').replace('<|user|>', '').replace('<|system|>', '').replace('<|assistant|>', '')
            new_final_ai_response = final_ai_response.split('Unhelpful Answer:')[0].strip()
            new_final_ai_response = new_final_ai_response.split('Note:')[0].strip()
            new_final_ai_response = new_final_ai_response.split('Please provide feedback on how to improve the chatbot.')[0].strip()  
            
            print("Final AI Response:")
            print(new_final_ai_response)
            
            st.write("AI Response:")
            st.write(new_final_ai_response)     
            
        # Write the results from the LLM to the UI
        #    st.write("<br><i>" + response + "</i><hr>", unsafe_allow_html=True )
        #st.write("<b>" + prompt + "</b><br><i>" + response + "</i><hr>", unsafe_allow_html=True )
        except Exception as e:
    # Handle the error, e.g., print an error message or return a default text            
            print("Documents not uploaded or Unknown error.")
