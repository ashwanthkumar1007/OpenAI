import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Streamlit UI setup
def main():
    """Main function to run the Streamlit app."""
    
    # Display the introductory information
    st.title("Ollama Document Retrieval System")
    st.markdown("""
        This app allows you to retrieve answers based on the content from a URL of documentation using the Ollama embedding model.
        Enter the URL of the documentation and ask a question to get a relevant answer.
    """)

    # URL input for the documentation
    url = st.text_input("Enter URL of the documentation:", "")

    if url:
        # Step 1: Load the documentation from the URL
        try:
            with st.spinner("Loading documents..."):
                web_loader = WebBaseLoader(url)
                docs = web_loader.load()
            
            # Split the loaded documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            documents = text_splitter.split_documents(docs)

            st.success("Documents loaded successfully!")
            
        except Exception as e:
            st.error(f"Error while loading documents: {e}")
            return

        # Step 2: Create a vector store from the documents
        try:
            with st.spinner("Creating vector store..."):
                embeddings = OllamaEmbeddings(model="gemma2:2b")
                vectorstore_db = FAISS.from_documents(documents=documents, embedding=embeddings)
            
            st.success("Vector store created successfully!")
            
        except Exception as e:
            st.error(f"Error while creating vector store: {e}")
            return

        # Step 3: Create the retrieval chain
        try:
            with st.spinner("Creating retrieval chain..."):
                # Initialize the LLM
                llm = OllamaLLM(model="gemma2:2b")

                # Setup the prompt template
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are an AI assistant who helps in answering the User's question based on the below given context: {context}"),
                        ("user", "{input}")
                    ]
                )

                # Setup the document chain
                document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

                # Create retrieval chain
                retriever = vectorstore_db.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

            st.success("Retrieval chain created successfully!")
        
        except Exception as e:
            st.error(f"Error while creating retrieval chain: {e}")
            return

        # Step 4: Ask the user for a question
        user_input = st.text_input("Ask a question related to the documentation:")

        if user_input:
            try:
                with st.spinner("Processing your query..."):
                    # Process the query using the retrieval chain
                    response = retrieval_chain.invoke({"input": user_input})
                    st.subheader("Answer:")
                    st.write(response['answer'])
            except Exception as e:
                st.error(f"Error while processing the query: {e}")

    else:
        st.warning("Please enter a valid URL to proceed.")

if __name__ == "__main__":
    main()
