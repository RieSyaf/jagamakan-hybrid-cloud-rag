from langchain_ollama import OllamaEmbeddings, ChatOllama # to turns chat into vectors, writer
from langchain_postgres import PGVector #to communicate with Postgres vector db
from langchain_core.prompts import ChatPromptTemplate #to create chat prompt template

#connecting to the database and colletion_name
DB_CONNECTION = "postgresql+psycopg://myuser:mypassword@localhost:5432/vector_db"  #protocol://username:password@host:port/database_name
COLLECTION_NAME = 'diet_guide' # like a table name

def main():

    #initialize the model to use
    embeddings = OllamaEmbeddings(model="nomic-embed-text") #to search the vector in db
    llm = ChatOllama(model="llama3.2") #to actually talk to the user

    #connect to existing vector store in Postgres
    vector_store = PGVector(
        embeddings=embeddings, #the model to embed text
        collection_name= COLLECTION_NAME, 
        connection=DB_CONNECTION, #connect the db
        use_jsonb=True, #postgres feature.. tells database to store metadata as seachable json blob, assist filtration
    )

    #user input query
    print("\n--- Manual RAG System ---")
    user_query = input("Enter your question about the diet guide: ")

    #using the seaching function of vector db to find top 3 relevant chunks (based on vector similarity)
    print("Searching for relevant document chunks...")
    docs = vector_store.similarity_search(user_query, k=7) #fun fact : k is k in knn

    # --- DEBUGGING BLOCK START ---
    print("\n--- EVIDENCE RETRIEVED FROM DATABASE ---")
    for i, doc in enumerate(docs):
        preview = doc.page_content.replace("\n", " ")[:150] # Show first 150 chars
        print(f"Chunk {i+1}: {preview}...")
    print("------------------------------------------\n")
    # --- DEBUGGING BLOCK END ---

    #structured the retrieved 3 chunks  from vector db into one long string
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    #this is because the database gave us 3 separate chunks, we need to combine them into one string
    #Because the AI cannot process a list of objects or a database connection. It only understands a sequence of characters or smtg

    #creates a usable blueprint to tells the AI how to behave and where to insert data(context_text and user_query) into a single message
    prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful nutrition assistant. Use the following pieces of extracted 
    context from a Diet Guide to answer the user's question. 
    If you don't know the answer based on the context, just say you don't know.
    
    CONTEXT:
    {context}
    
    QUESTION: 
    {question}
    
    ANSWER:
    """)

    #generate the final response by sending the structured prompt to the LLM
    chain = prompt_template | llm  #take the output of prompt_template and feed it to llm usgin the pipe operator (|)
    response = chain.invoke({ #invoke is like the go button, it sends the data to the model and waits for reply
        "context": context_text,
        "question": user_query
    })

    print("--- AI RESPONSE ---")
    print(response.content)

if __name__ == "__main__":
    main()

