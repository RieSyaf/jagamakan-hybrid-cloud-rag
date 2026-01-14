import boto3 #to connect to the AWS
import os
from dotenv import load_dotenv #this to read .env file that contains our passwords
from langchain_community.document_loaders import S3FileLoader #tool to get files from S3
from langchain_ollama import OllamaEmbeddings # to turns chat into vectors
from langchain_postgres import PGVector #to communicate with Postgres vector db
from langchain_text_splitters import RecursiveCharacterTextSplitter #to split large text into smaller chunks for vector db
#note : why we split text ? because feeding large text to vector db is inefficient and may exceed token limits (big and expensive)

#look for .env file and load access key for the AWS in temporary memory
load_dotenv()

#connect to vector db in Postgres
DB_CONNECTION = "postgresql+psycopg://myuser:mypassword@localhost:5432/vector_db"  #protocol://username:password@host:port/database_name

def main():
    #------------step 1: download document from S3(AWS cloud storage)-------------
    try:
        print("Connecting to S3 and downloading document...")
        loader = S3FileLoader(
            bucket = 'langchain-rag-storage', #exact name of out bucket
            key="latest-01.Buku-MDG-2020_12Mac2024.pdf", #exact name of our file in the bucket
            #the access key for the user of the AWS
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        doc = loader.load()
        print(f"Success! Downloaded {len(doc)} pages from S3.")
    except Exception as e:
        print(f"\nERROR: Failed to download document from S3.\n{e}")
        return

    #------------step 2: split document into smaller chunks---------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,#cut paper every 1000 char
        chunk_overlap=200 #keep the last 200 char of previous chunk on the next chunk (to maintain context)
    )
    chunks = text_splitter.split_documents(doc)
    print(f"Document split into {len(chunks)} chunks.")

    #------------step 3: embed chunks into vectors using Ollama and store in Postgres------------

    embeddings = OllamaEmbeddings(model="nomic-embed-text") #initialize Ollama embedding model

    vector_store = PGVector(
        embeddings=embeddings, #the model to embed text
        collection_name='diet_guide', # like a table name
        connection=DB_CONNECTION, #connect the db
        use_jsonb=True, #postgres feature.. tells database to store metadata as seachable json blob, assist filtration
    )

    #------------step 4: the actual embeddings process and saving to vector db(postgrsql)-----------------
    
    #This sends text to Ollama -> converts to vectors -> saves to vector db (postgres in container)
    vector_store.add_documents(chunks)

    print("--- INGESTION COMPLETE ---")
    print(f"Successfully saved {len(chunks)} vector embeddings to your local database.")

if __name__ == "__main__":
        main()
    



