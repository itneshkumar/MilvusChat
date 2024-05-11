from pymilvus import MilvusClient
from langchain.embeddings import HuggingFaceEmbeddings
import logging

# Initialize HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')


# Set up logging configuration
logging.basicConfig(level=logging.INFO)  # Adjust the level as needed


class TextMilvus:
    def __init__(self, milvus_url):
        """
        Constructor for TextMilvus class.

        Parameters:
        - milvus_url (str): URL of the Milvus server, e.g., 'http://example.com', 'http://localhost:19530'
        """
        self.milvus_url = milvus_url
        self.milvus_client = MilvusClient(uri=milvus_url)

    def __retrieve_text_by_id(self, collection_name, document_id):
        """
        Retrieve text by document ID from Milvus.

        Parameters:
        - collection_name (str): Name of the collection where the document is stored.
        - document_id (int): ID of the document to retrieve.

        Returns:
        - str or None: Text content of the document if found, None otherwise.
        """
        try:
            entities = self.milvus_client.get(collection_name, document_id)
            if entities and entities[0]:
                return entities[0].get("text")
            else:
                return None  # Indicate document not found
        except Exception as e:
            logging.error(f"Failed to retrieve text for document ID {document_id}: {e}")
            return None

    def get(self, collection_name, input_text,limit=16):
        """
        Search for similar documents and retrieve their text content.

        Parameters:
        - collection_name (str): Name of the collection to search in.
        - input_text (str): Input text for similarity search.

        Returns:
        - list of str: Text content of matched documents.
        """
        try:
            query_vector = embeddings.embed_query(input_text)
            search_param = {
                "metric_type": "L2",
                "params": {"nprobe": limit},
                "top_k": 2
            }
            results = self.milvus_client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=5,
                search_params=search_param
            )
            document_ids = [hit['id'] for hit in results[0]]
            matched_texts = []
            for doc_id in document_ids:
                matched_text = self.__retrieve_text_by_id(collection_name, doc_id)
                if matched_text:
                    matched_texts.append(matched_text)
            return matched_texts
        except Exception as e:
            logging.error(f"Failed to get matched documents: {e}")
            return []
