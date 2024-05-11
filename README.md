# TextMilvus

TextMilvus is a Python package designed to facilitate text similarity search using Milvus vector database. It allows users to retrieve similar texts from a collection stored in Milvus by providing an input text query.

## Features

- Retrieve similar texts from a Milvus vector database based on input text query.
- Utilizes Hugging Face embeddings for text representation.
- Flexible configuration options for Milvus connection.

## Installation

You can install TextMilvus using pip:
 ```sh
pip install textmilvus
```
```sh
from textmilvus import TextMilvus

client = TextMilvus("milvus_ip_address")
print(client.get('cluster_name', 'What is Financial health according to balance sheet of Tata consumer Company'))
```

