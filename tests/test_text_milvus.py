import unittest
from unittest.mock import Mock
from your_module import TextMilvus

class TestTextMilvus(unittest.TestCase):
    def setUp(self):
        # Mocking dependencies
        self.mock_client = Mock()
        self.mock_client.get.return_value = [{'id': 1, 'text': 'Sample text'}]
        self.mock_embeddings = Mock()
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]  # Mock query vector

        # Creating TextMilvus instance with mocked dependencies
        self.text_milvus = TextMilvus('http://localhost:19530')
        self.text_milvus.milvus_client = self.mock_client
        self.text_milvus.embeddings = self.mock_embeddings

    def test_retrieve_text_by_id(self):
        # Testing successful retrieval
        result = self.text_milvus._TextMilvus__retrieve_text_by_id('collection', 1)
        self.assertEqual(result, 'Sample text')

        # Testing retrieval failure
        self.mock_client.get.side_effect = Exception('Mocked error')
        result = self.text_milvus._TextMilvus__retrieve_text_by_id('collection', 1)
        self.assertIsNone(result)

    def test_get(self):
        # Mocking search results
        self.mock_client.search.return_value = [{'id': 1}, {'id': 2}]

        # Testing successful get
        result = self.text_milvus.get('collection', 'query')
        self.assertEqual(result, ['Sample text', 'Sample text'])

        # Testing failed search
        self.mock_client.search.side_effect = Exception('Mocked error')
        result = self.text_milvus.get('collection', 'query')
        self.assertEqual(result, [])

