from typing import Dict, Any, List, Optional

import pytest

from haystack.preview import Pipeline
from haystack.preview.components.retrievers.memory import MemoryRetriever
from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import Store, MemoryDocumentStore

from test.preview.components.base import BaseTestComponent

from haystack.preview.document_stores.protocols import DuplicatePolicy


@pytest.fixture()
def mock_docs():
    return [
        Document.from_dict({"content": "Javascript is a popular programming language"}),
        Document.from_dict({"content": "Java is a popular programming language"}),
        Document.from_dict({"content": "Python is a popular programming language"}),
        Document.from_dict({"content": "Ruby is a popular programming language"}),
        Document.from_dict({"content": "PHP is a popular programming language"}),
    ]


class Test_MemoryRetriever(BaseTestComponent):
    @pytest.mark.unit
    def test_save_load(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(MemoryRetriever(), tmp_path)

    @pytest.mark.unit
    def test_save_load_with_parameters(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(MemoryRetriever(top_k=5, scale_score=False), tmp_path)

    @pytest.mark.unit
    def test_init_default(self):
        retriever = MemoryRetriever()
        assert retriever.defaults == {"filters": {}, "top_k": 10, "scale_score": True}

    @pytest.mark.unit
    def test_init_with_parameters(self):
        retriever = MemoryRetriever(top_k=5, scale_score=False)
        assert retriever.defaults == {"filters": {}, "top_k": 5, "scale_score": False}

    @pytest.mark.unit
    def test_init_with_invalid_top_k_parameter(self):
        with pytest.raises(ValueError, match="top_k must be > 0, but got -2"):
            MemoryRetriever(top_k=-2, scale_score=False)

    @pytest.mark.unit
    def test_valid_run(self, mock_docs):
        top_k = 5
        ds = MemoryDocumentStore()
        ds.write_documents(mock_docs)

        retriever = MemoryRetriever(top_k=top_k)
        retriever.store = ds
        result = retriever.run(data=MemoryRetriever.Input(queries=["PHP", "Java"]))

        assert getattr(result, "documents")
        assert len(result.documents) == 2
        assert len(result.documents[0]) == top_k
        assert len(result.documents[1]) == top_k
        assert result.documents[0][0].content == "PHP is a popular programming language"
        assert result.documents[1][0].content == "Java is a popular programming language"

    @pytest.mark.unit
    def test_invalid_run_wrong_store_type(self):
        class MockStore:
            ...

        retriever = MemoryRetriever()
        with pytest.raises(ValueError, match="is not compatible with this component"):
            retriever.store = MockStore()

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query, query_result",
        [
            ("Javascript", "Javascript is a popular programming language"),
            ("Java", "Java is a popular programming language"),
        ],
    )
    def test_run_with_pipeline(self, mock_docs, query: str, query_result: str):
        ds = MemoryDocumentStore()
        ds.write_documents(mock_docs)
        retriever = MemoryRetriever()

        pipeline = Pipeline()
        pipeline.add_store("memory", ds)
        pipeline.add_component("retriever", retriever, store="memory")
        result: Dict[str, Any] = pipeline.run(data={"retriever": MemoryRetriever.Input(queries=[query])})

        assert result
        assert "retriever" in result
        results_docs = result["retriever"].documents
        assert results_docs
        assert results_docs[0][0].content == query_result

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query, query_result, top_k",
        [
            ("Javascript", "Javascript is a popular programming language", 1),
            ("Java", "Java is a popular programming language", 2),
            ("Ruby", "Ruby is a popular programming language", 3),
        ],
    )
    def test_run_with_pipeline_and_top_k(self, mock_docs, query: str, query_result: str, top_k: int):
        ds = MemoryDocumentStore()
        ds.write_documents(mock_docs)
        retriever = MemoryRetriever()

        pipeline = Pipeline()
        pipeline.add_store("memory", ds)
        pipeline.add_component("retriever", retriever, store="memory")
        result: Dict[str, Any] = pipeline.run(data={"retriever": MemoryRetriever.Input(queries=[query], top_k=top_k)})

        assert result
        assert "retriever" in result
        results_docs = result["retriever"].documents
        assert results_docs
        assert len(results_docs[0]) == top_k
        assert results_docs[0][0].content == query_result
