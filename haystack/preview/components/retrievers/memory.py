from typing import Dict, List, Any, Optional

from canals.errors import ComponentDeserializationError

from haystack.preview import component, Document
from haystack.preview.document_stores import store, MemoryDocumentStore, StoreAwareMixin


@component
class MemoryRetriever(StoreAwareMixin):
    """
    A component for retrieving documents from a MemoryDocumentStore using the BM25 algorithm.

    Needs to be connected to a MemoryDocumentStore to run.
    """

    supported_stores = [MemoryDocumentStore]

    def __init__(self, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, scale_score: bool = True):
        """
        Create a MemoryRetriever component.

        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).
        :param scale_score: Whether to scale the BM25 score or not (default is True).

        :raises ValueError: If the specified top_k is not > 0.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        self.filters = filters
        self.top_k = top_k
        self.scale_score = scale_score

    def to_dict(self) -> Dict[str, Any]:
        store = None
        if self._store:
            store = self._store.to_dict()
        if self._store_name:
            store = self._store_name
        return {
            "hash": id(self),
            "type": self.__class__.__name__,
            "store": store,
            "init_parameters": {"filters": self.filters, "top_k": self.top_k, "scale_score": self.scale_score},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRetriever":
        if "type" not in data:
            raise ComponentDeserializationError("Missing 'type' in component serialization data")
        if data["type"] != cls.__name__:
            raise ComponentDeserializationError(f"Component '{data['type']}' can't be deserialized as '{cls.__name__}'")
        init_params = data["init_parameters"]
        comp = cls(**init_params)
        if isinstance(data["store"], dict):
            # Deserialises the store only if it's been deserialised with the component.
            # If it's not a dictionary it must be a string, that means the component
            # is being deserialised inside a Pipeline, the Pipeline will take care
            # of setting the proper store to this component.
            store_type = data["store"]["type"]
            comp.store = store.registry[store_type].from_dict(data["store"])
        return comp

    @component.output_types(documents=List[List[Document]])
    def run(
        self,
        queries: List[str],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
    ):
        """
        Run the MemoryRetriever on the given input data.

        :param query: The query string for the retriever.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :param scale_score: Whether to scale the BM25 scores or not.
        :param stores: A dictionary mapping document store names to instances.
        :return: The retrieved documents.

        :raises ValueError: If the specified document store is not found or is not a MemoryDocumentStore instance.
        """
        self.store: MemoryDocumentStore
        if not self.store:
            raise ValueError("MemoryRetriever needs a store to run: set the store instance to the self.store attribute")

        if filters is None:
            filters = self.filters
        if top_k is None:
            top_k = self.top_k
        if scale_score is None:
            scale_score = self.scale_score

        docs = []
        for query in queries:
            docs.append(self.store.bm25_retrieval(query=query, filters=filters, top_k=top_k, scale_score=scale_score))
        return {"documents": docs}
