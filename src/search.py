from typing import Dict, List, Tuple


class SearchEngine:
    """
    Une el indexador con los distintos modelos de recuperación.
    """

    def __init__(self, indexer, boolean_model, vector_model, bm25_model):
        self.indexer = indexer
        self.boolean_model = boolean_model
        self.vector_model = vector_model
        self.bm25_model = bm25_model

    def search(self, mode: str, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Ejecuta la búsqueda según el modo elegido.
        """
        mode = mode.lower()

        if mode == "boolean":
            doc_ids = self.boolean_model.search(query)
            return [(doc_id, 1.0) for doc_id in doc_ids[:top_k]]

        if mode == "vector":
            return self.vector_model.search(query, top_k=top_k)

        if mode == "bm25":
            return self.bm25_model.search(query, top_k=top_k)

        raise ValueError(f"Modo no soportado: {mode}")

    def format_results(self, results: List[Tuple[int, float]]) -> str:
        """
        Formatea la salida para consola.
        """
        if not results:
            return "No se encontraron resultados."

        lines = []
        for rank, (doc_id, score) in enumerate(results, start=1):
            text = self.indexer.documents[doc_id]
            lines.append(f"{rank}. Doc {doc_id} | score={score:.4f} | {text}")
        return "\n".join(lines)