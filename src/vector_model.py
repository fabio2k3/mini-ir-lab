from collections import defaultdict
from math import sqrt
from typing import Dict, List, Tuple

from text_utils import preprocess


class VectorModel:
    """
    Modelo vectorial clásico.
    Representa documentos y consultas como vectores TF-IDF.
    Luego compara usando similitud coseno.
    """

    def __init__(self, documents: Dict[int, str], idf_function):
        self.documents = documents
        self.idf_function = idf_function
        self.doc_vectors: Dict[int, Dict[str, float]] = {}
        self._build_vectors()

    def _tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calcula frecuencia normalizada de términos.
        """
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        total = len(tokens)
        if total == 0:
            return {}

        return {term: count / total for term, count in tf.items()}

    def _build_vectors(self) -> None:
        """
        Construye los vectores TF-IDF de todos los documentos.
        """
        for doc_id, text in self.documents.items():
            tokens = preprocess(text)
            tf = self._tf(tokens)
            vec = {}
            for term, tf_value in tf.items():
                vec[term] = tf_value * self.idf_function(term)
            self.doc_vectors[doc_id] = vec

    def _query_vector(self, query: str) -> Dict[str, float]:
        """
        Construye el vector TF-IDF de la consulta.
        """
        tokens = preprocess(query)
        tf = self._tf(tokens)
        vec = {}
        for term, tf_value in tf.items():
            vec[term] = tf_value * self.idf_function(term)
        return vec

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Calcula similitud coseno entre dos vectores dispersos.
        """
        if not vec1 or not vec2:
            return 0.0

        common_terms = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[t] * vec2[t] for t in common_terms)

        norm1 = sqrt(sum(v * v for v in vec1.values()))
        norm2 = sqrt(sum(v * v for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Devuelve los mejores documentos ordenados por similitud coseno.
        """
        q_vec = self._query_vector(query)
        scores = []

        for doc_id, d_vec in self.doc_vectors.items():
            score = self._cosine_similarity(q_vec, d_vec)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]