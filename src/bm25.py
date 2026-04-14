from math import log
from typing import Dict, List, Tuple

from text_utils import preprocess


class BM25Model:
    """
    Implementación simple de BM25.
    BM25 es muy usado en buscadores reales porque rankea muy bien documentos.
    """

    def __init__(self, documents: Dict[int, str], term_frequencies, doc_lengths, avg_dl, N):
        self.documents = documents
        self.term_frequencies = term_frequencies
        self.doc_lengths = doc_lengths
        self.avg_dl = avg_dl if avg_dl > 0 else 1.0
        self.N = N

        # Parámetros típicos de BM25
        self.k1 = 1.5
        self.b = 0.75

    def _idf(self, term: str) -> float:
        """
        IDF de BM25 con suavizado.
        """
        df = 0
        for doc_id in self.documents:
            if term in self.term_frequencies.get(doc_id, {}):
                df += 1

        return log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _score_doc(self, query_terms: List[str], doc_id: int) -> float:
        """
        Calcula el score BM25 de un documento respecto a la consulta.
        """
        score = 0.0
        doc_tf = self.term_frequencies.get(doc_id, {})
        dl = self.doc_lengths.get(doc_id, 0)

        for term in query_terms:
            tf = doc_tf.get(term, 0)
            if tf == 0:
                continue

            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_dl))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Busca y ordena por score BM25.
        """
        query_terms = preprocess(query)
        scores = []

        for doc_id in self.documents:
            score = self._score_doc(query_terms, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]