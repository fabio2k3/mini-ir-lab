from collections import defaultdict
from math import log
from typing import Dict, List, Tuple

from text_utils import preprocess


class Indexer:
    """
    Construye la estructura de indexación del mini buscador.

    Guarda:
    - documentos originales
    - tokens por documento
    - índice invertido
    - frecuencias por documento
    - document frequency
    - longitud de documentos
    """

    def __init__(self, documents: Dict[int, str]):
        self.documents = documents
        self.tokenized_docs: Dict[int, List[str]] = {}
        self.inverted_index: Dict[str, set] = defaultdict(set)
        self.term_frequencies: Dict[int, Dict[str, int]] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.document_frequency: Dict[str, int] = {}
        self.N = len(documents)

    def build_index(self) -> None:
        """
        Recorre todos los documentos y crea las estructuras necesarias.
        """
        for doc_id, text in self.documents.items():
            tokens = preprocess(text)
            self.tokenized_docs[doc_id] = tokens
            self.doc_lengths[doc_id] = len(tokens)

            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
                self.inverted_index[token].add(doc_id)

            self.term_frequencies[doc_id] = dict(tf)

        # Calcula document frequency: cuántos documentos contienen cada término
        self.document_frequency = {
            term: len(doc_ids) for term, doc_ids in self.inverted_index.items()
        }

    def get_posting_list(self, term: str) -> List[int]:
        """
        Devuelve la lista de documentos donde aparece un término.
        """
        return sorted(list(self.inverted_index.get(term, set())))

    def idf(self, term: str) -> float:
        """
        Calcula una versión simple de IDF.
        Usamos suavizado para evitar división por cero.
        """
        df = self.document_frequency.get(term, 0)
        return log((self.N + 1) / (df + 1)) + 1

    def avg_doc_length(self) -> float:
        """
        Longitud media de documento, útil para BM25.
        """
        if self.N == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / self.N