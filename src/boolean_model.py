from typing import Dict, List, Set

from text_utils import preprocess


class BooleanModel:
    """
    Modelo booleano simple.
    Permite consultas como:
    - search AND ranking
    - search OR retrieval
    - search AND NOT deep
    """

    def __init__(self, inverted_index: Dict[str, set], all_doc_ids: Set[int]):
        self.inverted_index = inverted_index
        self.all_doc_ids = all_doc_ids

    def _term_docs(self, term: str) -> Set[int]:
        """
        Devuelve el conjunto de documentos que contienen un término.
        """
        return set(self.inverted_index.get(term, set()))

    def search(self, query: str) -> List[int]:
        """
        Evalúa una consulta booleana.

        Para mantenerlo sencillo:
        - convierte la consulta en tokens
        - interpreta AND, OR, NOT
        - usa conjuntos para hacer operaciones booleanas
        """
        tokens = query.replace("(", " ( ").replace(")", " ) ").split()
        if not tokens:
            return []

        # Convertimos términos normales a tokens preprocesados.
        # Los operadores se conservan en mayúsculas.
        normalized_tokens = []
        for tok in tokens:
            upper = tok.upper()
            if upper in {"AND", "OR", "NOT", "(", ")"}:
                normalized_tokens.append(upper)
            else:
                processed = preprocess(tok)
                if processed:
                    normalized_tokens.append(processed[0])

        # Para un proyecto simple, usamos una evaluación muy básica con precedencia:
        # NOT > AND > OR
        return sorted(list(self._evaluate(normalized_tokens)))

    def _evaluate(self, tokens: List[str]) -> Set[int]:
        """
        Evalúa la expresión booleana usando conjuntos.
        Esta versión es sencilla, no pretende ser un parser completo industrial.
        """
        if not tokens:
            return set()

        # Primero resolvemos NOT
        output = []
        i = 0
        while i < len(tokens):
            if tokens[i] == "NOT" and i + 1 < len(tokens):
                term = tokens[i + 1]
                result = self.all_doc_ids - self._term_docs(term)
                output.append(result)
                i += 2
            else:
                output.append(tokens[i])
                i += 1

        # Luego resolvemos AND
        output2 = []
        i = 0
        while i < len(output):
            if output[i] == "AND" and i > 0 and i + 1 < len(output):
                left = output2.pop()
                right = output[i + 1]
                if isinstance(right, str):
                    right = self._term_docs(right)
                output2.append(left & right)
                i += 2
            else:
                output2.append(output[i])
                i += 1

        # Finalmente resolvemos OR
        result = set()
        current = None
        op = None

        for item in output2:
            if item == "OR":
                op = "OR"
            else:
                if isinstance(item, str):
                    item = self._term_docs(item)

                if current is None:
                    current = set(item)
                elif op == "OR":
                    current = current | set(item)
                    op = None

        return current if current is not None else set()