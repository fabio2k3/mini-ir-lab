import argparse
from pathlib import Path

from src.indexer import Indexer
from src.boolean_model import BooleanModel
from src.vector_model import VectorModel
from src.bm25 import BM25Model
from src.search import SearchEngine


def load_documents(file_path: str) -> dict:
    """
    Carga documentos desde docs.txt.
    Formato esperado:
    id|texto
    """
    documents = {}
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue

            doc_id_str, text = line.split("|", 1)
            doc_id = int(doc_id_str.strip())
            documents[doc_id] = text.strip()

    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Mini Information Retrieval Lab: boolean, vectorial y BM25"
    )
    parser.add_argument("--mode", required=True, choices=["boolean", "vector", "bm25"])
    parser.add_argument("--query", required=True, help="Consulta a ejecutar")
    parser.add_argument("--top_k", type=int, default=5, help="Número de resultados")

    args = parser.parse_args()

    # 1) Cargar documentos
    documents = load_documents("data/docs.txt")

    # 2) Construir índice
    indexer = Indexer(documents)
    indexer.build_index()

    # 3) Crear modelos
    boolean_model = BooleanModel(indexer.inverted_index, set(documents.keys()))
    vector_model = VectorModel(documents, indexer.idf)
    bm25_model = BM25Model(
        documents=documents,
        term_frequencies=indexer.term_frequencies,
        doc_lengths=indexer.doc_lengths,
        avg_dl=indexer.avg_doc_length(),
        N=indexer.N,
    )

    # 4) Crear buscador unificado
    engine = SearchEngine(indexer, boolean_model, vector_model, bm25_model)

    # 5) Ejecutar búsqueda
    results = engine.search(args.mode, args.query, top_k=args.top_k)

    # 6) Mostrar resultados
    print(f"\nModo: {args.mode}")
    print(f"Query: {args.query}\n")
    print(engine.format_results(results))


if __name__ == "__main__":
    main()