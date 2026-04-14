# mini-ir-lab

Mini proyecto de Sistemas de Recuperación de Información hecho en Python puro.

## Qué hace

Este proyecto implementa tres enfoques clásicos de recuperación:

- Modelo booleano
- Modelo vectorial con TF-IDF y similitud coseno
- BM25

Además incluye:

- indexación de documentos
- índice invertido
- frecuencias de términos
- ranking de resultados desde consola

## Estructura

- `data/docs.txt`: colección pequeña de documentos
- `src/text_utils.py`: limpieza y tokenización
- `src/indexer.py`: construcción del índice
- `src/boolean_model.py`: búsqueda booleana
- `src/vector_model.py`: TF-IDF + cosine similarity
- `src/bm25.py`: ranking BM25
- `src/search.py`: capa de unión
- `main.py`: entrada principal

## Requisitos

No necesita librerías externas. Solo Python 3.10+.

## Cómo ejecutar

### Búsqueda booleana
```bash
python main.py --mode boolean --query "search AND ranking"
