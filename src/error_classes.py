class RagError(Exception):
    """Base exception for all RAG pipeline errors."""
    ...


class IndexerError(RagError):
    """Handle errors during document indexing."""
    ...


class RetrieverError(RagError):
    """Handle errors during document retrieval."""
    ...


class DatasetRetrieverError(RagError):
    """Handle errors during dataset retrieval."""
    ...


class GeneratorError(RagError):
    """Handle errors during answer generation."""
    ...


class DatasetGeneratorError(RagError):
    """Handle errors during dataset answer generation."""
    ...


class EvaluatorError(RagError):
    """Handle errors during retrieval evaluation."""
    ...
