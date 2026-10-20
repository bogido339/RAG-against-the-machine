class RagError(Exception):
    ...


class IndexerError(RagError):
    ...


class RetrieverError(RagError):
    ...


class DatasetRetrieverError(RagError):
    ...


class GeneratorErrro(RagError):
    ...


class DatasetGeneratorError(RagError):
    ...


class EvaluatorError(RagError):
    ...
