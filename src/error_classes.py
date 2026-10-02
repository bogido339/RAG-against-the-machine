class ClassError(Exception):
    ...


class IndexerError(ClassError):
    ...


class RetrieverError(ClassError):
    ...


class DatasetRetrieverError(ClassError):
    ...


class GeneratorErrro(ClassError):
    ...


class DatasetGeneratorErrro(ClassError):
    ...


class EvaluatorError(ClassError):
    ...
