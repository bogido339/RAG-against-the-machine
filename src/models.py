from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """Validate the file path and character indices of a source."""

    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(MinimalSource):
    """Validate a text chunk and its source location."""

    content: str


class UnansweredQuestion(BaseModel):
    """Validate a question and its unique identifier."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Validate a question, its answer, and reference sources."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Validate a dataset of answered or unanswered questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Validate a question and its retrieved source locations."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Validate retrieved sources and a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Validate dataset retrieval results and the requested top-k."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Validate dataset retrieval results with generated answers."""

    search_results: list[MinimalAnswer]
    k: int
