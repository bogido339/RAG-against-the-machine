from pydantic import BaseModel


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(MinimalSource):
    content: str


chunk = Chunk(
    content="text",
    file_path="path",
    first_character_index=0,
    last_character_index=10
)

print(chunk.content,
    chunk.file_path,
    chunk.first_character_index,
    chunk.last_character_index
)
