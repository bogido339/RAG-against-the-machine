from src.models import MinimalSource


def main():
    msrc = MinimalSource(
        file_path="hello",
        first_character_index=12
    )

    print(msrc.file_path)
    print(msrc.first_character_index)

main()