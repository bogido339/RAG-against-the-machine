import fire


class RagCLI:
    """Command-line interface for your RAG pipeline."""


    def index(self, a: str = "hello", b: int = 20):
        print("STR:", a)
        print("INT:", b)



if __name__ == "__main__":
    # Passing an instantiated object as in your snippet
    fire.Fire(RagCLI())
