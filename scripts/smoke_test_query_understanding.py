from app.query_understanding import QueryUnderstandingModule
from app.text_processor import TextProcessor


QUERIES = [
    "Adele",
    "sad female pop",
    "类似Taylor Swift但不要太热门，更梦幻一点",
    "Jay Chou",
]


def main() -> None:
    module = QueryUnderstandingModule(text_processor=TextProcessor())

    for query in QUERIES:
        parsed = module.understand(query)
        semantic_query = module.build_semantic_query(parsed)

        print("=" * 80)
        print("query:", query)
        print("parsed:", parsed)
        print("semantic_query:", semantic_query)


if __name__ == "__main__":
    main()