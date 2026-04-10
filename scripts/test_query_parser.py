from app.query_parser import LLMQueryParser

parser = LLMQueryParser(
    model="openai/gpt-4.1-mini",
    max_tokens=300,
)

queries = [
    "Jay Chou",
    "类似Taylor Swift但不要太热门，更梦幻一点",
    "适合深夜学习的电子乐，不要太吵",
    "想听轻盈一点的女声流行",
    "来点90年代的摇滚，不要现场版",
]

for q in queries:
    parsed = parser.parse(q)
    print("=" * 80)
    print(q)
    print(parsed.model_dump_json(indent=2, ensure_ascii=False))