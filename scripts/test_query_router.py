from app.query_router import classify_query

tests = [
    "Taylor Swift",
    "Jay Chou",
    "周杰伦",
    "类似Taylor Swift但不要太热门，更梦幻一点",
    "sad female pop",
    "80s dreamy female pop",
]

for t in tests:
    print(t)
    print(classify_query(t))
    print("-" * 60)