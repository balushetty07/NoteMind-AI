import time
from app import load_store, answer

chunks, sources, matrix = load_store()
tests = [
    ("When do classes commence?", ["31"]),
    ("How many holidays are in October?", ["gandhi", "mahalaya", "mahanavami", "vijaya"]),
    ("What is the last working day?", ["23"]),
    ("When does the next semester start?", ["feb"]),
    ("Which holiday is on 10th November?", ["bali"]),
    ("What is the course code of Advanced Communication Lab?", ["24ec58l"]),
    ("How many credits is the Advanced Communication Lab?", ["1.5"]),
    ("What is the placement rate?", ["85"]),
    ("What is the KCET annual fee?", ["1.2"]),
    ("Which experiment is in week 4?", ["antenna"]),
]
unknown = ["Who is the HOD of ECE?", "What is the hostel fee?"]
idk = ["don't know", "do not know", "not contain", "not mention", "not provided"]

score = 0
for q, keys in tests:
    r = answer(q, chunks, sources, matrix)[0].lower()
    ok = all(k in r for k in keys)
    score += ok
    print("✅" if ok else "❌", q)
    time.sleep(12)
for q in unknown:
    r = answer(q, chunks, sources, matrix)[0].lower()
    ok = any(w in r for w in idk)
    score += ok
    print("✅" if ok else "❌", q)
    time.sleep(5)
print(f"\nScore: {score}/{len(tests) + len(unknown)}")