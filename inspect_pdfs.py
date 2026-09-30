import glob
import pypdf

for f in sorted(glob.glob("documents/**/*.pdf", recursive=True)):
    reader = pypdf.PdfReader(f)
    words = [len((p.extract_text() or "").split()) for p in reader.pages]
    avg = sum(words) / len(words)
    print(f"{f:50} pages={len(words):3}  words/page: avg={avg:5.0f} min={min(words):4} max={max(words):4}  ≈tokens/page={avg * 1.33:5.0f}")