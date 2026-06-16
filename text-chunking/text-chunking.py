def text_chunking(tokens, chunk_size, overlap):
    step = chunk_size - overlap
    if step <= 0:
        raise("overlap must be smaller than chunk_size")
    chunks = []
    for i in range(0, len(tokens), step):
        chunk = tokens[i: i+chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        if i+ chunk_size >= len(tokens):
            break
    return chunks