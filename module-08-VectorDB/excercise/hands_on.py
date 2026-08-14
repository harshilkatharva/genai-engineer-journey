# 3
query = (
    """
SELECT chunks, 1 - (embedding <=> $1) as semilarity
FROM document_chunks
WHERE document_type == $2
ORDER BY embedding <=> $1
LIMIT $3;
""",
    # query_embedding,document_type,top_k
)
