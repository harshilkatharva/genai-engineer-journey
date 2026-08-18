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

# 4

# No index
result = [
    {
        "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
        "top_k": 10,
        "results_count": 10,
        "query_latency_ms": 147.61457799977507,
        "chunk_ids": [
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0699",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0700",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0697",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0693",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0692",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0695",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0677",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0795",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0796",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0698",
        ],
        "operation_type": "query",
        "logged_at": "2026-08-17T12:42:10.572587+00:00",
    },
    {
        "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
        "top_k": 10,
        "results_count": 10,
        "query_latency_ms": 132.4924580003426,
        "chunk_ids": [
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0415",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0370",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0361",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0496",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0406",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0505",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0613",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0226",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0451",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0136",
        ],
        "operation_type": "query",
        "logged_at": "2026-08-17T13:24:15.891128+00:00",
    },
    {
        "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
        "top_k": 10,
        "results_count": 10,
        "query_latency_ms": 217.39578199776588,
        "chunk_ids": [
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0058",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0040",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0076",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_2218",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0022",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0059",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0004",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_4383",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_2200",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_2236",
        ],
        "operation_type": "query",
        "logged_at": "2026-08-17T13:23:50.964591+00:00",
    },
    # hnsw m=8,ef_construction = 32
    {
        "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
        "top_k": 10,
        "results_count": 10,
        "query_latency_ms": 143.6771660009981,
        "chunk_ids": [
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0700",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0692",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0677",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0795",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0796",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0698",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0756",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0748",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0691",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0694",
        ],
        "operation_type": "query",
        "logged_at": "2026-08-17T13:10:32.239128+00:00",
    },
    {
        "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
        "top_k": 10,
        "results_count": 10,
        "query_latency_ms": 80.91740599775221,
        "chunk_ids": [
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0415",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0370",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0361",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0496",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0406",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0505",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0226",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0451",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0136",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0388",
        ],
        "operation_type": "query",
        "logged_at": "2026-08-17T13:12:51.123104+00:00",
    },
    {
        "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
        "top_k": 10,
        "results_count": 10,
        "query_latency_ms": 75.89765900047496,
        "chunk_ids": [
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0058",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0040",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0076",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_2218",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0022",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0059",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0004",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_4383",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_2200",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_2236",
        ],
        "operation_type": "query",
        "logged_at": "2026-08-17T13:13:40.883808+00:00",
    },
]


# 5

result = [
    # hnsw m=8,ef_construction = 32
    {
        "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
        "top_k": 10,
        "results_count": 10,
        "query_latency_ms": 143.6771660009981,
        "chunk_ids": [
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0700",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0692",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0677",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0795",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0796",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0698",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0756",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0748",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0691",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0694",
        ],
        "operation_type": "query",
        "logged_at": "2026-08-17T13:10:32.239128+00:00",
    },
    {
        "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
        "top_k": 10,
        "results_count": 10,
        "query_latency_ms": 80.91740599775221,
        "chunk_ids": [
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0415",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0370",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0361",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0496",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0406",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0505",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0226",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0451",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0136",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0388",
        ],
        "operation_type": "query",
        "logged_at": "2026-08-17T13:12:51.123104+00:00",
    },
    {
        "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
        "top_k": 10,
        "results_count": 10,
        "query_latency_ms": 75.89765900047496,
        "chunk_ids": [
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0058",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0040",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0076",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_2218",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0022",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0059",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0004",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_4383",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_2200",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_2236",
        ],
        "operation_type": "query",
        "logged_at": "2026-08-17T13:13:40.883808+00:00",
    },
    # hnsw m=16,ef_construction = 64
    {
        "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
        "top_k": 10,
        "results_count": 10,
        "query_latency_ms": 53.41078800120158,
        "chunk_ids": [
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0699",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0700",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0697",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0693",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0692",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0695",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0677",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0795",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0796",
            "ca6afd5f-d24c-44ee-9d1c-e00a19619bed_chunk_0698",
        ],
        "operation_type": "query",
        "logged_at": "2026-08-17T13:21:21.663947+00:00",
    },
    {
        "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
        "top_k": 10,
        "results_count": 10,
        "query_latency_ms": 50.279732000490185,
        "chunk_ids": [
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0415",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0370",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0361",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0496",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0406",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0505",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0613",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0226",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0451",
            "bd4ffacc-3bca-4d8f-a85f-a9a4aac0eaeb_chunk_0136",
        ],
        "operation_type": "query",
        "logged_at": "2026-08-17T13:21:42.836484+00:00",
    },
    {
        "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
        "top_k": 10,
        "results_count": 10,
        "query_latency_ms": 53.93129900039639,
        "chunk_ids": [
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0058",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0040",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0076",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_2218",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0022",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0059",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_0004",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_4383",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_2200",
            "eb444a5d-adfc-464c-8c37-0605fbd12448_chunk_2236",
        ],
        "operation_type": "query",
        "logged_at": "2026-08-17T13:21:59.072331+00:00",
    },
]

# 6 test mesure in tests folder with our real app

# 7 5000 entry insert in pgvector with out hnsw

row_by_row = {
    "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
    "batch_number": 1,
    "batch_size": 5000,
    "total_chunks": 5000,
    "insertion_latency_ms": 12323.110678997182,
    "operation_type": "index_batch",
    "logged_at": "2026-08-18T10:08:30.053961+00:00",
}
batch = {
    "tenant_id": "5c664d1a-616d-46a4-9ef3-d3934166a1ae",
    "batch_number": 1,
    "batch_size": 5000,
    "total_chunks": 5000,
    "insertion_latency_ms": 6033.614160998695,
    "operation_type": "index_batch",
    "logged_at": "2026-08-18T10:08:36.089016+00:00",
}

# 2x fast insertion in batch insertion
