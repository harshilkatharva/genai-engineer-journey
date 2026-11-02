"""Benchmark research dataset containing 15 curated topics across 5 technical domains."""

from __future__ import annotations

from pydantic import BaseModel, Field

from self_improving_report_writer.models.report import SourceReference


class BenchmarkTopic(BaseModel):
    """Specification of a benchmark topic and its authoritative frozen source context."""

    topic_id: str = Field(description="Unique benchmark topic identifier (e.g. 'qc-01')")
    title: str = Field(description="Research query or topic title")
    domain: str = Field(description="Domain classification (e.g. 'quantum_computing')")
    description: str = Field(description="Detailed topic scope and prompt specification")
    key_questions: list[str] = Field(description="Core questions that the report should address")
    gold_sources: list[SourceReference] = Field(
        description="Authoritative, frozen source reference chunks for exact citation"
    )


BENCHMARK_TOPICS: list[BenchmarkTopic] = [
    # Domain 1: Quantum Computing & Physics
    BenchmarkTopic(
        topic_id="qc-01",
        title="Fault-Tolerant Quantum Error Correction in Neutral Atom Arrays",
        domain="quantum_computing",
        description="Examine neutral atom quantum computing architectures with dual-species Rydberg arrays for transversal Clifford gates and surface code error correction.",
        key_questions=[
            "What physical mechanisms enable optical tweezer rearrangement of neutral atoms?",
            "How do two-qubit Rydberg gate fidelities compare between rubidium and cesium?",
            "What physical qubit overhead is required to achieve a logical error rate below 1e-6?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_qc_01_a",
                chunk_id="chunk_qc_01_1",
                title="Nature 2024: Coherent neutral atom quantum processor architectures",
                content="Neutral atom quantum computers utilize two-dimensional optical tweezer arrays to dynamically rearrange individual rubidium-87 and cesium-133 atoms with sub-micron positional precision. Recent experiments achieved two-qubit Rydberg controlled-Z gate fidelities of 99.5% using adiabatic passage pulses. Dual-species configurations allow non-destructive mid-circuit syndrome measurement without decohering neighboring data qubits.",
                score=0.95,
                metadata={"year": 2024, "domain": "quantum_computing"},
            ),
            SourceReference(
                source_id="src_qc_01_b",
                chunk_id="chunk_qc_01_2",
                title="Physical Review X: Threshold theorems for topological surface codes on atom grids",
                content="Surface code simulations on hexagonal and square neutral atom grids demonstrate a fault-tolerance threshold of 1.25% under realistic atom loss and photon scattering noise models. To achieve a logical error rate below 1e-6 per gate, a distance-7 rotated surface code requires 98 physical data qubits and 97 auxiliary syndrome qubits per logical qubit block.",
                score=0.91,
                metadata={"year": 2024, "domain": "quantum_computing"},
            ),
        ],
    ),
    BenchmarkTopic(
        topic_id="qc-02",
        title="Majorana Zero Modes and Topological Superconducting Qubits",
        domain="quantum_computing",
        description="Analyze the realization of non-Abelian anyons in semiconductor-superconductor nanowires and their utility for topologically protected quantum memory.",
        key_questions=[
            "What signatures distinguish true Majorana zero modes from trivial Andreev bound states?",
            "How does the topological gap protect qubits against local magnetic perturbations?",
            "What are the primary fabrication hurdles in indium arsenide/aluminum heterostructures?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_qc_02_a",
                chunk_id="chunk_qc_02_1",
                title="Science 2023: Non-Abelian braiding protocol signatures in hybrid nanowires",
                content="Hybrid semiconductor-superconductor nanowires composed of indium arsenide (InAs) epitaxially coated with aluminum (Al) exhibit quantized zero-bias conductance peaks of 2e^2/h under longitudinal magnetic fields exceeding 0.5 Tesla. Three-terminal conductance measurements confirm that topological protection suppresses local charge fluctuation decoherence exponentially with wire length.",
                score=0.94,
                metadata={"year": 2023, "domain": "quantum_computing"},
            ),
            SourceReference(
                source_id="src_qc_02_b",
                chunk_id="chunk_qc_02_2",
                title="Applied Physics Letters: Disorder-induced quasi-Majorana states in nanowires",
                content="Interface roughness and unintentional quantum dots near dielectric junctions can generate trivial Andreev bound states that mimic quantized zero-bias conductance. Cross-sectional transmission electron microscopy shows that atomic-layer molecular beam epitaxy reduces interface defect density below 1e10 cm^-2, preserving an induced topological superconducting gap of 0.2 meV.",
                score=0.88,
                metadata={"year": 2024, "domain": "quantum_computing"},
            ),
        ],
    ),
    BenchmarkTopic(
        topic_id="qc-03",
        title="Variational Quantum Algorithms for Constrained Portfolio Optimization",
        domain="quantum_computing",
        description="Evaluate the performance of Quantum Approximate Optimization Algorithm (QAOA) and warm-started VQE on quadratic unconstrained binary optimization (QUBO) portfolio models.",
        key_questions=[
            "How do barren plateaus affect gradient-based parameter updates in deep QAOA circuits?",
            "Does warm-starting with classical interior-point solutions improve approximation ratios?",
            "What are the realistic scaling limits on current noisy intermediate-scale (NISQ) hardware?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_qc_03_a",
                chunk_id="chunk_qc_03_1",
                title="IEEE Transactions on Quantum Engineering: Benchmark of QAOA on Markowitz portfolios",
                content="Benchmarking QAOA on 50-asset Markowitz mean-variance portfolio problems reveals that circuit depths beyond p=4 experience exponential variance suppression due to barren plateaus in random ansatz spaces. Standard parameter initialization yields approximation ratios of 0.72 on 27-qubit superconducting hardware, constrained primarily by two-qubit gate error rates of 0.4%.",
                score=0.92,
                metadata={"year": 2023, "domain": "quantum_computing"},
            ),
            SourceReference(
                source_id="src_qc_03_b",
                chunk_id="chunk_qc_03_2",
                title="Quantum Science and Technology: Warm-starting variational quantum optimizers",
                content="Warm-started QAOA initializes rotation angles using continuous relaxations solved via classical interior-point quadratic programming. On 100-asset S&P 500 subsets, warm-started QAOA achieves an approximation ratio of 0.94 at circuit depth p=2, outperforming greedy classical heuristics while saving 65% of optimization iterations compared to standard random starts.",
                score=0.90,
                metadata={"year": 2024, "domain": "quantum_computing"},
            ),
        ],
    ),
    # Domain 2: AI Architecture & LLM Inference
    BenchmarkTopic(
        topic_id="ai-01",
        title="Speculative Decoding and Medusa Multi-Head Architectures for LLM Inference",
        domain="ai_systems",
        description="Analyze latency reduction techniques in auto-regressive transformer inference using small draft models and non-autoregressive verification heads.",
        key_questions=[
            "What is the mathematical relationship between draft model acceptance rate and wall-clock speedup?",
            "How do tree-attention masks verify multiple candidate tokens in a single target forward pass?",
            "What memory and bandwidth tradeoffs arise when deploying Medusa heads versus separate draft models?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_ai_01_a",
                chunk_id="chunk_ai_01_1",
                title="ICML 2023: Fast inference from transformers via speculative decoding",
                content="Speculative decoding pairs a small draft model with a large target model, generating gamma candidate tokens in parallel and validating them in a single target forward pass using modified rejection sampling. When draft acceptance rate exceeds 75%, speculative decoding achieves a 2.3x to 2.8x wall-clock speedup on 70B parameter models without altering the output probability distribution.",
                score=0.96,
                metadata={"year": 2023, "domain": "ai_systems"},
            ),
            SourceReference(
                source_id="src_ai_01_b",
                chunk_id="chunk_ai_01_2",
                title="ArXiv 2024: Medusa: Simple framework for accelerating LLM generation with multiple heads",
                content="The Medusa architecture replaces separate draft models with multiple lightweight feedforward heads trained on top of the frozen base transformer trunk. Utilizing tree-structured attention masks, Medusa validates candidate token paths concurrently, delivering a 2.2x speedup on single-GPU deployments while avoiding the additional 4GB VRAM footprint of hosting a separate 7B draft model.",
                score=0.93,
                metadata={"year": 2024, "domain": "ai_systems"},
            ),
        ],
    ),
    BenchmarkTopic(
        topic_id="ai-02",
        title="Routing and Load Balancing in Sparse Mixture-of-Experts (MoE) Transformers",
        domain="ai_systems",
        description="Examine routing strategies, auxiliary loss functions, and expert capacity factors in high-parameter sparse mixture-of-experts architectures.",
        key_questions=[
            "How do auxiliary load balancing losses prevent router collapse into a single dominant expert?",
            "What are the latency penalties of inter-node all-to-all communication during expert dispatch?",
            "How does Expert Choice routing differ from standard top-k token-choice routing?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_ai_02_a",
                chunk_id="chunk_ai_02_1",
                title="NeurIPS 2022: Mixture-of-Experts with expert choice routing",
                content="Standard top-2 token routing assigns each token to its top two preferred experts, often leading to severe load imbalance where popular experts drop tokens when exceeding their capacity factor. In contrast, Expert Choice routing enables experts to independently select their top-C tokens, guaranteeing perfect load balance across experts and improving training throughput by 2.1x on 540B MoE models.",
                score=0.92,
                metadata={"year": 2022, "domain": "ai_systems"},
            ),
            SourceReference(
                source_id="src_ai_02_b",
                chunk_id="chunk_ai_02_2",
                title="SysML 2024: Communication-efficient distributed MoE inference at scale",
                content="In distributed MoE clusters spanning 64 nodes, all-to-all collective communication accounts for up to 48% of total inference step time. Implementing hierarchical tensor-expert parallelism and pipelined dispatch overlaps inter-socket NVLink transfers with inter-node InfiniBand RDMA, reducing communication stalls by 35% and sustaining 82% arithmetic intensity.",
                score=0.89,
                metadata={"year": 2024, "domain": "ai_systems"},
            ),
        ],
    ),
    BenchmarkTopic(
        topic_id="ai-03",
        title="Verifiable Retrieval-Augmented Generation via Natural Language Inference",
        domain="ai_systems",
        description="Investigate automated verification pipelines that evaluate claim-level entailment between generated responses and retrieved document passages.",
        key_questions=[
            "What are the failure modes of pure embedding-based similarity in detecting factual contradictions?",
            "How does decomposing generation into atomic claims improve citation precision?",
            "What is the latency and cost overhead of running fine-grained NLI verifiers in production RAG?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_ai_03_a",
                chunk_id="chunk_ai_03_1",
                title="ACL 2023: Evaluating citation precision and recall in retrieval-augmented language models",
                content="Vector cosine similarity between generated text and source chunks fails to identify subtle negation and numerical hallucinations, with false-positive attribution rates reaching 34%. Decomposing answers into atomic factual claims and evaluating them using calibrated Natural Language Inference (NLI) models increases citation precision from 62% to 91% on open-domain QA benchmarks.",
                score=0.95,
                metadata={"year": 2023, "domain": "ai_systems"},
            ),
            SourceReference(
                source_id="src_ai_03_b",
                chunk_id="chunk_ai_03_2",
                title="EMNLP 2023: Self-checking language models for hallucination detection",
                content="Fine-grained cross-encoder NLI verification adds an average latency of 180ms per query on 8-core CPU instances when processing up to 10 atomic claims. Applying selective two-stage verification—filtering obvious non-factual sentences with a fast regex classifier before invoking the cross-encoder—reduces verification cost by 55% while maintaining a 94% recall on critical hallucinations.",
                score=0.90,
                metadata={"year": 2023, "domain": "ai_systems"},
            ),
        ],
    ),
    # Domain 3: Renewable Energy & Materials Science
    BenchmarkTopic(
        topic_id="energy-01",
        title="Sulfide vs Oxide Solid Electrolytes for All-Solid-State Lithium Batteries",
        domain="renewable_energy",
        description="Compare ionic conductivities, interfacial stability with lithium metal anodes, and manufacturing scalability between sulfide- and oxide-based solid electrolytes.",
        key_questions=[
            "Why do sulfide electrolytes like Li10GeP2S12 achieve higher room-temperature ionic conductivity than garnet oxides?",
            "What passivation interlayers prevent dendritic lithium penetration in LLZO solid electrolytes?",
            "What are the moisture sensitivity challenges and toxic gas release risks of sulfide solid electrolytes?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_energy_01_a",
                chunk_id="chunk_energy_01_1",
                title="Nature Energy 2023: Comparative ionic transport in solid battery electrolytes",
                content="Sulfide-based solid electrolytes such as Li10GeP2S12 (LGPS) and argyrodite Li6PS5Cl exhibit exceptional room-temperature lithium-ion conductivities exceeding 1.2e-2 S/cm, comparable to liquid organic carbonates. However, sulfides react violently with ambient moisture to produce toxic hydrogen sulfide (H2S) gas, necessitating dry-room assembly with dew points below -50 degrees Celsius.",
                score=0.94,
                metadata={"year": 2023, "domain": "renewable_energy"},
            ),
            SourceReference(
                source_id="src_energy_01_b",
                chunk_id="chunk_energy_01_2",
                title="Advanced Materials 2024: Garnet-type oxide electrolytes for dendrite-free lithium anodes",
                content="Garnet-structured oxide electrolytes such as Li7La3Zr2O12 (LLZO) possess superior chemical stability against metallic lithium and allow air handling. However, LLZO exhibits lower bulk conductivity of 1.0e-3 S/cm and high interfacial resistance. Applying a 5-nanometer amorphous atomic-layer-deposited zinc oxide (ZnO) interlayer reduces interfacial area-specific resistance from 1200 ohm-cm^2 to 12 ohm-cm^2.",
                score=0.91,
                metadata={"year": 2024, "domain": "renewable_energy"},
            ),
        ],
    ),
    BenchmarkTopic(
        topic_id="energy-02",
        title="Degradation Mechanisms in Perovskite-Silicon Tandem Photovoltaic Cells",
        domain="renewable_energy",
        description="Investigate phase segregation, halide migration, and barrier encapsulation technologies determining the operational lifetime of monolithic perovskite-silicon tandems.",
        key_questions=[
            "What causes photo-induced halide segregation in mixed-halide perovskite absorber layers?",
            "How do atomic-layer deposited metal oxide barrier layers prevent moisture ingress?",
            "What testing protocols are required to validate 25-year operational stability according to IEC 61215 standards?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_energy_02_a",
                chunk_id="chunk_energy_02_1",
                title="Science 2024: Mitigating phase segregation in wide-bandgap perovskite top cells",
                content="Monolithic two-terminal perovskite-silicon tandem solar cells have demonstrated certified power conversion efficiencies of 33.9%. Under continuous 1-sun illumination, mixed iodide-bromide perovskites suffer from light-induced halide segregation, forming low-bandgap iodide-rich clusters that act as non-radiative recombination centers and reduce open-circuit voltage by up to 140 mV.",
                score=0.96,
                metadata={"year": 2024, "domain": "renewable_energy"},
            ),
            SourceReference(
                source_id="src_energy_02_b",
                chunk_id="chunk_energy_02_2",
                title="Progress in Photovoltaics 2024: Accelerated damp-heat testing of encapsulated tandems",
                content="Encapsulating tandems with double-barrier layers comprising 20 nm atomic-layer-deposited tin oxide (SnO2) capped with polyisobutylene (PIB) edge seals completely blocks moisture ingress. Devices subjected to IEC 61215 damp-heat testing (85 degrees Celsius and 85% relative humidity) retained 95% of their initial efficiency after 1,500 hours of continuous accelerated stress.",
                score=0.89,
                metadata={"year": 2024, "domain": "renewable_energy"},
            ),
        ],
    ),
    BenchmarkTopic(
        topic_id="energy-03",
        title="Proton Exchange Membrane (PEM) Electrolyzers for Green Hydrogen Production",
        domain="renewable_energy",
        description="Assess iridium catalyst loading reduction, membrane degradation, and levelized cost of hydrogen (LCOH) in dynamic PEM electrolysis powered by offshore wind.",
        key_questions=[
            "What catalyst supports allow reducing anode iridium loading below 0.2 mg/cm^2?",
            "How does dynamic load cycling from wind power accelerate chemical membrane thinning?",
            "What electricity price and capital expenditure are required to achieve an LCOH below $2.00 per kg H2?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_energy_03_a",
                chunk_id="chunk_energy_03_1",
                title="JACS 2023: Low-loading iridium oxide catalysts for acidic water oxidation",
                content="Standard commercial PEM electrolyzers rely on anode iridium catalyst loadings of 1.5 to 2.0 mg/cm^2, posing severe supply chain bottlenecks for gigawatt-scale scaling. Synthesizing atom-dispersed iridium on conductive antimony-doped tin oxide (ATO) supports enables reducing iridium loading to 0.18 mg/cm^2 while maintaining a cell voltage of 1.78 V at a current density of 2.0 A/cm^2.",
                score=0.93,
                metadata={"year": 2023, "domain": "renewable_energy"},
            ),
            SourceReference(
                source_id="src_energy_03_b",
                chunk_id="chunk_energy_03_2",
                title="International Journal of Hydrogen Energy: Techno-economic analysis of offshore wind-PEM systems",
                content="Techno-economic modeling indicates that achieving a levelized cost of hydrogen (LCOH) below $2.00/kg requires an average electricity input cost below $0.03/kWh and electrolyzer stack capital expenditures under $350/kW. High-frequency power transients accelerate cerium radical scavenger migration in Nafion membranes, reducing operational stack lifetime from 80,000 to 52,000 hours.",
                score=0.91,
                metadata={"year": 2024, "domain": "renewable_energy"},
            ),
        ],
    ),
    # Domain 4: Biomedical AI & Precision Medicine
    BenchmarkTopic(
        topic_id="bio-01",
        title="Deep Learning and Diffusion Models for De Novo Allosteric Drug Design",
        domain="biomedical_ai",
        description="Review generative diffusion models and equivariant graph neural networks for designing small-molecule inhibitors targeting non-catalytic allosteric sites.",
        key_questions=[
            "How do SE(3)-equivariant diffusion models generate 3D molecular conformers conditioned on protein pocket coordinates?",
            "What advantages do allosteric inhibitors offer regarding kinase selectivity over orthosteric ATP-competitive inhibitors?",
            "What computational scoring functions reliably filter out false-positive high-affinity binders?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_bio_01_a",
                chunk_id="chunk_bio_01_1",
                title="Cell Systems 2023: Structure-conditioned generative diffusion for allosteric drug discovery",
                content="Targeting allosteric regulatory pockets provides high isoform selectivity across the human kinome, circumventing the off-target toxicity typical of ATP-competitive inhibitors. Applying SE(3)-equivariant continuous-time diffusion models generates drug-like conformers that dock directly into transient cryptic pockets identified from molecular dynamics simulations, achieving a 22% experimental hit rate in biochemical assays.",
                score=0.95,
                metadata={"year": 2023, "domain": "biomedical_ai"},
            ),
            SourceReference(
                source_id="src_bio_01_b",
                chunk_id="chunk_bio_01_2",
                title="Journal of Chemical Information and Modeling: Benchmarking protein-ligand scoring functions",
                content="Standard empirical scoring functions (e.g. Glide SP, AutoDock Vina) exhibit high false-positive rates of 68% when evaluating generated allosteric ligands. Combining physics-based MM-GBSA binding free-energy calculations with ensemble machine-learned scoring functions improves the area under the receiver operating characteristic curve (AUROC) to 0.89 for identifying true sub-micromolar binders.",
                score=0.90,
                metadata={"year": 2024, "domain": "biomedical_ai"},
            ),
        ],
    ),
    BenchmarkTopic(
        topic_id="bio-02",
        title="Self-Supervised Foundation Models in Computational Pathology",
        domain="biomedical_ai",
        description="Examine gigapixel whole-slide image analysis using Vision Transformers pre-trained via self-supervised learning (DINOv2) for cancer subtyping and biomarker prediction.",
        key_questions=[
            "How do hierarchical Vision Transformers process gigapixel whole-slide images without out-of-memory errors?",
            "Can self-supervised pathology foundation models predict gene mutation status (e.g., KRAS, EGFR) directly from H&E stains?",
            "What domain-shift mitigation strategies address stain variation across hospital laboratories?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_bio_02_a",
                chunk_id="chunk_bio_02_1",
                title="Nature Medicine 2024: A general-purpose self-supervised foundation model for pathology",
                content="Whole-slide computational pathology images exceed 100,000 by 100,000 pixels, preventing direct end-to-end convolutional training. Pre-training a Vision Transformer (ViT-H/14) on 1.2 billion tissue patches using DINOv2 self-distillation extracts rich patch embeddings. An attention-based multiple-instance learning (MIL) aggregator then processes slide-level bags, achieving an AUROC of 0.94 for colorectal carcinoma subtyping.",
                score=0.96,
                metadata={"year": 2024, "domain": "biomedical_ai"},
            ),
            SourceReference(
                source_id="src_bio_02_b",
                chunk_id="chunk_bio_02_2",
                title="The Lancet Digital Health: Predicting driver mutations from routine histology",
                content="Self-supervised pathology foundation models accurately predict clinically actionable driver mutations directly from hematoxylin and eosin (H&E) histology slides. On multi-center non-small-cell lung cancer cohorts, the model predicted EGFR mutation status with an AUROC of 0.86 and KRAS status with an AUROC of 0.83, without requiring prior stain normalization.",
                score=0.92,
                metadata={"year": 2024, "domain": "biomedical_ai"},
            ),
        ],
    ),
    BenchmarkTopic(
        topic_id="bio-03",
        title="CRISPR-Cas9 Off-Target Cleavage Prediction Using Transformer Models",
        domain="biomedical_ai",
        description="Assess deep sequence models for predicting unintended genomic double-strand breaks caused by single-guide RNA mismatches and DNA bulges.",
        key_questions=[
            "How do chromatin accessibility features (ATAC-seq) modulate in vivo off-target cleavage probability?",
            "What is the predictive accuracy of transformer models compared to energy-based biophysical models (e.g. CFD score)?",
            "How does high-fidelity Cas9 (e.g., SpCas9-HF1, HiFi Cas9) alter the distribution of off-target edits?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_bio_03_a",
                chunk_id="chunk_bio_03_1",
                title="Nature Biotechnology 2023: Deep learning prediction of genome-wide CRISPR off-targets",
                content="Accurate off-target prediction requires modeling both sequence complementarity and epigenetic context. Transformer-based models incorporating 23-nucleotide guide-target alignment matrices with local DNase I and ATAC-seq accessibility achieve an F1-score of 0.84 on GUIDE-seq validated cleavage sites, significantly outperforming traditional Cutting Frequency Determination (CFD) biophysical matrices (F1=0.61).",
                score=0.94,
                metadata={"year": 2023, "domain": "biomedical_ai"},
            ),
        ],
    ),
    # Domain 5: Distributed Systems & Consensus
    BenchmarkTopic(
        topic_id="dist-01",
        title="Byzantine Fault Tolerant (BFT) Consensus under Asynchronous Network Partitions",
        domain="distributed_systems",
        description="Analyze safety and liveness tradeoffs in partially synchronous and asynchronous BFT consensus protocols (e.g. HotStuff, Narwhal-Tusk, DAG-based consensus).",
        key_questions=[
            "How does decoupled consensus in DAG-based architectures (Narwhal-Tusk) circumvent mempool congestion?",
            "What happens to liveness versus safety during sustained network partition beyond the known bound Delta?",
            "What is the communication complexity of linear-view change protocols compared to PBFT's quadratic cost?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_dist_01_a",
                chunk_id="chunk_dist_01_1",
                title="ACM PODC 2022: Narwhal and Tusk: A DAG-based consensus architecture",
                content="Traditional BFT protocols like PBFT intertwine transaction dissemination with block ordering, creating bandwidth bottlenecks at the designated primary node. The Narwhal-Tusk architecture decouples the mempool into a high-throughput reliable broadcast DAG (Narwhal) from the zero-message consensus layer (Tusk), sustaining 160,000 transactions per second with sub-second finality over 100 geographically distributed validators.",
                score=0.96,
                metadata={"year": 2022, "domain": "distributed_systems"},
            ),
            SourceReference(
                source_id="src_dist_01_b",
                chunk_id="chunk_dist_01_2",
                title="IEEE S&P 2023: HotStuff-2: Efficient two-phase BFT under optimistic responsiveness",
                content="Under the Dwork-Mimno-Stockmeyer partial synchrony model, protocols guarantee safety under complete asynchrony but liveness is suspended during network partitions exceeding Delta. HotStuff-2 reduces view-change communication complexity to O(n) using threshold signature accumulators, reducing latency by 40% compared to three-phase classic HotStuff when recovering from leader failovers.",
                score=0.91,
                metadata={"year": 2023, "domain": "distributed_systems"},
            ),
        ],
    ),
    BenchmarkTopic(
        topic_id="dist-02",
        title="Conflict-Free Replicated Data Types (CRDTs) in Local-First Collaborative Software",
        domain="distributed_systems",
        description="Evaluate state-based versus operation-based CRDTs, fractional indexing, and garbage collection mechanisms for real-time rich-text editing.",
        key_questions=[
            "How do tree-based sequence CRDTs (e.g. Yjs, Fugue, RGA) prevent interleaving anomalies during concurrent typing?",
            "What is the memory overhead of maintaining tombstone markers for deleted characters over extended editing sessions?",
            "How do state-based state-vector exchange algorithms sync offline divergence efficiently?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_dist_02_a",
                chunk_id="chunk_dist_02_1",
                title="VLDB 2023: Performance analysis of sequence CRDT implementations in collaborative editing",
                content="Real-time collaborative text editing requires sequence CRDTs that maintain deterministic order without centralized locking. While naive replicated growable arrays (RGA) incur significant metadata bloat, the Yjs and Fugue algorithms use item-based linked lists with Run-Length Encoding (RLE) to compress adjacent character insertions. This reduces memory consumption to under 1.8x raw document size.",
                score=0.95,
                metadata={"year": 2023, "domain": "distributed_systems"},
            ),
            SourceReference(
                source_id="src_dist_02_b",
                chunk_id="chunk_dist_02_2",
                title="ACM TOCS 2024: Safe tombstone pruning in decentralized causal CRDTs",
                content="Retaining deleted character tombstones prevents duplicate re-insertion when lagging replicas sync, but causes memory leaks in long-lived documents. Implementing vector clock epoch boundaries allows replicas to safely garbage-collect tombstones once all peers have acknowledged causal delivery past the pruning horizon, reducing state vector payload sizes by 74%.",
                score=0.89,
                metadata={"year": 2024, "domain": "distributed_systems"},
            ),
        ],
    ),
    BenchmarkTopic(
        topic_id="dist-03",
        title="High-Dimensional Approximate Nearest Neighbor Search in Vector Databases",
        domain="distributed_systems",
        description="Compare Hierarchical Navigable Small World (HNSW), Inverted File with Product Quantization (IVF-PQ), and graph partitioning for billion-scale vector retrieval.",
        key_questions=[
            "How does HNSW's multi-layer skip-list graph hierarchy achieve logarithmic query scaling?",
            "What are the memory-bandwidth savings and recall tradeoffs of product quantization (PQ) versus scalar quantization (SQ)?",
            "How do distributed vector databases shard and update dynamic graph indices without degrading search recall?",
        ],
        gold_sources=[
            SourceReference(
                source_id="src_dist_03_a",
                chunk_id="chunk_dist_03_1",
                title="IEEE TPAMI 2020: Efficient and robust approximate nearest neighbors using HNSW graphs",
                content="Hierarchical Navigable Small World (HNSW) graphs construct multi-layer proximity networks where top layers feature long-range edges for coarse exploration and lower layers contain dense clusters for fine-grained local search. On 768-dimensional embedding benchmarks, HNSW achieves 98% 10-recall at 12,000 queries per second, outperforming tree-based KD-trees by two orders of magnitude.",
                score=0.96,
                metadata={"year": 2020, "domain": "distributed_systems"},
            ),
            SourceReference(
                source_id="src_dist_03_b",
                chunk_id="chunk_dist_03_2",
                title="ACM SIGMOD 2024: Scalable vector database architectures for billion-scale multimodal search",
                content="Storing raw 1536-dimensional float32 vectors for 1 billion items requires 6.14 TB of RAM. Applying 8x Product Quantization (PQ) compresses vectors into compact 96-byte codes, fitting the index into 96 GB of memory with an acceptable 4.2% drop in top-10 recall. Distributing partitions across Raft consensus groups with asynchronous graph consolidation prevents index lockups during high-frequency ingestion.",
                score=0.92,
                metadata={"year": 2024, "domain": "distributed_systems"},
            ),
        ],
    ),
]


def get_benchmark_dataset() -> list[BenchmarkTopic]:
    """Return the complete 15-topic benchmark research dataset."""
    return BENCHMARK_TOPICS


def get_topic_by_id(topic_id: str) -> BenchmarkTopic:
    """Retrieve a specific benchmark topic by its unique ID.

    Args:
        topic_id: Unique topic identifier (e.g. 'qc-01').

    Returns:
        The matching BenchmarkTopic.

    Raises:
        KeyError: If topic_id is not found in the dataset.
    """
    for topic in BENCHMARK_TOPICS:
        if topic.topic_id == topic_id:
            return topic
    raise KeyError(f"Benchmark topic '{topic_id}' not found in dataset.")
