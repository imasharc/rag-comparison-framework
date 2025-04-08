"""
RAG variants package initialization.
"""
from rag_variants.base_variant import RAGVariant, BaselineRAG
from rag_variants.query_expansion import QueryExpansionRAG
from rag_variants.hybrid_search import HybridSearchRAG
from rag_variants.adaptive_chunking import AdaptiveChunkingRAG
from rag_variants.prompting.chain_of_thought import ChainOfThoughtRAG
from rag_variants.prompting.few_shot import FewShotRAG
from rag_variants.prompting.role_based import RoleBasedRAG

# List of all available variants
AVAILABLE_VARIANTS = [
    BaselineRAG,
    QueryExpansionRAG,
    HybridSearchRAG,
    AdaptiveChunkingRAG,
    ChainOfThoughtRAG,
    FewShotRAG, 
    RoleBasedRAG
]