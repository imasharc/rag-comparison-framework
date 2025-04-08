"""
Prompting techniques package initialization.
"""
from rag_variants.prompting.chain_of_thought import ChainOfThoughtRAG
from rag_variants.prompting.few_shot import FewShotRAG
from rag_variants.prompting.role_based import RoleBasedRAG

# List of all available prompting techniques
AVAILABLE_PROMPTING_TECHNIQUES = [
    ChainOfThoughtRAG,
    FewShotRAG,
    RoleBasedRAG
]