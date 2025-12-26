from sentence_transformers import SentenceTransformer, util

similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def check_risk(user_clause, standard_clauses):
    """
    Calculates how 'far' a clause is from standard fair-use language.
    High distance = High risk.
    """
    user_embedding = similarity_model.encode(user_clause)
    standard_embeddings = similarity_model.encode(standard_clauses)
    
    # Calculate cosine similarity
    scores = util.cos_sim(user_embedding, standard_embeddings)
    max_similarity = torch.max(scores).item()
    
    # If similarity is < 0.6, the language is highly non-standard/risky
    return "High Risk" if max_similarity < 0.6 else "Standard"
