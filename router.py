def classify_query(query, llm=None):
    query_lower = query.lower()
    
    # 1. Rule-based Layer (Fast & Deterministic)
    if any(word in query_lower for word in ["compare", "vs", "difference", "versus"]):
        return "Comparison"
    if any(word in query_lower for word in ["fix", "how to", "problem", "issue", "why", "reset", "troubleshoot"]):
        return "Troubleshooting"
    
    # 2. LLM-based Fallback Layer (Intelligent analysis)
    if llm:
        system_prompt = "Classify the user's intent into 'Troubleshooting', 'Comparison', or 'General'. Return only the category name."
        response = llm.invoke(f"{system_prompt}\nQuery: {query}")
        return response.content.strip()
    
    return "General"