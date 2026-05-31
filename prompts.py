def get_prompt_template(q_type):
    if q_type == "Troubleshooting":
        return "Answer based on context. Format: 1. Possible Causes, 2. Step-by-Step Solution, 3. When to Escalate."
    elif q_type == "Comparison":
        return "Answer based on context. Format: Markdown table for features, Key Differences, and Recommendation."
    return "Answer based on context. Format: Direct Answer, Explanation, Additional Notes."