"""Chat helper utilities"""

def build_llm_conversation_text(messages_list: list) -> str:
    """
    Formats the chat history for the LLM prompt.
    
    Args:
        messages_list: List of message dicts
        
    Returns:
        str: Formatted conversation history string
    """
    history = []
    for msg in messages_list:
        role = "User" if msg['role'] == 'user' else "Assistant"
        content = msg.get('content', '') or msg.get('plan_text', '')
        history.append(f"{role}: {content}")
    return "\n".join(history)
