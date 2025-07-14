from transformers import AutoTokenizer, pipeline
from typing import List

def generate_fallback_persona(username: str, posts: List[dict], comments: List[dict]) -> tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Generates a fallback persona based on scraped data, focused on gaming context."""
    summary = ["* User is an active commenter, primarily engaged in gaming-related discussions on subreddits like r/ManorLords."]
    behavior = []
    frustrations = []
    motivations = []
    goals = []
    personality = []

    gaming_subreddits = {'manorlords', 'chatgpt'} 

    for item in posts + comments:
        text = item.get("text", "").lower()
        citation = item.get("url", "No citation available")
        subreddit = citation.split('/r/')[1].split('/')[0].lower() if '/r/' in citation else ''
        if subreddit in gaming_subreddits:
            if "issue" in text or "problem" in text or "burned" in text:
                frustrations.append(f"* Frustrated with game mechanics or bugs  * Citation: {citation}")
            if "play" in text or "game" in text:
                behavior.append(f"* Regularly plays and discusses video games  * Citation: {citation}")
            if "hope" in text or "new" in text:
                motivations.append(f"* Motivated to see game improvements  * Citation: {citation}")
                goals.append(f"* Aims to enhance gaming experience with new features  * Citation: {citation}")
            if "incredible" in text or "great" in text:
                personality.append(f"* [Introvert vs. Extrovert: Leans Extrovert - enthusiastic and supportive]  * Citation: {citation}")
                quote = next((line for line in item.get("text", "").split('\n') if line), "No quote available")
                personality.append(f"* User Quote: {quote[:50]}...  * Citation: {citation}")

    if not behavior:
        behavior.append("* No specific behaviors identified.  * Citation: No citation available")
    if not frustrations:
        frustrations.append("* No frustrations identified.  * Citation: No citation available")
    if not motivations:
        motivations.append("* No motivations identified.  * Citation: No citation available")
    if not goals:
        goals.append("* No goals identified.  * Citation: No citation available")
    if not personality:
        personality.append("* No personality traits identified.  * Citation: No citation available")
        personality.append("* User Quote: No quote available.  * Citation: No citation available")

    return summary, behavior, frustrations, motivations, goals, personality

def build_persona_with_huggingface(username: str, posts: List[dict], comments: List[dict]) -> str:
    """
    Generates a user persona with a structured format and saves it to a text file,
    including citations for each characteristic based on scraped Reddit data.
    """
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_input_tokens = 1024 - 256  
    generator = pipeline("text-generation", model=model_name, device=-1)

    user_data_entries = []
    for item in posts + comments:
        if isinstance(item, dict) and item.get("text", "").strip():
            text = item.get("text", "")
            citation = item.get("url", "No citation available")
            user_data_entries.append(f"{text}\nCitation: {citation}")

    prompt_prefix = (
        f"Reddit user u/{username} has posted and commented the following:\n"
        f"{'-'*40}\n"
    )

    prompt = prompt_prefix
    for entry in user_data_entries:
        test_prompt = prompt + entry + "\n"
        input_ids = tokenizer.encode(test_prompt, truncation=True, max_length=max_input_tokens)
        if len(input_ids) <= max_input_tokens:
            prompt = test_prompt
        else:
            break

    if prompt == prompt_prefix:
        prompt += "No user activity data available within token limit.\n"

    prompt += (
        "\nBased on the above, create a detailed user persona in the following structured format:\n"
        "Header: u/{username}\n"
        "Age: [Estimate based on context, or 'Unknown']\n"
        "Occupation: [Guess based on context, or 'Unknown']\n"
        "Status: [e.g., Single, Married, or 'Unknown']\n"
        "Location: [Infer from subreddits or context, or 'Unknown']\n"
        "Tier: [e.g., Early Adopter, Mainstream, or 'Unknown']\n"
        "Archetype: [e.g., The Creator, The Explorer, or 'Unknown']\n"
        "\n"
        "**Summary:**\n"
        "* A brief, one-paragraph overview of the user’s general disposition and main focus based on their activity.\n"
        "Example: * User is a tech enthusiast who frequently shares coding tips.\n"
        "\n"
        "**Behavior & Habits:**\n"
        "* [List specific behaviors or habits observed, e.g., 'Frequently discusses local issues']  * Citation: [URL]\n"
        "Example: * Regularly posts about gaming strategies  * Citation: https://example.com\n"
        "\n"
        "**Frustrations:**\n"
        "* [List specific frustrations, e.g., 'Dealing with police bribery']  * Citation: [URL]\n"
        "Example: * Frustrated with slow internet speeds  * Citation: https://example.com\n"
        "\n"
        "**Motivations:**\n"
        "* [List motivations, e.g., 'Seeking healthy food options']  * Citation: [URL]\n"
        "Example: * Motivated to learn new skills  * Citation: https://example.com\n"
        "\n"
        "**Goals & Needs:**\n"
        "* [List goals and needs, e.g., 'To improve food quality awareness']  * Citation: [URL]\n"
        "Example: * Aims to build a personal website  * Citation: https://example.com\n"
        "\n"
        "**Personality:**\n"
        "* [Assess personality traits, e.g., 'Introvert vs. Extrovert', 'Sensing vs. Intuition', etc., using a scale or binary choice]  * Citation: [URL]\n"
        "* User Quote: [A direct quote from their posts/comments that reflects their personality]  * Citation: [URL]\n"
        "Example: * [Introvert vs. Extrovert: Extrovert - enjoys community discussions]  * Citation: https://example.com\n"
        "* User Quote: 'I love chatting with others!'  * Citation: https://example.com\n"
        "\n"
        "For **EVERY characteristic** under Behavior & Habits, Frustrations, Motivated to see game improvementsons, Goals & Needs, and Personality, you **MUST** include the specific 'Citation' URL from the data that supports it. Base all conclusions strictly on the provided data. Do not invent information beyond the examples."
    )

    try:
        input_ids = tokenizer.encode(prompt, truncation=True, max_length=max_input_tokens)
        truncated_prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"Token count: {len(input_ids)}, Prompt preview: {truncated_prompt[:200]}...") 

        result = generator(
            truncated_prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = result[0]['generated_text']

        persona = f"u/{username}\n"
        persona += "Age: Unknown\n"
        persona += "Occupation: Unknown\n"
        persona += "Status: Unknown\n"
        persona += "Location: [Infer from subreddits or context, or 'Unknown']\n"
        persona += "Tier: Unknown\n"
        persona += "Archetype: Unknown\n\n"

        lines = generated_text.split('\n')
        summary = []
        behavior = []
        frustrations = []
        motivations = []
        goals = []
        personality = []
        current_section = None
        current_citation = None

        for line in lines:
            if line.startswith("**Summary:**"):
                current_section = "summary"
            elif line.startswith("**Behavior & Habits:**"):
                current_section = "behavior"
            elif line.startswith("**Frustrations:**"):
                current_section = "frustrations"
            elif line.startswith("**Motivations:**"):
                current_section = "motivations"
            elif line.startswith("**Goals & Needs:**"):
                current_section = "goals"
            elif line.startswith("**Personality:**"):
                current_section = "personality"
            elif line.strip():
                if current_section:
                    if "Citation:" in line:
                        current_citation = line.replace("Citation:", "").strip()
                    elif line.startswith("*") and current_citation:
                        if current_section == "summary" and line.startswith("*"):
                            summary.append(line)
                        elif current_section == "behavior":
                            behavior.append(f"{line}  * Citation: {current_citation or 'No citation available'}")
                        elif current_section == "frustrations":
                            frustrations.append(f"{line}  * Citation: {current_citation or 'No citation available'}")
                        elif current_section == "motivations":
                            motivations.append(f"{line}  * Citation: {current_citation or 'No citation available'}")
                        elif current_section == "goals":
                            goals.append(f"{line}  * Citation: {current_citation or 'No citation available'}")
                        elif current_section == "personality":
                            personality.append(line if "User Quote:" not in line else f"{line}  * Citation: {current_citation or 'No citation available'}")
                        current_citation = None
                    elif current_section == "summary" and not line.startswith("*"):
                        summary.append(line)
                    elif current_section == "personality" and "User Quote:" in line:
                        personality.append(line)

        if not any([summary, behavior, frustrations, motivations, goals, personality]):
            summary, behavior, frustrations, motivations, goals, personality = generate_fallback_persona(username, posts, comments)

        persona += "**Summary:**\n" + "\n".join(summary) + "\n\n"
        persona += "**Behavior & Habits:**\n" + "\n".join(behavior) + "\n\n"
        persona += "**Frustrations:**\n" + "\n".join(frustrations) + "\n\n"
        persona += "**Motivations:**\n" + "\n".join(motivations) + "\n\n"
        persona += "**Goals & Needs:**\n" + "\n".join(goals) + "\n\n"
        persona += "**Personality:**\n" + "\n".join(personality)

        with open(f"output\\{username}_persona.txt", "w", encoding="utf-8") as f:
            f.write(persona)
        print(f"Persona saved to: output\\{username}_persona.txt")

        return persona
    except Exception as e:
        error_msg = f"Error generating persona: {str(e)}"
        with open(f"output\\{username}_persona.txt", "w", encoding="utf-8") as f:
            f.write(error_msg)
        print(f"Persona saved to: output\\{username}_persona.txt with error: {str(e)}")
        return error_msg