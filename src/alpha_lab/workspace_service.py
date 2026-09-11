from pathlib import Path


def execute_request(*, root, request_id, user_request, api_key):
    """Run one research request; API key is not part of saved input."""
    from google import genai
    from google.genai import types
    from alpha_lab import research_agents

    api_key = api_key.strip()
    if not api_key:
        raise ValueError("Missing API key.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60000),
    )
    try:
        return research_agents.run_research(
            client=client,
            model="gemini-3.6-flash",
            root=Path(root),
            user_request=user_request,
            request_id=request_id,
        )
    finally:
        client.close()
