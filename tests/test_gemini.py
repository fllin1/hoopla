from google.genai import Client


class TestGeminiClient:
    def test_client_initialization(self, gemini_client: Client) -> None:
        assert gemini_client is not None

    def test_gemini_generation(self, gemini_client: Client) -> None:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
        )
        assert response is not None
        assert response.text is not None
        print("\n\nModel Response:", response.text)
        assert response.usage_metadata is not None
        print("Prompt Tokens:", response.usage_metadata.prompt_token_count)
        print("Response Tokens:", response.usage_metadata.candidates_token_count)
