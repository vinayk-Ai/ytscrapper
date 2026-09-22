import os
import types
import unittest

import services


class TestEmbeddingApi(unittest.TestCase):
    def test_embed_query_uses_groq_embedding_api(self):
        calls = {}

        class FakeEmbeddings:
            def create(self, *, model, input):
                calls["model"] = model
                calls["input"] = input
                return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[0.1, 0.2, 0.3])])

        class FakeGroq:
            def __init__(self, api_key):
                self.embeddings = FakeEmbeddings()

        old_provider = os.environ.get("EMBEDDING_PROVIDER")
        old_key = os.environ.get("GROQ_API_KEY")
        try:
            os.environ["GROQ_API_KEY"] = "test-key"
            os.environ["EMBEDDING_PROVIDER"] = "auto"
            original_get_model = services.get_embedding_model
            services.Groq = FakeGroq
            services.get_embedding_model = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("local model should not be used for Groq embeddings")
            )

            result = services.embed_query("two pointers")

            self.assertEqual(result, [0.1, 0.2, 0.3])
            self.assertEqual(calls["model"], "text-embedding-3-small")
            self.assertEqual(calls["input"], ["two pointers"])
        finally:
            if old_provider is None:
                os.environ.pop("EMBEDDING_PROVIDER", None)
            else:
                os.environ["EMBEDDING_PROVIDER"] = old_provider
            if old_key is None:
                os.environ.pop("GROQ_API_KEY", None)
            else:
                os.environ["GROQ_API_KEY"] = old_key
            services.get_embedding_model = original_get_model


if __name__ == "__main__":
    unittest.main()
