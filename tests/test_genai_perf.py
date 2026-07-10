import unittest

from bench.genai_perf import build_genai_perf_command


class GenAIPerfCommandTests(unittest.TestCase):
    def test_builds_openai_profile_command(self) -> None:
        cmd = build_genai_perf_command(
            executable="genai-perf",
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            model="qwen3-8b",
            input_tokens=256,
            output_tokens=128,
            num_prompts=32,
        )

        self.assertEqual(cmd[0], "genai-perf")
        self.assertIn("profile", cmd)
        self.assertIn("--service-kind", cmd)
        self.assertIn("openai", cmd)
        self.assertIn("--endpoint", cmd)
        self.assertIn("http://127.0.0.1:8000/v1/chat/completions", cmd)
        self.assertIn("--num-prompts", cmd)
        self.assertIn("32", cmd)

    def test_builds_aiperf_alias_command(self) -> None:
        cmd = build_genai_perf_command(
            executable="aiperf",
            endpoint="http://127.0.0.1:8000/v1/completions",
            model="qwen3-8b",
            input_tokens=128,
            output_tokens=64,
            num_prompts=16,
        )

        self.assertEqual(cmd[0], "aiperf")
        self.assertIn("profile", cmd)
        self.assertIn("--endpoint", cmd)
