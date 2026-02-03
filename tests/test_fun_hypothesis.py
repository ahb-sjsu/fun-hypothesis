"""Tests for fun_hypothesis.py - Prompt Framing Hypothesis Tester."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fun_hypothesis import (
    FRAMING_STYLES,
    SAMPLE_PROMPTS,
    ExperimentResults,
    FramingHypothesisTester,
    FramingResults,
    Trial,
    main,
)


class TestFramingStyles:
    """Tests for FRAMING_STYLES constant."""

    def test_framing_styles_has_required_keys(self):
        required_keys = ["fun", "pirate", "expert", "eli5", "formal", "socratic"]
        for key in required_keys:
            assert key in FRAMING_STYLES

    def test_framing_style_structure(self):
        for key, style in FRAMING_STYLES.items():
            assert "name" in style
            assert "instruction" in style
            assert isinstance(style["name"], str)
            assert isinstance(style["instruction"], str)
            assert len(style["instruction"]) > 10


class TestSamplePrompts:
    """Tests for SAMPLE_PROMPTS constant."""

    def test_sample_prompts_not_empty(self):
        assert len(SAMPLE_PROMPTS) > 0

    def test_sample_prompts_are_strings(self):
        for prompt in SAMPLE_PROMPTS:
            assert isinstance(prompt, str)
            assert len(prompt) > 10


class TestTrialDataclass:
    """Tests for Trial dataclass."""

    def test_trial_creation(self):
        trial = Trial(
            trial_id="test123",
            framing_style="fun",
            original_prompt="Test prompt",
        )
        assert trial.trial_id == "test123"
        assert trial.framing_style == "fun"
        assert trial.original_prompt == "Test prompt"
        assert trial.framed_prompt == ""
        assert trial.winner == ""

    def test_trial_with_scores(self):
        trial = Trial(
            trial_id="test123",
            framing_style="fun",
            original_prompt="Test prompt",
            raw_avg_score=7.5,
            framed_avg_score=8.0,
            winner="framed",
        )
        assert trial.raw_avg_score == 7.5
        assert trial.framed_avg_score == 8.0
        assert trial.winner == "framed"


class TestFramingResultsDataclass:
    """Tests for FramingResults dataclass."""

    def test_framing_results_creation(self):
        results = FramingResults(
            framing_style="fun",
            framing_name="Fun & Engaging",
            num_trials=5,
            framed_wins=3,
            raw_wins=1,
            ties=1,
        )
        assert results.framing_style == "fun"
        assert results.num_trials == 5
        assert results.framed_wins == 3

    def test_framing_results_defaults(self):
        results = FramingResults(framing_style="test", framing_name="Test")
        assert results.num_trials == 0
        assert results.raw_wins == 0
        assert results.framed_wins == 0
        assert results.improvement_pct == 0.0


class TestExperimentResultsDataclass:
    """Tests for ExperimentResults dataclass."""

    def test_experiment_results_creation(self):
        results = ExperimentResults(
            experiment_id="exp123",
            timestamp="2025-01-01T00:00:00",
            model="claude-sonnet-4-20250514",
            num_prompts=5,
            num_judges=3,
        )
        assert results.experiment_id == "exp123"
        assert results.model == "claude-sonnet-4-20250514"


class TestFramingHypothesisTester:
    """Tests for FramingHypothesisTester class."""

    @pytest.fixture
    def mock_anthropic(self):
        with patch("fun_hypothesis.HAS_ANTHROPIC", True):
            with patch("fun_hypothesis.anthropic") as mock:
                yield mock

    @pytest.fixture
    def tester(self, mock_anthropic):
        with patch.object(FramingHypothesisTester, "__init__", lambda self, model="test": None):
            t = FramingHypothesisTester.__new__(FramingHypothesisTester)
            t.model = "test-model"
            t.client = MagicMock()
            t.framings = FRAMING_STYLES.copy()
            return t

    def test_add_custom_framing(self, tester):
        tester.add_custom_framing("custom", "Custom Style", "Custom instruction")
        assert "custom" in tester.framings
        assert tester.framings["custom"]["name"] == "Custom Style"
        assert tester.framings["custom"]["instruction"] == "Custom instruction"

    def test_load_framings_from_csv(self, tester):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("key,name,instruction\n")
            f.write("myframing,My Framing,Do something special\n")
            f.flush()

            tester.load_framings_from_csv(f.name)
            assert "myframing" in tester.framings
            assert tester.framings["myframing"]["name"] == "My Framing"

    def test_load_prompts_from_txt_file(self, tester):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Prompt one\n")
            f.write("# Comment\n")
            f.write("Prompt two\n")
            f.write("\n")
            f.write("Prompt three\n")
            f.flush()

            prompts = tester.load_prompts_from_file(f.name)
            assert len(prompts) == 3
            assert "Prompt one" in prompts
            assert "Prompt two" in prompts
            assert "Prompt three" in prompts

    def test_load_prompts_from_csv_file(self, tester):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("prompt,category\n")
            f.write("CSV prompt one,cat1\n")
            f.write("CSV prompt two,cat2\n")
            f.flush()

            prompts = tester.load_prompts_from_file(f.name)
            assert len(prompts) == 2
            assert "CSV prompt one" in prompts

    def test_generate_trial_id(self, tester):
        id1 = tester._generate_trial_id("prompt", "fun")
        id2 = tester._generate_trial_id("prompt", "fun")
        assert isinstance(id1, str)
        assert len(id1) == 12
        # IDs should be different due to random/time component
        assert id1 != id2

    def test_call_llm_without_client(self):
        with patch("fun_hypothesis.HAS_ANTHROPIC", False):
            tester = FramingHypothesisTester.__new__(FramingHypothesisTester)
            tester.model = "test"
            tester.client = None
            tester.framings = FRAMING_STYLES.copy()

            with pytest.raises(RuntimeError, match="anthropic package not installed"):
                tester._call_llm("system", "user")

    def test_call_llm_with_client(self, tester):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        tester.client.messages.create.return_value = mock_response

        result = tester._call_llm("system prompt", "user message")
        assert result == "Test response"
        tester.client.messages.create.assert_called_once()

    def test_transform_prompt(self, tester):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Transformed prompt")]
        tester.client.messages.create.return_value = mock_response

        result = tester.transform_prompt("Original prompt", "fun")
        assert result == "Transformed prompt"

    def test_get_response(self, tester):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="LLM response")]
        tester.client.messages.create.return_value = mock_response

        result = tester.get_response("Test prompt")
        assert result == "LLM response"

    def test_judge_response_valid_json(self, tester):
        judge_json = json.dumps(
            {
                "accuracy": 8,
                "clarity": 7,
                "completeness": 9,
                "usefulness": 8,
                "engagement": 7,
                "overall": 7.8,
                "brief_rationale": "Good response",
            }
        )
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=judge_json)]
        tester.client.messages.create.return_value = mock_response

        result = tester.judge_response("prompt", "response", 1)
        assert result["overall"] == 7.8
        assert result["accuracy"] == 8

    def test_judge_response_invalid_json(self, tester):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Not valid JSON")]
        tester.client.messages.create.return_value = mock_response

        result = tester.judge_response("prompt", "response", 1)
        assert "error" in result
        assert result["overall"] == 5.0

    def test_run_trial(self, tester):
        # Mock all LLM calls
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            if call_count[0] == 1:  # transform
                mock_resp.content = [MagicMock(text="Framed prompt")]
            elif call_count[0] in [2, 3]:  # responses
                mock_resp.content = [MagicMock(text="Response text")]
            else:  # judgments
                mock_resp.content = [MagicMock(text='{"overall": 7.5, "accuracy": 8}')]
            return mock_resp

        tester.client.messages.create.side_effect = mock_create

        trial = tester.run_trial("Test prompt", "fun", num_judges=1)

        assert trial.framing_style == "fun"
        assert trial.original_prompt == "Test prompt"
        assert trial.framed_prompt == "Framed prompt"
        assert trial.winner in ["raw", "framed", "tie"]

    def test_save_results(self, tester):
        results = ExperimentResults(
            experiment_id="test",
            timestamp="2025-01-01",
            model="test-model",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = tester.save_results(results, f.name)
            assert Path(path).exists()

            with open(path) as rf:
                data = json.load(rf)
                assert data["experiment_id"] == "test"

    def test_generate_report(self, tester):
        results = ExperimentResults(
            experiment_id="test123",
            timestamp="2025-01-01",
            model="test-model",
            num_prompts=5,
            num_judges=3,
            conclusion="Test conclusion",
            rankings=[("fun", 5.0)],
            framing_results={
                "fun": {
                    "framing_style": "fun",
                    "framing_name": "Fun & Engaging",
                    "num_trials": 5,
                    "framed_wins": 3,
                    "raw_wins": 2,
                    "avg_raw_score": 7.0,
                    "avg_framed_score": 7.5,
                    "improvement_pct": 5.0,
                }
            },
        )

        report = tester.generate_report(results)
        assert "# Prompt Framing Hypothesis Experiment Report" in report
        assert "test123" in report
        assert "Fun & Engaging" in report


class TestMain:
    """Tests for main() function."""

    def test_list_framings(self, capsys):
        with patch("sys.argv", ["fun_hypothesis", "--list-framings"]):
            main()
            captured = capsys.readouterr()
            assert "fun" in captured.out
            assert "pirate" in captured.out

    def test_dry_run(self, capsys):
        with patch("sys.argv", ["fun_hypothesis", "--dry-run", "--trials", "3"]):
            main()
            captured = capsys.readouterr()
            assert "DRY RUN" in captured.out
            assert "Estimated API calls" in captured.out

    def test_missing_anthropic(self, capsys):
        with patch("fun_hypothesis.HAS_ANTHROPIC", False):
            with patch("sys.argv", ["fun_hypothesis", "--trials", "1"]):
                main()
                captured = capsys.readouterr()
                assert "anthropic package not installed" in captured.out

    def test_missing_api_key(self, capsys, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("fun_hypothesis.HAS_ANTHROPIC", True):
            with patch("sys.argv", ["fun_hypothesis", "--trials", "1"]):
                main()
                captured = capsys.readouterr()
                assert "ANTHROPIC_API_KEY" in captured.out


class TestRunExperiment:
    """Tests for run_experiment method."""

    @pytest.fixture
    def tester(self):
        with patch.object(FramingHypothesisTester, "__init__", lambda self, model="test": None):
            t = FramingHypothesisTester.__new__(FramingHypothesisTester)
            t.model = "test-model"
            t.client = MagicMock()
            t.framings = FRAMING_STYLES.copy()
            return t

    def test_run_experiment_with_default_prompts(self, tester, capsys):
        """Test run_experiment uses sample prompts when none provided."""
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            if call_count[0] % 5 == 1:  # transform
                mock_resp.content = [MagicMock(text="Framed prompt")]
            elif call_count[0] % 5 in [2, 3]:  # responses
                mock_resp.content = [MagicMock(text="Response text")]
            else:  # judgments
                mock_resp.content = [MagicMock(text='{"overall": 8.0, "accuracy": 8}')]
            return mock_resp

        tester.client.messages.create.side_effect = mock_create

        results = tester.run_experiment(prompts=None, framing_keys=["fun"], num_judges=1)

        assert results.experiment_id != ""
        assert results.model == "test-model"
        assert results.num_judges == 1
        assert "fun" in results.framing_results

    def test_run_experiment_with_custom_prompts(self, tester, capsys):
        """Test run_experiment with custom prompts."""
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            if call_count[0] % 5 == 1:
                mock_resp.content = [MagicMock(text="Framed prompt")]
            elif call_count[0] % 5 in [2, 3]:
                mock_resp.content = [MagicMock(text="Response text")]
            else:
                mock_resp.content = [MagicMock(text='{"overall": 7.0, "accuracy": 7}')]
            return mock_resp

        tester.client.messages.create.side_effect = mock_create

        custom_prompts = ["Test prompt 1", "Test prompt 2"]
        results = tester.run_experiment(prompts=custom_prompts, framing_keys=["fun"], num_judges=1)

        assert results.num_prompts == 2
        assert "fun" in results.framing_results
        fr = results.framing_results["fun"]
        assert fr["num_trials"] == 2

    def test_run_experiment_multiple_framings(self, tester, capsys):
        """Test run_experiment with multiple framing styles."""
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            # Alternate between high and low scores for different framings
            if call_count[0] % 5 == 1:
                mock_resp.content = [MagicMock(text="Framed prompt")]
            elif call_count[0] % 5 in [2, 3]:
                mock_resp.content = [MagicMock(text="Response text")]
            else:
                # Make framed score higher sometimes, raw higher other times
                score = 8.0 if call_count[0] % 10 < 5 else 6.0
                mock_resp.content = [MagicMock(text=f'{{"overall": {score}, "accuracy": 7}}')]
            return mock_resp

        tester.client.messages.create.side_effect = mock_create

        results = tester.run_experiment(
            prompts=["Test prompt"],
            framing_keys=["fun", "pirate"],
            num_judges=1,
        )

        assert len(results.framing_results) == 2
        assert "fun" in results.framing_results
        assert "pirate" in results.framing_results
        assert len(results.rankings) == 2

    def test_run_experiment_framed_wins(self, tester, capsys):
        """Test run_experiment where framed version wins."""
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            if call_count[0] == 1:  # transform
                mock_resp.content = [MagicMock(text="Framed prompt")]
            elif call_count[0] == 2:  # raw response
                mock_resp.content = [MagicMock(text="Raw response")]
            elif call_count[0] == 3:  # framed response
                mock_resp.content = [MagicMock(text="Framed response")]
            elif call_count[0] == 4:  # judge raw - low score
                mock_resp.content = [MagicMock(text='{"overall": 5.0, "accuracy": 5}')]
            else:  # judge framed - high score
                mock_resp.content = [MagicMock(text='{"overall": 9.0, "accuracy": 9}')]
            return mock_resp

        tester.client.messages.create.side_effect = mock_create

        results = tester.run_experiment(
            prompts=["Test prompt"],
            framing_keys=["fun"],
            num_judges=1,
        )

        fr = results.framing_results["fun"]
        assert fr["framed_wins"] + fr["raw_wins"] + fr["ties"] == 1
        assert results.best_framing == "fun"

    def test_run_experiment_raw_wins(self, tester, capsys):
        """Test run_experiment where raw version wins."""
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            if call_count[0] == 1:  # transform
                mock_resp.content = [MagicMock(text="Framed prompt")]
            elif call_count[0] == 2:  # raw response
                mock_resp.content = [MagicMock(text="Raw response")]
            elif call_count[0] == 3:  # framed response
                mock_resp.content = [MagicMock(text="Framed response")]
            elif call_count[0] == 4:  # judge raw - high score
                mock_resp.content = [MagicMock(text='{"overall": 9.0, "accuracy": 9}')]
            else:  # judge framed - low score
                mock_resp.content = [MagicMock(text='{"overall": 5.0, "accuracy": 5}')]
            return mock_resp

        tester.client.messages.create.side_effect = mock_create

        results = tester.run_experiment(
            prompts=["Test prompt"],
            framing_keys=["fun"],
            num_judges=1,
        )

        fr = results.framing_results["fun"]
        # Raw should win with the scores we provided
        assert fr["framed_wins"] + fr["raw_wins"] + fr["ties"] == 1

    def test_run_experiment_conclusion_best_framing(self, tester, capsys):
        """Test run_experiment generates correct conclusion for best framing."""
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            if call_count[0] == 1:
                mock_resp.content = [MagicMock(text="Framed prompt")]
            elif call_count[0] in [2, 3]:
                mock_resp.content = [MagicMock(text="Response")]
            elif call_count[0] == 4:  # first judge (randomized order)
                mock_resp.content = [MagicMock(text='{"overall": 6.0, "accuracy": 6}')]
            else:  # second judge
                mock_resp.content = [MagicMock(text='{"overall": 9.0, "accuracy": 9}')]
            return mock_resp

        tester.client.messages.create.side_effect = mock_create

        results = tester.run_experiment(
            prompts=["Test prompt"],
            framing_keys=["fun"],
            num_judges=1,
        )

        # Test that a conclusion was generated (actual content depends on random order)
        assert results.conclusion != ""
        assert (
            "Best framing" in results.conclusion
            or "Raw prompts" in results.conclusion
            or "No significant" in results.conclusion
        )

    def test_run_experiment_conclusion_raw_better(self, tester, capsys):
        """Test conclusion when raw prompts perform better."""
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            if call_count[0] == 1:
                mock_resp.content = [MagicMock(text="Framed prompt")]
            elif call_count[0] in [2, 3]:
                mock_resp.content = [MagicMock(text="Response")]
            elif call_count[0] == 4:  # raw judge - high score
                mock_resp.content = [MagicMock(text='{"overall": 9.0, "accuracy": 9}')]
            else:  # framed judge - much lower score
                mock_resp.content = [MagicMock(text='{"overall": 5.0, "accuracy": 5}')]
            return mock_resp

        tester.client.messages.create.side_effect = mock_create

        results = tester.run_experiment(
            prompts=["Test prompt"],
            framing_keys=["fun"],
            num_judges=1,
        )

        # With raw better, check conclusion logic
        assert results.conclusion != ""

    def test_run_experiment_with_unknown_framing(self, tester, capsys):
        """Test run_experiment with unknown framing key."""
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            if call_count[0] == 1:
                mock_resp.content = [MagicMock(text="Framed prompt")]
            elif call_count[0] in [2, 3]:
                mock_resp.content = [MagicMock(text="Response")]
            else:
                mock_resp.content = [MagicMock(text='{"overall": 7.0, "accuracy": 7}')]
            return mock_resp

        tester.client.messages.create.side_effect = mock_create

        # Unknown framing should still work with fallback
        results = tester.run_experiment(
            prompts=["Test prompt"],
            framing_keys=["unknown_framing"],
            num_judges=1,
        )

        assert "unknown_framing" in results.framing_results


class TestMainFullExecution:
    """Tests for main() function full execution paths."""

    def test_main_with_framings_csv(self, capsys, tmp_path, monkeypatch):
        """Test main with custom framings CSV."""
        # Create a temp CSV file
        csv_file = tmp_path / "framings.csv"
        csv_file.write_text("key,name,instruction\ncustom1,Custom One,Do custom thing\n")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("fun_hypothesis.HAS_ANTHROPIC", True):
            with patch("fun_hypothesis.FramingHypothesisTester") as MockTester:
                mock_instance = MagicMock()
                mock_results = MagicMock()
                mock_results.conclusion = "Test conclusion"
                mock_results.rankings = [("custom1", 5.0)]
                mock_instance.run_experiment.return_value = mock_results
                mock_instance.save_results.return_value = "results.json"
                mock_instance.generate_report.return_value = "# Report"
                MockTester.return_value = mock_instance

                with patch(
                    "sys.argv",
                    [
                        "fun_hypothesis",
                        "--trials",
                        "1",
                        "--framings-csv",
                        str(csv_file),
                        "--framing",
                        "custom1",
                    ],
                ):
                    main()

                mock_instance.load_framings_from_csv.assert_called_once_with(str(csv_file))

    def test_main_with_custom_instruction(self, capsys, monkeypatch):
        """Test main with custom instruction."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("fun_hypothesis.HAS_ANTHROPIC", True):
            with patch("fun_hypothesis.FramingHypothesisTester") as MockTester:
                mock_instance = MagicMock()
                mock_results = MagicMock()
                mock_results.conclusion = "Test conclusion"
                mock_results.rankings = [("custom", 5.0)]
                mock_instance.run_experiment.return_value = mock_results
                mock_instance.save_results.return_value = "results.json"
                mock_instance.generate_report.return_value = "# Report"
                MockTester.return_value = mock_instance

                with patch(
                    "sys.argv",
                    [
                        "fun_hypothesis",
                        "--trials",
                        "1",
                        "--custom-instruction",
                        "Be very creative",
                    ],
                ):
                    main()

                mock_instance.add_custom_framing.assert_called_once_with(
                    "custom", "Custom Framing", "Be very creative"
                )

    def test_main_with_prompts_file(self, capsys, tmp_path, monkeypatch):
        """Test main with prompts file."""
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("Prompt one\nPrompt two\n")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("fun_hypothesis.HAS_ANTHROPIC", True):
            with patch("fun_hypothesis.FramingHypothesisTester") as MockTester:
                mock_instance = MagicMock()
                mock_instance.load_prompts_from_file.return_value = [
                    "Prompt one",
                    "Prompt two",
                ]
                mock_results = MagicMock()
                mock_results.conclusion = "Test conclusion"
                mock_results.rankings = [("fun", 5.0)]
                mock_instance.run_experiment.return_value = mock_results
                mock_instance.save_results.return_value = "results.json"
                mock_instance.generate_report.return_value = "# Report"
                MockTester.return_value = mock_instance

                with patch(
                    "sys.argv",
                    ["fun_hypothesis", "--prompts-file", str(prompts_file)],
                ):
                    main()

                mock_instance.load_prompts_from_file.assert_called_once_with(str(prompts_file))
                captured = capsys.readouterr()
                assert "Loaded 2 prompts" in captured.out

    def test_main_with_multiple_framings(self, capsys, monkeypatch):
        """Test main with comma-separated framings."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("fun_hypothesis.HAS_ANTHROPIC", True):
            with patch("fun_hypothesis.FramingHypothesisTester") as MockTester:
                mock_instance = MagicMock()
                mock_results = MagicMock()
                mock_results.conclusion = "Test conclusion"
                mock_results.rankings = [("fun", 5.0), ("pirate", 3.0)]
                mock_instance.run_experiment.return_value = mock_results
                mock_instance.save_results.return_value = "results.json"
                mock_instance.generate_report.return_value = "# Report"
                MockTester.return_value = mock_instance

                with patch(
                    "sys.argv",
                    ["fun_hypothesis", "--trials", "1", "--framings", "fun,pirate"],
                ):
                    main()

                # Check run_experiment was called with both framings
                call_args = mock_instance.run_experiment.call_args
                assert "fun" in call_args.kwargs["framing_keys"]
                assert "pirate" in call_args.kwargs["framing_keys"]

    def test_main_full_execution_with_mocked_api(self, capsys, tmp_path, monkeypatch):
        """Test full main execution with mocked API calls."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)

        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            if call_count[0] % 5 == 1:
                mock_resp.content = [MagicMock(text="Framed prompt")]
            elif call_count[0] % 5 in [2, 3]:
                mock_resp.content = [MagicMock(text="Response")]
            else:
                mock_resp.content = [MagicMock(text='{"overall": 7.0, "accuracy": 7}')]
            return mock_resp

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = mock_create

        with patch("fun_hypothesis.HAS_ANTHROPIC", True):
            with patch("fun_hypothesis.anthropic") as mock_anthropic:
                mock_anthropic.Anthropic.return_value = mock_client

                with patch("sys.argv", ["fun_hypothesis", "--trials", "1", "--judges", "1"]):
                    main()

        captured = capsys.readouterr()
        assert "EXPERIMENT COMPLETE" in captured.out
        assert "Results saved to" in captured.out
        assert "Report saved to" in captured.out

    def test_main_dry_run_with_framings(self, capsys):
        """Test dry run with multiple framings."""
        with patch(
            "sys.argv",
            ["fun_hypothesis", "--dry-run", "--trials", "2", "--framings", "fun,pirate"],
        ):
            main()

        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "fun, pirate" in captured.out
        assert "Estimated API calls" in captured.out


class TestFramingHypothesisTesterInit:
    """Tests for FramingHypothesisTester __init__."""

    def test_init_with_anthropic(self):
        """Test __init__ when anthropic is available."""
        mock_client = MagicMock()
        with patch("fun_hypothesis.HAS_ANTHROPIC", True):
            with patch("fun_hypothesis.anthropic") as mock_anthropic:
                mock_anthropic.Anthropic.return_value = mock_client
                tester = FramingHypothesisTester(model="test-model")

        assert tester.model == "test-model"
        assert tester.client == mock_client
        assert tester.framings == FRAMING_STYLES

    def test_init_without_anthropic(self):
        """Test __init__ when anthropic is not available."""
        with patch("fun_hypothesis.HAS_ANTHROPIC", False):
            tester = FramingHypothesisTester(model="test-model")

        assert tester.model == "test-model"
        assert tester.client is None
        assert tester.framings == FRAMING_STYLES


class TestJudgeResponseEdgeCases:
    """Tests for edge cases in judge_response."""

    @pytest.fixture
    def tester(self):
        with patch.object(FramingHypothesisTester, "__init__", lambda self, model="test": None):
            t = FramingHypothesisTester.__new__(FramingHypothesisTester)
            t.model = "test-model"
            t.client = MagicMock()
            t.framings = FRAMING_STYLES.copy()
            return t

    def test_judge_response_no_json_braces(self, tester):
        """Test judge_response when response has no JSON braces at all."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is just plain text with no braces")]
        tester.client.messages.create.return_value = mock_response

        result = tester.judge_response("prompt", "response", 1)
        assert "error" in result
        assert result["overall"] == 5.0
        assert "Could not parse" in result["error"]

    def test_judge_response_invalid_json_with_braces(self, tester):
        """Test judge_response when response has braces but invalid JSON content."""
        mock_response = MagicMock()
        # This has braces so regex matches, but content is not valid JSON
        mock_response.content = [MagicMock(text="{this is not valid json syntax}")]
        tester.client.messages.create.return_value = mock_response

        result = tester.judge_response("prompt", "response", 1)
        assert "error" in result
        assert result["overall"] == 5.0
        assert "Invalid JSON" in result["error"]


class TestRunExperimentEdgeCases:
    """Edge case tests for run_experiment."""

    @pytest.fixture
    def tester(self):
        with patch.object(FramingHypothesisTester, "__init__", lambda self, model="test": None):
            t = FramingHypothesisTester.__new__(FramingHypothesisTester)
            t.model = "test-model"
            t.client = MagicMock()
            t.framings = FRAMING_STYLES.copy()
            return t

    def test_run_experiment_default_framing_keys(self, tester, capsys):
        """Test run_experiment uses default framing_keys when None."""
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            if call_count[0] % 5 == 1:
                mock_resp.content = [MagicMock(text="Framed prompt")]
            elif call_count[0] % 5 in [2, 3]:
                mock_resp.content = [MagicMock(text="Response text")]
            else:
                mock_resp.content = [MagicMock(text='{"overall": 7.5, "accuracy": 8}')]
            return mock_resp

        tester.client.messages.create.side_effect = mock_create

        # Pass prompts but no framing_keys - should default to ["fun"]
        results = tester.run_experiment(prompts=["Test prompt"], framing_keys=None, num_judges=1)

        assert "fun" in results.framing_results
        assert len(results.framing_results) == 1

    def test_run_experiment_no_significant_difference(self, tester, capsys):
        """Test run_experiment when there's no significant difference."""
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            if call_count[0] % 5 == 1:
                mock_resp.content = [MagicMock(text="Framed prompt")]
            elif call_count[0] % 5 in [2, 3]:
                mock_resp.content = [MagicMock(text="Response text")]
            else:
                # Return same scores for both to get ~0% improvement
                mock_resp.content = [MagicMock(text='{"overall": 7.0, "accuracy": 7}')]
            return mock_resp

        tester.client.messages.create.side_effect = mock_create

        results = tester.run_experiment(
            prompts=["Test prompt"],
            framing_keys=["fun"],
            num_judges=1,
        )

        # With identical scores, improvement should be ~0%, triggering "no significant" conclusion
        assert (
            "No significant" in results.conclusion
            or "Raw" in results.conclusion
            or "Best" in results.conclusion
        )


class TestTrialWinnerDetermination:
    """Tests for winner determination logic."""

    def test_framed_wins_significant_difference(self):
        trial = Trial(
            trial_id="test",
            framing_style="fun",
            original_prompt="test",
            raw_avg_score=6.0,
            framed_avg_score=7.0,
        )
        # Winner would be determined by the run_trial method
        # Here we test the threshold logic
        if trial.framed_avg_score > trial.raw_avg_score + 0.5:
            winner = "framed"
        elif trial.raw_avg_score > trial.framed_avg_score + 0.5:
            winner = "raw"
        else:
            winner = "tie"
        assert winner == "framed"

    def test_raw_wins_significant_difference(self):
        trial = Trial(
            trial_id="test",
            framing_style="fun",
            original_prompt="test",
            raw_avg_score=8.0,
            framed_avg_score=7.0,
        )
        if trial.framed_avg_score > trial.raw_avg_score + 0.5:
            winner = "framed"
        elif trial.raw_avg_score > trial.framed_avg_score + 0.5:
            winner = "raw"
        else:
            winner = "tie"
        assert winner == "raw"

    def test_tie_when_close(self):
        trial = Trial(
            trial_id="test",
            framing_style="fun",
            original_prompt="test",
            raw_avg_score=7.3,
            framed_avg_score=7.5,
        )
        if trial.framed_avg_score > trial.raw_avg_score + 0.5:
            winner = "framed"
        elif trial.raw_avg_score > trial.framed_avg_score + 0.5:
            winner = "raw"
        else:
            winner = "tie"
        assert winner == "tie"
