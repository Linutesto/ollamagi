import unittest
import time
import shutil
import uuid
import json
from unittest.mock import patch

from core.agents import normalize_role_name
from core.orchestrator import (
    Flow,
    Subtask,
    Task,
    _execution_failed,
    _execute_subtask,
    _agent_capability_errors,
    _external_python_imports,
    _generate_subtasks,
    _python_syntax_error,
    _infer_deliverable_kind,
    _normalize_expected_artifacts,
    _objective_constraints,
    _undeclared_imports,
    _object_list,
    _reconcile_task_status,
    _recovered_after_replan,
    _role_for_flow,
    _supersede_failed_attempts,
    _validate_execution,
    _validate_flow_deliverables,
    _validate_text_result,
    _workspace_snapshot,
)
from core.config import MODELS, SINGLE_MODEL, WORKSPACE_DIR
from core.model_router import _CALL_PROFILES
from executor.docker_manager import _exec_with_timeout, _strict_bash


class RoleNormalizationTests(unittest.TestCase):
    def test_generated_role_aliases_are_normalized(self):
        expected = {
            "research_agent": "researcher",
            "planning_agent": "generator",
            "ResearchPlanner": "researcher",
            "WebCrawler": "researcher",
            "ContentExtractor": "researcher",
            "MemoryWriter": "coder",
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_role_name(source), target)

    def test_role_is_constrained_to_flow(self):
        self.assertEqual(_role_for_flow("MemoryWriter", "research"), "coder")
        self.assertEqual(_role_for_flow("pentester", "research"), "primary_agent")


class PlannerValidationTests(unittest.TestCase):
    def test_only_object_lists_are_accepted(self):
        self.assertEqual(_object_list({"id": 1}), [])
        self.assertEqual(_object_list([{"id": 1}, "bad", None]), [{"id": 1}])

    def test_all_agent_calls_use_single_model(self):
        for task_type in ("orchestrator", "coder", "tools", "analysis", "fast"):
            self.assertEqual(MODELS[task_type], SINGLE_MODEL)

    def test_fast_calls_are_bounded(self):
        self.assertLessEqual(_CALL_PROFILES["fast"]["num_predict"], 512)
        self.assertLessEqual(_CALL_PROFILES["fast"]["timeout"], 90)
        self.assertFalse(_CALL_PROFILES["fast"]["think"])

    def test_deliverable_fallback_prioritizes_actual_output_type(self):
        self.assertEqual(
            _infer_deliverable_kind(
                "Draft README.md",
                "Document the Python agent codebase and usage",
                True,
            ),
            "documentation",
        )
        self.assertEqual(
            _infer_deliverable_kind(
                "Validate README",
                "Review the existing agent documentation",
                True,
            ),
            "test",
        )

    def test_expected_artifacts_are_work_relative_and_safe(self):
        self.assertEqual(
            _normalize_expected_artifacts(
                ["/work/README.md", "src/*.py", "../secret", "/etc/passwd"]
            ),
            ["README.md", "src/*.py"],
        )

    def test_all_workflow_types_preserve_explicit_report_contracts(self):
        response = (
            '[{"id":1,"title":"Write report","description":"Create report.md",'
            '"agent":"researcher","needs_container":true,"container_type":"python",'
            '"deliverable_kind":"report","expected_artifacts":["report.md"]}]'
        )
        for flow_type in (
            "agent_development", "product_development", "research", "security", "general"
        ):
            flow = Flow(
                id=f"flow-{flow_type}",
                title="flow",
                objective="produce a report",
                flow_type=flow_type,
            )
            task = Task(
                id="t1",
                flow_id=flow.id,
                title="Report",
                description="Create report.md",
                agent="primary_agent",
            )
            with self.subTest(flow_type=flow_type), patch(
                "core.orchestrator.chat", return_value=response
            ):
                subtask = _generate_subtasks(task, flow, "")[0]
                self.assertEqual(subtask.deliverable_kind, "report")
                self.assertEqual(subtask.expected_artifacts, ["report.md"])
                self.assertTrue(subtask.needs_container)
                self.assertIn(subtask.agent, {"coder", "pentester"})

    def test_email_constraints_forbid_mandatory_external_services(self):
        constraints = _objective_constraints(
            "Build an email automation agent", "agent_development"
        )
        self.assertIn("local .eml fixtures", constraints)
        self.assertIn("Redis may be optional", constraints)
        self.assertIn("--self-test", constraints)

    def test_text_subtask_is_not_assigned_to_coder(self):
        response = (
            '[{"id":1,"title":"Analyze design","description":"Reason about tradeoffs",'
            '"agent":"coder","needs_container":false,"container_type":"python",'
            '"deliverable_kind":"text","expected_artifacts":[]}]'
        )
        flow = Flow(
            id="flow-general",
            title="flow",
            objective="analyze a system",
            flow_type="general",
        )
        task = Task(
            id="t1",
            flow_id=flow.id,
            title="Analysis",
            description="Reason about tradeoffs",
            agent="primary_agent",
        )
        with patch("core.orchestrator.chat", return_value=response):
            subtask = _generate_subtasks(task, flow, "")[0]
        self.assertFalse(subtask.needs_container)
        self.assertNotEqual(subtask.agent, "coder")


class ExecutionClassificationTests(unittest.TestCase):
    def test_zero_exit_with_fatal_output_is_failure(self):
        self.assertTrue(_execution_failed(0, "Error: required file not created"))
        self.assertTrue(_execution_failed(0, "python3: command not found"))
        self.assertTrue(_execution_failed(0, "Traceback (most recent call last):"))

    def test_handled_error_log_does_not_fail_successful_process(self):
        self.assertFalse(
            _execution_failed(
                0,
                "2026-06-19 INFO crawl started\n"
                "2026-06-19 ERROR handled 404 for fixture\n"
                "crawl completed successfully\n",
            )
        )

    def test_clean_zero_exit_is_success(self):
        self.assertFalse(_execution_failed(0, "Created /work/report.json\n"))

    def test_nonzero_exit_is_failure(self):
        self.assertTrue(_execution_failed(124, "Command timed out"))

    def test_bash_is_made_strict(self):
        script = _strict_bash("#!/usr/bin/env bash\necho ok")
        self.assertEqual(script.splitlines()[1], "set -euo pipefail")

    def test_docker_exec_timeout_kills_container(self):
        class SlowContainer:
            killed = False

            def exec_run(self, *_args, **_kwargs):
                time.sleep(2)

            def kill(self):
                self.killed = True

        container = SlowContainer()
        exit_code, output = _exec_with_timeout(container, ["sleep", "2"], timeout=1)
        self.assertEqual(exit_code, 124)
        self.assertTrue(container.killed)
        self.assertIn("timed out", output)


class ArtifactValidationTests(unittest.TestCase):
    def setUp(self):
        self.flow_id = f"test-{uuid.uuid4().hex[:8]}"
        self.root = WORKSPACE_DIR / self.flow_id
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_implementation_without_files_fails(self):
        subtask = Subtask(
            id="s1",
            task_id="t1",
            title="Implement core logic",
            description="Write the main Python implementation",
            agent="coder",
            needs_container=True,
        )
        valid, artifacts, report = _validate_execution(
            self.flow_id, subtask, _workspace_snapshot(self.flow_id), "(done)"
        )
        self.assertFalse(valid)
        self.assertEqual(artifacts, [])
        self.assertIn("no deliverable file", report)

    def test_valid_changed_python_is_evidence(self):
        subtask = Subtask(
            id="s1",
            task_id="t1",
            title="Implement core logic",
            description="Create main.py",
            agent="coder",
            needs_container=True,
        )
        before = _workspace_snapshot(self.flow_id)
        (self.root / "main.py").write_text("print('ok')\n")
        valid, artifacts, report = _validate_execution(
            self.flow_id, subtask, before, "created main.py"
        )
        self.assertTrue(valid)
        self.assertEqual(artifacts, ["main.py"])
        self.assertIn("/work/main.py", report)

    def test_script_task_rejects_non_source_artifacts(self):
        subtask = Subtask(
            id="s1",
            task_id="t1",
            title="Create Main Trading Bot Script",
            description="Implement the core trading bot logic",
            agent="coder",
            needs_container=True,
        )
        before = _workspace_snapshot(self.flow_id)
        (self.root / "config.yaml").write_text("symbol: BTCUSDT\n")
        (self.root / "trading_results.json").write_text("{}\n")
        valid, _, report = _validate_execution(
            self.flow_id, subtask, before, "completed"
        )
        self.assertFalse(valid)
        self.assertIn("no source file", report)

    def test_documentation_task_requires_document(self):
        subtask = Subtask(
            id="s1",
            task_id="t1",
            title="Generate README Documentation",
            description="Create setup documentation",
            agent="coder",
            needs_container=True,
        )
        before = _workspace_snapshot(self.flow_id)
        (self.root / "main.py").write_text("print('ok')\n")
        valid, _, report = _validate_execution(
            self.flow_id, subtask, before, "completed"
        )
        self.assertFalse(valid)
        self.assertIn("no README", report)

    def test_documentation_contract_is_not_reclassified_as_source(self):
        subtask = Subtask(
            id="s1",
            task_id="t1",
            title="Draft README.md",
            description="Document the Python agent codebase and script usage",
            agent="coder",
            needs_container=True,
            deliverable_kind="documentation",
            expected_artifacts=["README.md"],
        )
        before = _workspace_snapshot(self.flow_id)
        (self.root / "README.md").write_text("# Agent\n\nUsage instructions.\n")
        valid, artifacts, report = _validate_execution(
            self.flow_id, subtask, before, "README generated"
        )
        self.assertTrue(valid, report)
        self.assertEqual(artifacts, ["README.md"])

    def test_exact_config_python_path_overrides_generic_suffix_rule(self):
        subtask = Subtask(
            id="s1",
            task_id="t1",
            title="Define logging configuration",
            description="Create logging constants",
            agent="coder",
            needs_container=True,
            deliverable_kind="configuration",
            expected_artifacts=["config/logging_config.py"],
        )
        before = _workspace_snapshot(self.flow_id)
        target = self.root / "config" / "logging_config.py"
        target.parent.mkdir()
        target.write_text("LOG_LEVEL = 'INFO'\n")
        valid, artifacts, report = _validate_execution(
            self.flow_id, subtask, before, "configuration created"
        )
        self.assertTrue(valid, report)
        self.assertEqual(artifacts, ["config/logging_config.py"])

    def test_multi_file_contract_requires_every_expected_artifact(self):
        subtask = Subtask(
            id="s1",
            task_id="t1",
            title="Create configuration bundle",
            description="Create both configuration files",
            agent="coder",
            needs_container=True,
            deliverable_kind="configuration",
            expected_artifacts=["config/app.json", "config/errors.yaml"],
        )
        before = _workspace_snapshot(self.flow_id)
        target = self.root / "config" / "app.json"
        target.parent.mkdir()
        target.write_text("{}\n")
        valid, _, report = _validate_execution(
            self.flow_id, subtask, before, "one file created"
        )
        self.assertFalse(valid)
        self.assertIn("config/errors.yaml", report)

    def test_dependency_contract_accepts_requirements_without_source(self):
        subtask = Subtask(
            id="s1",
            task_id="t1",
            title="Generate requirements.txt",
            description="Define dependencies for the Python codebase",
            agent="coder",
            needs_container=True,
            deliverable_kind="dependency",
            expected_artifacts=["requirements.txt"],
        )
        before = _workspace_snapshot(self.flow_id)
        (self.root / "requirements.txt").write_text("requests>=2.31\n")
        valid, artifacts, report = _validate_execution(
            self.flow_id, subtask, before, "requirements generated"
        )
        self.assertTrue(valid, report)
        self.assertEqual(artifacts, ["requirements.txt"])

    def test_test_contract_can_pass_without_modifying_source(self):
        subtask = Subtask(
            id="s1",
            task_id="t1",
            title="Validate local execution",
            description="Run the existing project test suite",
            agent="coder",
            needs_container=True,
            deliverable_kind="test",
        )
        valid, artifacts, report = _validate_execution(
            self.flow_id,
            subtask,
            _workspace_snapshot(self.flow_id),
            "12 tests passed",
        )
        self.assertTrue(valid, report)
        self.assertEqual(artifacts, [])

    def test_python_bytecode_is_not_deliverable_evidence(self):
        subtask = Subtask(
            id="s1",
            task_id="t1",
            title="Validate imports",
            description="Run import validation",
            agent="coder",
            needs_container=True,
            deliverable_kind="test",
        )
        before = _workspace_snapshot(self.flow_id)
        cache = self.root / "__pycache__"
        cache.mkdir()
        (cache / "bot.cpython-311.pyc").write_bytes(b"bytecode")
        valid, artifacts, report = _validate_execution(
            self.flow_id, subtask, before, "imports passed"
        )
        self.assertTrue(valid, report)
        self.assertEqual(artifacts, [])

    def test_failed_historical_contract_is_not_a_final_requirement(self):
        failed = Subtask(
            id="s1",
            task_id="t1",
            title="Abandoned module layout",
            description="Create src/old_layout.py",
            agent="coder",
            status="failed",
            needs_container=True,
            deliverable_kind="source",
            expected_artifacts=["src/old_layout.py"],
        )
        task = Task(
            id="t1",
            flow_id=self.flow_id,
            title="Old plan",
            description="",
            agent="coder",
            status="failed",
            subtasks=[failed],
        )
        flow = Flow(
            id=self.flow_id,
            title="general",
            objective="Produce a result",
            flow_type="general",
            tasks=[task],
        )
        valid, report = _validate_flow_deliverables(flow)
        self.assertTrue(valid, report)

    def test_python_build_script_syntax_is_checked_before_execution(self):
        error = _python_syntax_error(
            'content = """\\nclass Agent:\\n    """nested docstring"""\\n"""\\n'
        )
        self.assertIsNotNone(error)
        self.assertIn("syntax error", error.lower())

    def test_documentation_is_written_directly_without_python_builder(self):
        flow = Flow(
            id=self.flow_id,
            title="docs",
            objective="Document the agent",
            flow_type="general",
        )
        task = Task(
            id="t1",
            flow_id=self.flow_id,
            title="Documentation",
            description="Create README",
            agent="coder",
        )
        subtask = Subtask(
            id="s1",
            task_id=task.id,
            title="Generate README",
            description="Create complete usage documentation",
            agent="coder",
            needs_container=True,
            deliverable_kind="documentation",
            expected_artifacts=["README.md"],
        )
        with patch("core.orchestrator.context_for_task", return_value=""), patch(
            "core.orchestrator.chat", return_value="# Agent\n\nRun with `python app.py`."
        ):
            result = _execute_subtask(
                subtask, flow, task, [], lambda *_args: None, self.flow_id
            )
        self.assertFalse(result.startswith("[FAILED"))
        self.assertEqual(
            (self.root / "README.md").read_text(),
            "# Agent\n\nRun with `python app.py`.\n",
        )

    def test_configuration_bundle_is_written_without_runtime_dependencies(self):
        flow = Flow(
            id=self.flow_id,
            title="config",
            objective="Build an email agent",
            flow_type="agent_development",
        )
        task = Task(
            id="t1",
            flow_id=self.flow_id,
            title="Configuration",
            description="Create configuration",
            agent="coder",
        )
        subtask = Subtask(
            id="s1",
            task_id=task.id,
            title="Create configuration",
            description="Create JSON and YAML configuration",
            agent="coder",
            needs_container=True,
            deliverable_kind="configuration",
            expected_artifacts=["config/schema.json", "config/errors.yaml"],
        )
        response = json.dumps({
            "config/schema.json": '{"queue": "email"}',
            "config/errors.yaml": "retry_count: 3",
        })
        with patch("core.orchestrator.context_for_task", return_value=""), patch(
            "core.orchestrator.chat", return_value=response
        ):
            result = _execute_subtask(
                subtask, flow, task, [], lambda *_args: None, self.flow_id
            )
        self.assertFalse(result.startswith("[FAILED"), result)
        self.assertTrue((self.root / "config/schema.json").exists())
        self.assertTrue((self.root / "config/errors.yaml").exists())

    def test_invalid_changed_json_fails(self):
        subtask = Subtask(
            id="s1",
            task_id="t1",
            title="Generate configuration",
            description="Create config.json",
            agent="coder",
            needs_container=True,
        )
        before = _workspace_snapshot(self.flow_id)
        (self.root / "config.json").write_text("{broken")
        valid, _, report = _validate_execution(
            self.flow_id, subtask, before, "created config"
        )
        self.assertFalse(valid)
        self.assertIn("config.json", report)

    def test_agent_flow_requires_code_and_readme(self):
        flow = Flow(
            id=self.flow_id,
            title="agent",
            objective="Build an agent",
            flow_type="agent_development",
        )
        valid, report = _validate_flow_deliverables(flow)
        self.assertFalse(valid)
        self.assertIn("no source code", report)
        self.assertIn("no README", report)

    def test_placeholder_python_project_is_rejected(self):
        (self.root / "trading_bot.py").write_text(
            "#!/usr/bin/env python3\nprint('Trading Bot Initialized')\n"
        )
        (self.root / "README.md").write_text("# Trading Bot\n" + "usage\n" * 100)
        flow = Flow(
            id=self.flow_id,
            title="agent",
            objective="Build a trading bot",
            flow_type="agent_development",
        )
        valid, report = _validate_flow_deliverables(flow)
        self.assertFalse(valid)
        self.assertIn("placeholder code", report)

    def test_trading_project_requires_paper_mode(self):
        (self.root / "trading_bot.py").write_text(
            "class TradingBot:\n"
            "    def run(self):\n"
            "        return 'hold'\n\n"
            "def main():\n"
            "    return TradingBot().run()\n\n"
            "if __name__ == '__main__':\n"
            "    print(main())\n"
            + "# implementation\n" * 50
        )
        (self.root / "README.md").write_text("# Trading Bot\n" + "usage\n" * 100)
        flow = Flow(
            id=self.flow_id,
            title="agent",
            objective="Build a crypto trading bot",
            flow_type="agent_development",
        )
        valid, report = _validate_flow_deliverables(flow)
        self.assertFalse(valid)
        self.assertIn("paper/dry-run", report)

    def test_external_imports_require_persistent_manifest(self):
        app = self.root / "app.py"
        app.write_text(
            "import os\n"
            "from telegram import Update\n\n"
            "class App:\n"
            "    def run(self): return Update\n\n"
            "def main(): return App().run()\n"
            + "# implementation\n" * 50
        )
        (self.root / "README.md").write_text("# App\n" + "usage\n" * 100)
        imports = _external_python_imports([app], self.root)
        self.assertEqual(imports, {"telegram"})
        flow = Flow(
            id=self.flow_id,
            title="agent",
            objective="Build an API integration agent",
            flow_type="agent_development",
        )
        with patch(
            "core.orchestrator._validate_python_project_runtime",
            return_value=(True, "ok"),
        ):
            valid, report = _validate_flow_deliverables(flow)
        self.assertFalse(valid)
        self.assertIn("no dependency manifest", report)
        self.assertIn("telegram", report)

    def test_import_distribution_aliases_are_checked(self):
        manifest = self.root / "requirements.txt"
        manifest.write_text("telegram\n")
        self.assertEqual(
            _undeclared_imports({"telegram"}, [manifest]),
            {"telegram"},
        )
        manifest.write_text("python-telegram-bot>=21,<22\n")
        self.assertEqual(_undeclared_imports({"telegram"}, [manifest]), set())

    def test_email_capability_requires_real_email_logic_and_offline_fixture(self):
        errors = _agent_capability_errors(
            "Build an email automation agent",
            "def main(): pass\n# --self-test\n",
        )
        self.assertTrue(any("transport/message" in error for error in errors))
        self.assertTrue(any("fixture/fake" in error for error in errors))
        self.assertEqual(
            _agent_capability_errors(
                "Build an email automation agent",
                "import smtplib\nfrom email.message import EmailMessage\n"
                "# --self-test uses a fake .eml fixture\n",
            ),
            [],
        )

    def test_non_agent_workflows_require_durable_outputs(self):
        for flow_type, expected in (
            ("research", "no durable report"),
            ("product_development", "no durable plan"),
            ("security", "no durable assessment"),
        ):
            flow = Flow(
                id=self.flow_id,
                title=flow_type,
                objective="complete workflow",
                flow_type=flow_type,
            )
            with self.subTest(flow_type=flow_type):
                valid, report = _validate_flow_deliverables(flow)
                self.assertFalse(valid)
                self.assertIn(expected, report)

    def test_telegram_project_requires_offline_mode_and_no_token_literal(self):
        (self.root / "bot.py").write_text(
            "from telegram import Update\n\n"
            "TOKEN = '123456789:ABCdefGHIjklMNOpqrsTUVwxyz'\n\n"
            "class Bot:\n"
            "    def run(self): return Update\n\n"
            "def main(): return Bot().run()\n"
            + "# implementation\n" * 50
        )
        (self.root / "README.md").write_text("# Telegram Bot\n" + "usage\n" * 100)
        (self.root / "requirements.txt").write_text("python-telegram-bot>=21,<22\n")
        flow = Flow(
            id=self.flow_id,
            title="telegram",
            objective="Build a Telegram bot",
            flow_type="agent_development",
        )
        with patch(
            "core.orchestrator._validate_python_project_runtime",
            return_value=(True, "ok"),
        ):
            valid, report = _validate_flow_deliverables(flow)
        self.assertFalse(valid)
        self.assertIn("offline --self-test", report)
        self.assertIn("hardcoded token-like", report)

    def test_crawler_project_requires_local_self_test(self):
        (self.root / "scraper.py").write_text(
            "import requests\n\n"
            "class Scraper:\n"
            "    def crawl(self, url):\n"
            "        return requests.get(url).text\n\n"
            "def main():\n"
            "    return Scraper().crawl('https://httpbin.org/get')\n"
            + "# implementation\n" * 50
        )
        (self.root / "README.md").write_text("# Scraper\n" + "usage\n" * 100)
        flow = Flow(
            id=self.flow_id,
            title="crawler",
            objective="Build a web scraper/crawler",
            flow_type="agent_development",
        )
        valid, report = _validate_flow_deliverables(flow)
        self.assertFalse(valid)
        self.assertIn("local fixture/self-test", report)

    def test_successful_replan_can_supersede_earlier_failures(self):
        failed = Task(
            id="t1",
            flow_id=self.flow_id,
            title="failed approach",
            description="",
            agent="coder",
            status="failed",
        )
        recovered_subtask = Subtask(
            id="r2-s1",
            task_id="r2",
            title="recovery",
            description="",
            agent="coder",
            status="finished",
            artifacts=["main.py"],
        )
        recovered = Task(
            id="r2",
            flow_id=self.flow_id,
            title="recovered",
            description="",
            agent="coder",
            status="finished",
            subtasks=[recovered_subtask],
        )
        flow = Flow(
            id=self.flow_id,
            title="agent",
            objective="Build an agent",
            flow_type="agent_development",
            tasks=[failed, recovered],
            replan_count=1,
        )
        self.assertTrue(_recovered_after_replan(flow))

    def test_validated_replacement_marks_failures_as_superseded(self):
        subtask = Subtask(
            id="s1",
            task_id="t1",
            title="old attempt",
            description="",
            agent="coder",
            status="failed",
            result="failed output",
        )
        task = Task(
            id="t1",
            flow_id=self.flow_id,
            title="old task",
            description="",
            agent="coder",
            status="failed",
            result="failed task",
            subtasks=[subtask],
        )
        flow = Flow(
            id=self.flow_id,
            title="flow",
            objective="build agent",
            flow_type="agent_development",
            tasks=[task],
        )
        _supersede_failed_attempts(flow)
        self.assertEqual(task.status, "finished")
        self.assertEqual(subtask.status, "superseded")
        self.assertTrue(task.result.startswith("FINAL RECOVERY"))
        self.assertTrue(subtask.result.startswith("SUPERSEDED"))

    def test_later_validation_can_recover_parent_task(self):
        (self.root / "agent.py").write_text(
            "class Agent:\n"
            "    def crawl(self):\n"
            "        return []\n\n"
            "def main():\n"
            "    return Agent().crawl()\n"
            + "# implementation\n" * 50
        )
        failed = Subtask(
            id="s1",
            task_id="t1",
            title="Implement core scraper script",
            description="",
            agent="coder",
            status="failed",
        )
        verified = Subtask(
            id="s2",
            task_id="t1",
            title="Validate scraper execution",
            description="",
            agent="coder",
            status="finished",
            artifacts=["agent.py"],
            validation="Validation passed. Process returned meaningful output",
        )
        task = Task(
            id="t1",
            flow_id=self.flow_id,
            title="Implement Core Scraper Logic",
            description="Create the main Python script for the crawler",
            agent="coder",
            status="failed",
            subtasks=[failed, verified],
        )
        flow = Flow(
            id=self.flow_id,
            title="scraper",
            objective="Build a scraper",
            flow_type="agent_development",
            tasks=[task],
        )
        recovered, report, evidence = _reconcile_task_status(task, flow)
        self.assertTrue(recovered)
        self.assertIn("agent.py", evidence)
        self.assertIn("required deliverables", report)

    def test_text_only_agent_cannot_claim_file_creation(self):
        valid, report = _validate_text_result(
            "I've created the module and saved the file to /work/main.py"
        )
        self.assertFalse(valid)
        self.assertIn("claimed filesystem", report)


if __name__ == "__main__":
    unittest.main()
