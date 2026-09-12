"""
Unit tests for the recoverable create-document transaction.

The tests inject a fake commit target (the ExportAction subset) so that
index-build and export failures can be triggered deterministically.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from strictdoc.backend.sdoc.writer import SDWriter
from strictdoc.core.project_config import ProjectConfig
from strictdoc.server.document_creation import (
    CreateDocumentFailure,
    CreateDocumentTransaction,
)


class FakeDocumentTree:
    def __init__(self, document_list: List["FakeDocument"]) -> None:
        self.document_list: List["FakeDocument"] = document_list


class FakeIndex:
    def __init__(
        self, document_list: Optional[List["FakeDocument"]] = None
    ) -> None:
        self.document_tree = FakeDocumentTree(
            document_list if document_list is not None else []
        )
        self.update_last_updated_calls = 0

    def update_last_updated(self) -> None:
        self.update_last_updated_calls += 1


class FakeMeta:
    def __init__(
        self, input_doc_full_path: str, output_document_dir_full_path: str
    ) -> None:
        self.input_doc_full_path = input_doc_full_path
        self.output_document_dir_full_path = (
            output_document_dir_full_path
        )

    def _output_path(self, suffix: str) -> str:
        return os.path.join(
            self.output_document_dir_full_path,
            f"document1{suffix}",
        )

    def get_html_doc_path(self) -> str:
        return self._output_path(".html")

    def get_html_table_path(self) -> str:
        return self._output_path("-TABLE.html")

    def get_html_traceability_path(self) -> str:
        return self._output_path("-TRACE.html")

    def get_html_deep_traceability_path(self) -> str:
        return self._output_path("-DEEP-TRACE.html")

    def get_html_pdf_path(self) -> str:
        return self._output_path("-PDF.html")


class FakeDocument:
    def __init__(self, meta: FakeMeta) -> None:
        self.meta = meta


class FakeExportAction:
    def __init__(
        self,
        *,
        build_error: Optional[BaseException] = None,
        export_errors: Optional[List[BaseException]] = None,
        include_document_after_build: bool = False,
        output_dir: Optional[str] = None,
        document_full_path: Optional[str] = None,
    ) -> None:
        self.old_index = FakeIndex()
        self.traceability_index = self.old_index
        self.build_error = build_error
        self.export_errors = export_errors if export_errors is not None else []
        self.build_calls = 0
        self.export_calls = 0
        self._include_document_after_build = include_document_after_build
        self._output_dir = output_dir
        self._document_full_path = document_full_path

    def build_index(self) -> FakeIndex:
        self.build_calls += 1
        if self.build_error is not None:
            raise self.build_error
        document_list: List[FakeDocument] = []
        if self._include_document_after_build:
            assert self._document_full_path is not None
            assert self._output_dir is not None
            document_list.append(
                FakeDocument(
                    FakeMeta(
                        input_doc_full_path=self._document_full_path,
                        output_document_dir_full_path=self._output_dir,
                    )
                )
            )
        self.traceability_index = FakeIndex(document_list)
        return self.traceability_index

    def export(self) -> None:
        self.export_calls += 1
        if len(self.export_errors) > 0:
            raise self.export_errors.pop(0)


class FakeWatcher:
    def __init__(self) -> None:
        self.inhibited_paths: List[str] = []

    def inhibit_next_change(self, path: str) -> None:
        self.inhibited_paths.append(path)


def _create_project_config(input_root: str) -> ProjectConfig:
    project_config = ProjectConfig.default_config()
    project_config.input_paths = [input_root]
    return project_config


def _create_transaction(
    *,
    input_root: str,
    export_action: FakeExportAction,
    watcher: Optional[FakeWatcher] = None,
) -> CreateDocumentTransaction:
    project_config = _create_project_config(input_root)
    return CreateDocumentTransaction(
        project_config=project_config,
        export_action=export_action,
        sdoc_writer=SDWriter(project_config),
        document_watcher=watcher,
    )


def test_001_create_lands_source_file_and_commits():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_root = os.path.join(temp_dir, "input")
        os.mkdir(input_root)
        export_action = FakeExportAction(
            include_document_after_build=True,
            document_full_path=os.path.join(
                input_root, "docs", "document1.sdoc"
            ),
            output_dir=os.path.join(temp_dir, "output"),
        )
        watcher = FakeWatcher()
        transaction = _create_transaction(
            input_root=input_root,
            export_action=export_action,
            watcher=watcher,
        )

        result = transaction.create(
            document_title="Document 1",
            document_path="docs/document1.sdoc",
        )

        assert result.recovered is False
        assert os.path.isfile(result.document_full_path)
        with open(result.document_full_path, encoding="utf8") as document_file:
            assert (
                document_file.read()
                == "[DOCUMENT]\nTITLE: Document 1\n"
            )
        assert export_action.build_calls == 1
        assert export_action.export_calls == 1
        assert watcher.inhibited_paths == [result.document_full_path]
        assert not list(
            Path(input_root).rglob("*.tmp")
        )


def test_002_resubmitting_same_form_adopts_file_without_duplicate():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_root = os.path.join(temp_dir, "input")
        os.mkdir(input_root)
        document_full_path = os.path.join(
            input_root, "docs", "document1.sdoc"
        )
        export_action = FakeExportAction(
            include_document_after_build=True,
            document_full_path=document_full_path,
            output_dir=os.path.join(temp_dir, "output"),
        )
        transaction = _create_transaction(
            input_root=input_root,
            export_action=export_action,
        )

        first_result = transaction.create(
            document_title="Document 1",
            document_path="docs/document1.sdoc",
        )
        with open(
            first_result.document_full_path, encoding="utf8"
        ) as document_file:
            content_after_first_create = document_file.read()

        second_result = transaction.create(
            document_title="Document 1",
            document_path="docs/document1.sdoc",
        )

        assert second_result.recovered is True
        assert (
            second_result.document_full_path
            == first_result.document_full_path
        )
        with open(
            second_result.document_full_path, encoding="utf8"
        ) as document_file:
            assert (
                document_file.read() == content_after_first_create
            )
        assert export_action.build_calls == 2
        assert export_action.export_calls == 2
        assert (
            list(Path(input_root).rglob("*.sdoc"))
            == [Path(document_full_path)]
        )


def test_003_existing_document_with_content_is_not_overwritten():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_root = Path(temp_dir) / "input"
        input_root.mkdir()
        existing_path = input_root / "docs" / "document1.sdoc"
        existing_path.parent.mkdir()
        existing_content = (
            "[DOCUMENT]\n"
            "TITLE: Document 1\n\n"
            "[REQUIREMENT]\n"
            "UID: REQ-001\n"
            "STATEMENT: Existing content.\n"
        )
        existing_path.write_text(existing_content, encoding="utf8")

        export_action = FakeExportAction()
        transaction = _create_transaction(
            input_root=str(input_root),
            export_action=export_action,
        )

        try:
            transaction.create(
                document_title="Document 1",
                document_path="docs/document1.sdoc",
            )
            raised = False
        except CreateDocumentFailure as failure:
            raised = True
            assert failure.field_name == "document_path"
            assert "already exists" in failure.message

        assert raised is True
        assert (
            existing_path.read_text(encoding="utf8")
            == existing_content
        )
        assert export_action.build_calls == 0
        assert export_action.export_calls == 0


def test_004_existing_file_with_different_title_is_rejected():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_root = Path(temp_dir) / "input"
        input_root.mkdir()
        existing_path = input_root / "document1.sdoc"
        existing_path.write_text(
            "[DOCUMENT]\nTITLE: Another title\n",
            encoding="utf8",
        )

        export_action = FakeExportAction()
        transaction = _create_transaction(
            input_root=str(input_root),
            export_action=export_action,
        )

        try:
            transaction.create(
                document_title="Document 1",
                document_path="document1.sdoc",
            )
            raised = False
        except CreateDocumentFailure as failure:
            raised = True
            assert failure.field_name == "document_path"

        assert raised is True
        assert "Another title" in existing_path.read_text(
            encoding="utf8"
        )


def test_005_unparseable_existing_file_is_rejected_and_kept():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_root = Path(temp_dir) / "input"
        input_root.mkdir()
        existing_path = input_root / "document1.sdoc"
        broken_content = "this is not an sdoc document\n"
        existing_path.write_text(broken_content, encoding="utf8")

        export_action = FakeExportAction()
        transaction = _create_transaction(
            input_root=str(input_root),
            export_action=export_action,
        )

        try:
            transaction.create(
                document_title="Document 1",
                document_path="document1.sdoc",
            )
            raised = False
        except CreateDocumentFailure:
            raised = True

        assert raised is True
        assert (
            existing_path.read_text(encoding="utf8") == broken_content
        )
        assert export_action.build_calls == 0


def test_006_write_failure_removes_everything_created():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_root = Path(temp_dir) / "input"
        input_root.mkdir()
        # A regular file blocks the creation of a directory with the
        # same name, which makes Path.mkdir fail with OSError.
        blocker_path = input_root / "blocker"
        blocker_path.write_text("x", encoding="utf8")

        export_action = FakeExportAction()
        transaction = _create_transaction(
            input_root=str(input_root),
            export_action=export_action,
        )

        try:
            transaction.create(
                document_title="Document 1",
                document_path="blocker/document1.sdoc",
            )
            raised = False
        except CreateDocumentFailure as failure:
            raised = True
            assert failure.field_name == "document_path"
            assert "could not be created" in failure.message

        assert raised is True
        assert blocker_path.read_text(encoding="utf8") == "x"
        assert export_action.build_calls == 0
        assert export_action.export_calls == 0
        assert not list(input_root.rglob("*.tmp"))


def test_007_build_failure_rolls_back_landed_file_and_index():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_root = Path(temp_dir) / "input"
        input_root.mkdir()
        document_full_path = str(
            input_root / "docs" / "document1.sdoc"
        )
        export_action = FakeExportAction(
            build_error=RuntimeError("index boom"),
        )
        watcher = FakeWatcher()
        transaction = _create_transaction(
            input_root=str(input_root),
            export_action=export_action,
            watcher=watcher,
        )

        try:
            transaction.create(
                document_title="Document 1",
                document_path="docs/document1.sdoc",
            )
            raised = False
        except CreateDocumentFailure as failure:
            raised = True
            assert "project index" in failure.message

        assert raised is True
        assert not os.path.exists(document_full_path)
        assert not (input_root / "docs").exists()
        assert export_action.traceability_index is export_action.old_index
        assert export_action.export_calls == 0
        # One inhibition for the landing write and one for the rollback
        # deletion.
        assert watcher.inhibited_paths == [
            document_full_path,
            document_full_path,
        ]

        # After the cause is fixed, a deterministic retry succeeds.
        export_action.build_error = None
        export_action._include_document_after_build = True
        export_action._document_full_path = document_full_path
        export_action._output_dir = str(
            Path(temp_dir) / "output"
        )
        result = transaction.create(
            document_title="Document 1",
            document_path="docs/document1.sdoc",
        )
        assert result.recovered is False
        assert os.path.isfile(document_full_path)
        assert export_action.build_calls == 2
        assert export_action.export_calls == 1


def test_008_build_failure_with_system_exit_is_recovered():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_root = Path(temp_dir) / "input"
        input_root.mkdir()
        document_full_path = str(
            input_root / "document1.sdoc"
        )
        export_action = FakeExportAction(
            build_error=SystemExit(1),
        )
        transaction = _create_transaction(
            input_root=str(input_root),
            export_action=export_action,
        )

        try:
            transaction.create(
                document_title="Document 1",
                document_path="document1.sdoc",
            )
            raised = False
        except CreateDocumentFailure:
            raised = True

        assert raised is True
        assert not os.path.exists(document_full_path)
        assert (
            export_action.traceability_index is export_action.old_index
        )


def test_009_export_failure_keeps_valid_source_and_restores_index():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_root = Path(temp_dir) / "input"
        input_root.mkdir()
        output_dir = Path(temp_dir) / "output" / "docs"
        output_dir.mkdir(parents=True)
        document_full_path = str(
            input_root / "docs" / "document1.sdoc"
        )
        # Simulate derived outputs produced before the export failure.
        derived_html = output_dir / "document1.html"
        derived_trace = output_dir / "document1-TRACE.html"
        derived_html.write_text("html", encoding="utf8")
        derived_trace.write_text("trace", encoding="utf8")

        export_action = FakeExportAction(
            export_errors=[RuntimeError("export boom")],
            include_document_after_build=True,
            document_full_path=document_full_path,
            output_dir=str(output_dir),
        )
        transaction = _create_transaction(
            input_root=str(input_root),
            export_action=export_action,
        )

        try:
            transaction.create(
                document_title="Document 1",
                document_path="docs/document1.sdoc",
            )
            raised = False
        except CreateDocumentFailure as failure:
            raised = True
            assert "derived outputs" in failure.message

        assert raised is True
        # The valid source document is kept for recovery.
        assert os.path.isfile(document_full_path)
        # The index is restored, and screens are forced to regenerate.
        assert (
            export_action.traceability_index is export_action.old_index
        )
        assert (
            export_action.old_index.update_last_updated_calls == 1
        )
        # The failed attempt's derived outputs are removed.
        assert not derived_html.exists()
        assert not derived_trace.exists()
        # The repair export ran after the failed attempt.
        assert export_action.export_calls == 2

        # The retry adopts the kept file and commits successfully.
        result = transaction.create(
            document_title="Document 1",
            document_path="docs/document1.sdoc",
        )
        assert result.recovered is True
        assert export_action.build_calls == 2
        assert export_action.export_calls == 3


def test_010_export_failure_after_adoption_keeps_preexisting_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_root = Path(temp_dir) / "input"
        input_root.mkdir()
        document_full_path = input_root / "document1.sdoc"
        document_full_path.write_text(
            "[DOCUMENT]\nTITLE: Document 1\n",
            encoding="utf8",
        )
        content_before = document_full_path.read_bytes()

        export_action = FakeExportAction(
            export_errors=[RuntimeError("export boom")],
        )
        transaction = _create_transaction(
            input_root=str(input_root),
            export_action=export_action,
        )

        try:
            transaction.create(
                document_title="Document 1",
                document_path="document1.sdoc",
            )
            raised = False
        except CreateDocumentFailure:
            raised = True

        assert raised is True
        assert document_full_path.read_bytes() == content_before


def test_011_markdown_document_is_idempotent_on_resubmit():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_root = Path(temp_dir) / "input"
        input_root.mkdir()
        export_action = FakeExportAction()
        transaction = _create_transaction(
            input_root=str(input_root),
            export_action=export_action,
        )

        result = transaction.create(
            document_title="Document 1",
            document_path="docs/document1.md",
        )
        assert result.recovered is False
        with open(result.document_full_path, encoding="utf8") as md_file:
            assert md_file.read() == "# Document 1\n"

        recovered_result = transaction.create(
            document_title="Document 1",
            document_path="docs/document1.md",
        )
        assert recovered_result.recovered is True
        assert export_action.build_calls == 2


def test_012_path_escaping_input_root_is_rejected():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_root = Path(temp_dir) / "input"
        input_root.mkdir()
        export_action = FakeExportAction()
        transaction = _create_transaction(
            input_root=str(input_root),
            export_action=export_action,
        )

        try:
            transaction.create(
                document_title="Document 1",
                document_path="../escape.sdoc",
            )
            raised = False
        except CreateDocumentFailure as failure:
            raised = True
            assert failure.field_name == "document_path"

        assert raised is True
        assert not (Path(temp_dir) / "escape.sdoc").exists()
        assert export_action.build_calls == 0
