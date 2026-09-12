"""
Unit tests for the recoverable Web UI document-creation commit.
"""

import os
from pathlib import Path

from strictdoc.backend.sdoc.errors.document_tree_error import (
    DocumentTreeError,
)
from strictdoc.core.project_config import ProjectConfig
from strictdoc.features.export.export_action import ExportAction
from strictdoc.helpers.parallelizer import NullParallelizer
from strictdoc.server import document_creation as document_creation_module
from strictdoc.server.document_creation import DocumentCreationCommit


def _create_project_config(
    input_root: Path, output_root: Path
) -> ProjectConfig:
    project_config = ProjectConfig.default_config()
    project_config.input_paths = [str(input_root)]
    project_config.output_dir = str(output_root)
    project_config.export_output_html_root = str(output_root / "html")
    project_config.export_formats = ["html"]
    return project_config


def _create_commit(
    tmp_path: Path,
) -> tuple[DocumentCreationCommit, ExportAction, Path]:
    input_root = tmp_path / "input"
    input_root.mkdir(parents=True)
    output_root = tmp_path / "output"
    project_config = _create_project_config(input_root, output_root)
    export_action = ExportAction(project_config, NullParallelizer())
    commit = DocumentCreationCommit(
        project_config=project_config,
        export_action=export_action,
        parallelizer=NullParallelizer(),
        document_watcher=None,
    )
    return commit, export_action, input_root


def _document_paths(export_action: ExportAction, document_full_path: str):
    documents = [
        document
        for document in export_action.traceability_index.document_tree.document_list
        if document.meta is not None
        and document.meta.input_doc_full_path == document_full_path
    ]
    return documents


def test_001_success_writes_file_and_refreshes_index(tmp_path):
    commit, export_action, input_root = _create_commit(tmp_path)

    outcome = commit.perform(
        title="Document 1", relative_path="docs/document1.sdoc"
    )

    assert outcome.success
    assert outcome.failure is None
    assert not outcome.already_present

    document_full_path = str(input_root / "docs" / "document1.sdoc")
    assert os.path.isfile(document_full_path)
    with open(document_full_path, encoding="utf8") as document_file:
        assert document_file.read() == "[DOCUMENT]\nTITLE: Document 1\n"

    documents = _document_paths(export_action, document_full_path)
    assert len(documents) == 1
    assert documents[0].title == "Document 1"

    assert os.path.isfile(documents[0].meta.get_html_doc_path())


def test_002_existing_file_with_other_content_is_rejected_untouched(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir(parents=True)

    existing_path = input_root / "docs" / "existing.sdoc"
    existing_path.parent.mkdir(parents=True)
    existing_path.write_text("[DOCUMENT]\nTITLE: Existing\n", encoding="utf8")

    output_root = tmp_path / "output"
    project_config = _create_project_config(input_root, output_root)
    export_action = ExportAction(project_config, NullParallelizer())
    commit = DocumentCreationCommit(
        project_config=project_config,
        export_action=export_action,
        parallelizer=NullParallelizer(),
        document_watcher=None,
    )

    outcome = commit.perform(
        title="Brand New Title", relative_path="docs/existing.sdoc"
    )

    assert not outcome.success
    assert outcome.failure is not None
    assert outcome.failure.field == "document_path"
    assert "already exists" in outcome.failure.message

    assert existing_path.read_text(encoding="utf8") == (
        "[DOCUMENT]\nTITLE: Existing\n"
    )
    assert len(_document_paths(export_action, str(existing_path))) == 1


def test_003_same_form_resubmission_is_idempotent(tmp_path):
    commit, export_action, input_root = _create_commit(tmp_path)

    first_outcome = commit.perform(
        title="Document 1", relative_path="docs/document1.sdoc"
    )
    second_outcome = commit.perform(
        title="Document 1", relative_path="docs/document1.sdoc"
    )

    assert first_outcome.success
    assert second_outcome.success
    assert second_outcome.already_present

    document_full_path = str(input_root / "docs" / "document1.sdoc")
    documents = _document_paths(export_action, document_full_path)
    assert len(documents) == 1
    assert os.path.isfile(document_full_path)


def test_004_write_failure_removes_file_directories_and_keeps_index(
    tmp_path, monkeypatch
):
    commit, export_action, input_root = _create_commit(tmp_path)
    previous_index = export_action.traceability_index

    def failing_replace(_source, _destination):
        raise OSError("simulated disk full")

    monkeypatch.setattr(document_creation_module.os, "replace", failing_replace)

    outcome = commit.perform(
        title="Document 1", relative_path="nested/deep/document1.sdoc"
    )

    assert not outcome.success
    assert outcome.failure is not None
    assert "Could not write the document file" in outcome.failure.message

    assert not (input_root / "nested").exists()
    assert export_action.traceability_index is previous_index
    assert (
        len(export_action.traceability_index.document_tree.document_list) == 0
    )


def test_005_index_failure_rolls_back_and_same_form_retries(
    tmp_path, monkeypatch
):
    commit, export_action, input_root = _create_commit(tmp_path)
    previous_index = export_action.traceability_index
    document_full_path = str(input_root / "docs" / "document1.sdoc")

    def failing_create(**_kwargs):
        raise DocumentTreeError.cycle_error("DUP-1", ["DUP-1"])

    monkeypatch.setattr(
        document_creation_module.TraceabilityIndexBuilder,
        "create",
        failing_create,
    )

    outcome = commit.perform(
        title="Document 1", relative_path="docs/document1.sdoc"
    )

    assert not outcome.success
    assert outcome.failure is not None
    assert "Failed to rebuild the project index" in outcome.failure.message

    assert not os.path.exists(document_full_path)
    assert not (input_root / "docs").exists()
    assert export_action.traceability_index is previous_index

    monkeypatch.undo()

    retry_outcome = commit.perform(
        title="Document 1", relative_path="docs/document1.sdoc"
    )

    assert retry_outcome.success
    assert os.path.isfile(document_full_path)
    assert len(_document_paths(export_action, document_full_path)) == 1


def test_006_export_failure_rolls_back_source_outputs_and_index(
    tmp_path, monkeypatch
):
    commit, export_action, input_root = _create_commit(tmp_path)
    previous_index = export_action.traceability_index
    document_full_path = str(input_root / "docs" / "document1.sdoc")

    original_export = export_action.export
    export_calls = {"count": 0}
    stale_output_paths = []

    def failing_export_first_time():
        export_calls["count"] += 1
        if export_calls["count"] == 1:
            # Simulate a derived output left behind by a failed export.
            documents = _document_paths(export_action, document_full_path)
            assert len(documents) == 1
            html_doc_path = documents[0].meta.get_html_doc_path()
            Path(html_doc_path).parent.mkdir(parents=True, exist_ok=True)
            with open(html_doc_path, "w", encoding="utf8") as html_file:
                html_file.write("<html>stale</html>")
            stale_output_paths.append(html_doc_path)
            raise RuntimeError("simulated HTML export failure")
        return original_export()

    monkeypatch.setattr(export_action, "export", failing_export_first_time)

    outcome = commit.perform(
        title="Document 1", relative_path="docs/document1.sdoc"
    )

    assert not outcome.success
    assert outcome.failure is not None
    assert "Failed to generate the HTML/PDF outputs" in (
        outcome.failure.message
    )

    # The source file is rolled back and the restored index no longer
    # contains the document.
    assert not os.path.exists(document_full_path)
    assert len(_document_paths(export_action, document_full_path)) == 0
    assert export_action.traceability_index is not previous_index
    assert (
        len(export_action.traceability_index.document_tree.document_list) == 0
    )

    # The stale derived output is removed and the recovery re-export runs.
    assert len(stale_output_paths) == 1
    assert not os.path.exists(stale_output_paths[0])
    assert export_calls["count"] == 2

    # After the cause is fixed, the same form retries deterministically.
    monkeypatch.undo()
    retry_outcome = commit.perform(
        title="Document 1", relative_path="docs/document1.sdoc"
    )
    assert retry_outcome.success
    assert os.path.isfile(document_full_path)
    assert len(_document_paths(export_action, document_full_path)) == 1


def test_007_markdown_extension_writes_markdown_document(tmp_path):
    commit, export_action, input_root = _create_commit(tmp_path)

    outcome = commit.perform(
        title="Document 1", relative_path="docs/document1.md"
    )

    assert outcome.success
    document_full_path = str(input_root / "docs" / "document1.md")
    with open(document_full_path, encoding="utf8") as document_file:
        assert document_file.read() == "# Document 1\n"
    assert len(_document_paths(export_action, document_full_path)) == 1


def test_008_path_component_that_is_a_file_is_rejected(tmp_path):
    commit, export_action, input_root = _create_commit(tmp_path)

    blocking_file = input_root / "docs"
    blocking_file.write_text("I am a file", encoding="utf8")

    outcome = commit.perform(
        title="Document 1", relative_path="docs/document1.sdoc"
    )

    assert not outcome.success
    assert outcome.failure is not None
    assert "is not a directory" in outcome.failure.message
    assert blocking_file.read_text(encoding="utf8") == "I am a file"
    assert (
        len(export_action.traceability_index.document_tree.document_list) == 0
    )
