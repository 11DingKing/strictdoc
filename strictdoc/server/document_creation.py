"""
Recoverable create-document flow for the StrictDoc Web UI.

The flow is staged around a single commit state:

1. Validate the target path and serialize the new document in memory.
2. Land the source file atomically (temporary file plus os.replace).
3. Rebuild the TraceabilityIndex (and its GraphDatabase) and regenerate
   the derived outputs.

A failure at any stage is either rolled back or left in a recoverable
state. The browser is only informed about success after every stage
succeeds. Repeating the same submission adopts the document left behind
by a previous attempt instead of creating a duplicate.
"""

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

from strictdoc.backend.markdown.reader import SDMarkdownReader
from strictdoc.backend.markdown.writer import SDMarkdownWriter
from strictdoc.backend.sdoc.models.document import SDocDocument
from strictdoc.backend.sdoc.models.document_grammar import (
    DocumentGrammar,
)
from strictdoc.backend.sdoc.reader import SDReader
from strictdoc.backend.sdoc.writer import SDWriter
from strictdoc.core.document_meta import DocumentMeta
from strictdoc.core.project_config import ProjectConfig
from strictdoc.core.traceability_index import TraceabilityIndex
from strictdoc.helpers.file_system import file_open_read_utf8
from strictdoc.helpers.paths import SDocRelativePath

# The file mode mirrors the default mode of open(..., "w"), which is
# 0o666 modified by the process umask (0o644 for the usual 0o022).
# tempfile.mkstemp creates files with 0o600, which is stricter than the
# mode the previous create-document flow used to produce.
_DEFAULT_DOCUMENT_FILE_MODE = 0o644


class CreateDocumentFailure(Exception):
    """
    Raised when a document cannot be created.

    The field name matches the Web UI form field that the failure is
    reported against.
    """

    def __init__(self, field_name: str, message: str) -> None:
        super().__init__(message)
        self.field_name = field_name
        self.message = message


class DocumentCommitTarget(Protocol):
    """
    The subset of ExportAction that the transaction commits against.
    """

    traceability_index: TraceabilityIndex

    def build_index(self) -> TraceabilityIndex: ...

    def export(self) -> None: ...


class DocumentChangeInhibitor(Protocol):
    """
    The subset of DocumentWatcher that suppresses watcher-triggered
    rebuilds for files the server writes itself.
    """

    def inhibit_next_change(self, path: str) -> None: ...


@dataclass
class CreatedDocument:
    document_full_path: str
    recovered: bool


@dataclass
class _DocumentPaths:
    input_root_full_path: str
    document_full_path: str
    document_full_path_dir: str
    document_file_name: str
    document_dir_rel_path: str
    file_tree_mount_folder: str
    assets_dir_rel_path: str


@dataclass
class _PreparedSource:
    recovered: bool
    created_directories: List[str]


class CreateDocumentTransaction:
    def __init__(
        self,
        *,
        project_config: ProjectConfig,
        export_action: DocumentCommitTarget,
        sdoc_writer: SDWriter,
        document_watcher: Optional[DocumentChangeInhibitor] = None,
    ) -> None:
        self.project_config: ProjectConfig = project_config
        self.export_action: DocumentCommitTarget = export_action
        self.sdoc_writer: SDWriter = sdoc_writer
        self.document_watcher: Optional[DocumentChangeInhibitor] = (
            document_watcher
        )

    def create(
        self, *, document_title: str, document_path: str
    ) -> CreatedDocument:
        """
        Run the staged create-document flow.

        Raises CreateDocumentFailure when any stage fails. No exception
        means that the source file, the index and the derived outputs are
        consistent with each other.
        """

        document_paths = self._resolve_document_paths(document_path)
        prepared_source = self._prepare_source_file(
            document_title=document_title,
            document_path=document_path,
            document_paths=document_paths,
        )
        self._commit(
            document_paths=document_paths,
            prepared_source=prepared_source,
        )
        return CreatedDocument(
            document_full_path=document_paths.document_full_path,
            recovered=prepared_source.recovered,
        )

    def _resolve_document_paths(self, document_path: str) -> _DocumentPaths:
        assert isinstance(self.project_config.input_paths, list)
        input_root_full_path = os.path.abspath(
            self.project_config.input_paths[0]
        )
        document_full_path = os.path.join(input_root_full_path, document_path)
        document_full_path = os.path.abspath(document_full_path)

        # The form validation already rejects ".." components. This check
        # keeps the transaction safe when it is reused outside of the Web
        # UI form.
        if not self._is_path_within(document_full_path, input_root_full_path):
            raise CreateDocumentFailure(
                "document_path",
                "Document path must remain within the project input path.",
            )

        document_full_path_dir = os.path.dirname(document_full_path)
        document_file_name = os.path.basename(document_full_path)
        document_dir_rel_path = os.path.dirname(document_path)
        file_tree_mount_folder = os.path.basename(
            os.path.dirname(input_root_full_path)
        )
        if len(document_dir_rel_path) > 0:
            assets_dir_rel_path = "/".join(
                (
                    file_tree_mount_folder,
                    document_dir_rel_path,
                    "_assets",
                )
            )
        else:
            assets_dir_rel_path = "/".join((file_tree_mount_folder, "_assets"))

        return _DocumentPaths(
            input_root_full_path=input_root_full_path,
            document_full_path=document_full_path,
            document_full_path_dir=document_full_path_dir,
            document_file_name=document_file_name,
            document_dir_rel_path=document_dir_rel_path,
            file_tree_mount_folder=file_tree_mount_folder,
            assets_dir_rel_path=assets_dir_rel_path,
        )

    def _prepare_source_file(
        self,
        *,
        document_title: str,
        document_path: str,
        document_paths: _DocumentPaths,
    ) -> _PreparedSource:
        existing_document = self._read_existing_document(
            document_title=document_title,
            document_paths=document_paths,
        )
        if existing_document is not None:
            # The same form was submitted again, or a previous attempt
            # landed the source file but failed while committing. Adopt
            # the file instead of overwriting it or creating a duplicate.
            return _PreparedSource(
                recovered=True,
                created_directories=[],
            )

        if os.path.exists(document_paths.document_full_path):
            raise CreateDocumentFailure(
                "document_path",
                "A document already exists at this path. Choose a "
                "different path or open the existing document.",
            )

        created_directories = self._plan_missing_directories(document_paths)
        document = self._build_document(
            document_title=document_title,
            document_path=document_path,
            document_paths=document_paths,
        )
        document_content = self._serialize_document(
            document_paths=document_paths,
            document=document,
        )
        self._land_source_file(
            document_paths=document_paths,
            document_content=document_content,
            created_directories=created_directories,
        )
        return _PreparedSource(
            recovered=False,
            created_directories=created_directories,
        )

    def _commit(
        self,
        *,
        document_paths: _DocumentPaths,
        prepared_source: _PreparedSource,
    ) -> None:
        previous_index = self.export_action.traceability_index
        new_index: Optional[TraceabilityIndex] = None
        try:
            self.export_action.build_index()
            new_index = self.export_action.traceability_index
            self.export_action.export()
        except (Exception, SystemExit) as commit_error:
            self._rollback_after_commit_failure(
                document_paths=document_paths,
                prepared_source=prepared_source,
                previous_index=previous_index,
                new_index=new_index,
                commit_error=commit_error,
            )

    def _build_document(
        self,
        *,
        document_title: str,
        document_path: str,
        document_paths: _DocumentPaths,
    ) -> SDocDocument:
        document = SDocDocument(
            mid=None,
            title=document_title,
            config=None,
            view=None,
            grammar=DocumentGrammar.create_default(parent=None),
            section_contents=[],
        )
        # FIXME: Fill in the document meta correctly.
        document.meta = DocumentMeta(
            level=0,
            file_tree_mount_folder="NOT_RELEVANT",
            document_filename=document_paths.document_file_name,
            document_filename_base="NOT_RELEVANT",
            input_doc_full_path=document_paths.document_full_path,
            input_doc_rel_path=SDocRelativePath(document_path),
            input_doc_dir_rel_path=SDocRelativePath(
                document_paths.document_dir_rel_path
            ),
            input_doc_assets_dir_rel_path=SDocRelativePath(
                document_paths.assets_dir_rel_path
            ),
            output_document_dir_full_path="NOT_RELEVANT",
            output_document_dir_rel_path=SDocRelativePath("FIXME"),
        )
        return document

    def _serialize_document(
        self,
        *,
        document_paths: _DocumentPaths,
        document: SDocDocument,
    ) -> str:
        if self._is_markdown_path(document_paths.document_full_path):
            return SDMarkdownWriter.write(
                document,
                line_width=self.project_config.document_line_width,
            )
        return self.sdoc_writer.write(document)

    def _read_existing_document(
        self,
        *,
        document_title: str,
        document_paths: _DocumentPaths,
    ) -> Optional[SDocDocument]:
        """
        Recognize the target file as a document that a previous identical
        submission started creating. Everything else (a document with
        content, a different title, an unparseable file) is treated as an
        unrelated existing file that must not be touched.
        """

        if not os.path.isfile(document_paths.document_full_path):
            return None
        try:
            with file_open_read_utf8(
                document_paths.document_full_path
            ) as document_file:
                document_content = document_file.read()
            if self._is_markdown_path(document_paths.document_full_path):
                existing_document = SDMarkdownReader.read(
                    document_content,
                    document_paths.document_full_path,
                    self.project_config,
                )
            else:
                existing_document = SDReader.read(
                    document_content,
                    file_path=document_paths.document_full_path,
                )
        except Exception:  # noqa: BLE001
            return None
        if existing_document.title != document_title:
            return None
        if existing_document.has_any_nodes():
            return None
        return existing_document

    def _land_source_file(
        self,
        *,
        document_paths: _DocumentPaths,
        document_content: str,
        created_directories: List[str],
    ) -> None:
        temporary_path: Optional[str] = None
        try:
            Path(document_paths.document_full_path_dir).mkdir(
                parents=True,
                exist_ok=True,
            )
            file_descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{document_paths.document_file_name}.",
                suffix=".tmp",
                dir=document_paths.document_full_path_dir,
            )
            with os.fdopen(
                file_descriptor,
                "w",
                encoding="utf8",
            ) as output_file:
                output_file.write(document_content)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.chmod(
                temporary_path,
                _DEFAULT_DOCUMENT_FILE_MODE,
            )
            self._inhibit_next_change(document_paths.document_full_path)
            os.replace(
                temporary_path,
                document_paths.document_full_path,
            )
            temporary_path = None
            self._synchronize_directory(document_paths.document_full_path_dir)
        except OSError as os_error:
            if temporary_path is not None:
                self._remove_file_if_exists(temporary_path)
            self._remove_empty_directories(created_directories)
            raise CreateDocumentFailure(
                "document_path",
                "The document file could not be created at this path: "
                f"{os_error}. Fix the file permissions or free disk "
                "space and retry.",
            ) from os_error

    def _rollback_after_commit_failure(
        self,
        *,
        document_paths: _DocumentPaths,
        prepared_source: _PreparedSource,
        previous_index: TraceabilityIndex,
        new_index: Optional[TraceabilityIndex],
        commit_error: BaseException,
    ) -> None:
        # Restore the index, including its GraphDatabase, to the state it
        # had before the commit so in-memory data never points to a
        # half-committed tree.
        self.export_action.traceability_index = previous_index

        if new_index is None:
            # The index build failed. The landed source file is not part
            # of any consistent tree, so remove the files created by this
            # attempt. Derived outputs were never generated.
            if not prepared_source.recovered:
                self._inhibit_next_change(document_paths.document_full_path)
                self._remove_file_if_exists(document_paths.document_full_path)
                self._remove_empty_directories(
                    prepared_source.created_directories
                )
            raise CreateDocumentFailure(
                "document_path",
                "The document could not be added to the project index: "
                f"{commit_error}. No partial document was kept. Fix the "
                "cause and retry.",
            ) from commit_error

        # The index build succeeded but exporting derived outputs failed.
        # The source file is a valid document, so keep it: a retry adopts
        # it deterministically and resumes at the export stage. Remove the
        # derived outputs that this attempt produced, then regenerate the
        # shared screens from the restored index.
        new_document_meta = self._find_document_meta(
            traceability_index=new_index,
            document_full_path=document_paths.document_full_path,
        )
        if new_document_meta is not None:
            self._remove_derived_outputs(new_document_meta)
        previous_index.update_last_updated()
        self._regenerate_derived_outputs_after_rollback()
        raise CreateDocumentFailure(
            "document_path",
            "The document was created but its derived outputs could not "
            f"be generated: {commit_error}. The source document was "
            "kept. Fix the cause and submit the same form again to "
            "finish the creation.",
        ) from commit_error

    def _regenerate_derived_outputs_after_rollback(self) -> None:
        try:
            self.export_action.export()
        except (Exception, SystemExit) as recovery_error:
            # The failure is already reported to the browser. On-demand
            # screen generation and the bumped index timestamp converge
            # the derived outputs to the restored index later.
            print(  # noqa: T201
                "CREATE: could not regenerate derived outputs after "
                f"rolling back a failed creation: {recovery_error}"
            )

    def _find_document_meta(
        self,
        *,
        traceability_index: TraceabilityIndex,
        document_full_path: str,
    ) -> Optional[DocumentMeta]:
        for document in traceability_index.document_tree.document_list:
            if document.meta is None:
                continue
            if document.meta.input_doc_full_path == document_full_path:
                return document.meta
        return None

    def _remove_derived_outputs(self, document_meta: DocumentMeta) -> None:
        derived_paths = [
            document_meta.get_html_doc_path(),
            document_meta.get_html_table_path(),
            document_meta.get_html_traceability_path(),
            document_meta.get_html_deep_traceability_path(),
            document_meta.get_html_pdf_path(),
        ]
        for derived_path in derived_paths:
            self._remove_file_if_exists(derived_path)
        try:
            os.rmdir(document_meta.output_document_dir_full_path)
        except OSError:
            # The directory does not exist or still contains other
            # files. It is safe to leave it in both cases.
            pass

    def _plan_missing_directories(
        self, document_paths: _DocumentPaths
    ) -> List[str]:
        """
        Collect the leaf-to-root chain of directories that do not exist
        yet, so a rollback removes only directories created by this
        attempt.
        """

        missing_directories: List[str] = []
        current_path = document_paths.document_full_path_dir
        root_path = document_paths.input_root_full_path
        while (
            len(current_path) >= len(root_path)
            and current_path != root_path
            and not os.path.exists(current_path)
        ):
            missing_directories.append(current_path)
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:
                break
            current_path = parent_path
        return missing_directories

    def _remove_empty_directories(self, created_directories: List[str]) -> None:
        for directory_path in created_directories:
            try:
                os.rmdir(directory_path)
            except OSError:
                # The directory does not exist or is not empty. Prefer
                # leaving it over deleting user content.
                pass

    def _inhibit_next_change(self, path: str) -> None:
        if self.document_watcher is not None:
            self.document_watcher.inhibit_next_change(path)

    @staticmethod
    def _remove_file_if_exists(file_path: str) -> None:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        except OSError:
            # The remaining file can be removed manually. Keeping it is
            # safer than hiding a persistent permission problem.
            pass

    @staticmethod
    def _synchronize_directory(directory_path: str) -> None:
        if sys.platform == "win32":
            return
        try:
            directory_descriptor = os.open(
                directory_path,
                os.O_RDONLY,
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            # Directory fsync is a durability hint, not a requirement
            # for process-level consistency.
            pass

    @staticmethod
    def _is_path_within(child_path: str, parent_path: str) -> bool:
        try:
            common_path = os.path.commonpath([child_path, parent_path])
        except ValueError:
            return False
        return common_path == parent_path

    @staticmethod
    def _is_markdown_path(document_full_path: str) -> bool:
        return document_full_path.lower().endswith((".md", ".markdown"))
