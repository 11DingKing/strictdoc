"""
Recoverable commit for the Web UI "Add new document" action.

The creation of a document is a single commit spanning four resources:

1. the source document file on disk (plus any missing parent directories);
2. the in-memory TraceabilityIndex rebuilt from that file;
3. the HTML/PDF outputs derived from the rebuilt index;
4. the project tree rendered from the rebuilt index.

The commit either finishes all four stages or leaves the project as it was
before the commit started. A failure at any stage is reported back to the
caller as a DocumentCreationOutcome carrying a form-field error; success is
reported only after every stage succeeds.

The commit is also idempotent: re-submitting a form whose document was
already created with the very same content refreshes the index and the
derived outputs without creating a second document and without overwriting
an unrelated file that happens to share the path.
"""

import dataclasses
import os
import tempfile
from typing import List, Optional

from strictdoc.backend.markdown.writer import SDMarkdownWriter
from strictdoc.backend.sdoc.errors.document_tree_error import (
    DocumentTreeError,
)
from strictdoc.backend.sdoc.models.document import SDocDocument
from strictdoc.backend.sdoc.models.document_grammar import DocumentGrammar
from strictdoc.backend.sdoc.writer import SDWriter
from strictdoc.core.document_meta import DocumentMeta
from strictdoc.core.project_config import ProjectConfig
from strictdoc.core.traceability_index import TraceabilityIndex
from strictdoc.core.traceability_index_builder import TraceabilityIndexBuilder
from strictdoc.features.export.export_action import ExportAction
from strictdoc.helpers.parallelizer import Parallelizer
from strictdoc.helpers.paths import SDocRelativePath
from strictdoc.server.document_watcher import DocumentWatcher

DOCUMENT_PATH_FORM_FIELD = "document_path"


@dataclasses.dataclass
class DocumentCreationFailure:
    field: str
    message: str


@dataclasses.dataclass
class DocumentCreationOutcome:
    success: bool
    failure: Optional[DocumentCreationFailure]
    # True when the source file was already present with the exact content
    # this commit would have written, so the commit only refreshed the index
    # and the derived outputs (idempotent form re-submission).
    already_present: bool = False

    @staticmethod
    def successful(*, already_present: bool) -> "DocumentCreationOutcome":
        return DocumentCreationOutcome(
            success=True, failure=None, already_present=already_present
        )

    @staticmethod
    def failed(field: str, message: str) -> "DocumentCreationOutcome":
        return DocumentCreationOutcome(
            success=False,
            failure=DocumentCreationFailure(field=field, message=message),
        )


@dataclasses.dataclass
class _DocumentTarget:
    relative_path: str
    full_input_path: str
    document_full_path: str
    document_directory: str
    document_file_name: str
    input_doc_dir_rel_path: str
    input_doc_assets_dir_rel_path: str


class DocumentCreationCommit:
    def __init__(
        self,
        *,
        project_config: ProjectConfig,
        export_action: ExportAction,
        parallelizer: Parallelizer,
        document_watcher: Optional[DocumentWatcher],
    ) -> None:
        self.project_config: ProjectConfig = project_config
        self.export_action: ExportAction = export_action
        self.parallelizer: Parallelizer = parallelizer
        self.document_watcher: Optional[DocumentWatcher] = document_watcher
        self.sdoc_writer: SDWriter = SDWriter(project_config)

        # Set to True when the target file already holds exactly the content
        # this commit would write (an idempotent form re-submission).
        self._source_already_present: bool = False
        # Set to True only after the target file is atomically put in place
        # by this commit; rollback removes the file only in this case.
        self._source_file_created: bool = False
        self._created_directories: List[str] = []

    def perform(
        self, *, title: str, relative_path: str
    ) -> DocumentCreationOutcome:
        """
        Run the commit. Must be called while the server's global write lock
        is held.
        """

        self._source_already_present = False
        self._source_file_created = False
        self._created_directories = []

        target = self._resolve_target_paths(relative_path)
        document = self._build_empty_document(title, target)
        document_content = self._render_document_content(document)

        preflight_failure = self._run_preflight(target, document_content)
        if preflight_failure is not None:
            return preflight_failure

        if not self._source_already_present:
            try:
                self._stage_source_file(target, document_content)
            except Exception as error:  # noqa: BLE001
                self._rollback_created_paths(target)
                return DocumentCreationOutcome.failed(
                    DOCUMENT_PATH_FORM_FIELD,
                    "Could not write the document file at "
                    f"'{relative_path}': {error}",
                )

        try:
            self._commit_rebuilt_index()
        except DocumentTreeError as error:
            self._rollback_created_paths(target)
            return DocumentCreationOutcome.failed(
                DOCUMENT_PATH_FORM_FIELD,
                "Failed to rebuild the project index. The created document "
                "was rolled back. " + error.to_validation_message(),
            )
        except Exception as error:  # noqa: BLE001
            self._rollback_created_paths(target)
            return DocumentCreationOutcome.failed(
                DOCUMENT_PATH_FORM_FIELD,
                "Failed to rebuild the project index. The created document "
                f"was rolled back. {error}",
            )

        try:
            self.export_action.export()
        except Exception as export_error:  # noqa: BLE001
            self._recover_after_failed_export(target)
            return DocumentCreationOutcome.failed(
                DOCUMENT_PATH_FORM_FIELD,
                "Failed to generate the HTML/PDF outputs. The created "
                f"document was rolled back. {export_error}",
            )

        return DocumentCreationOutcome.successful(
            already_present=self._source_already_present
        )

    def _resolve_target_paths(self, relative_path: str) -> _DocumentTarget:
        assert isinstance(self.project_config.input_paths, list)
        full_input_path = os.path.abspath(self.project_config.input_paths[0])
        document_full_path = os.path.join(full_input_path, relative_path)
        document_directory = os.path.dirname(document_full_path)
        document_file_name = os.path.basename(document_full_path)
        input_doc_dir_rel_path = os.path.dirname(relative_path)
        file_tree_mount_folder = os.path.basename(
            os.path.dirname(full_input_path)
        )
        if len(input_doc_dir_rel_path) > 0:
            input_doc_assets_dir_rel_path = "/".join(
                (
                    file_tree_mount_folder,
                    input_doc_dir_rel_path,
                    "_assets",
                )
            )
        else:
            input_doc_assets_dir_rel_path = "/".join(
                (file_tree_mount_folder, "_assets")
            )
        return _DocumentTarget(
            relative_path=relative_path,
            full_input_path=full_input_path,
            document_full_path=document_full_path,
            document_directory=document_directory,
            document_file_name=document_file_name,
            input_doc_dir_rel_path=input_doc_dir_rel_path,
            input_doc_assets_dir_rel_path=input_doc_assets_dir_rel_path,
        )

    def _build_empty_document(
        self, title: str, target: _DocumentTarget
    ) -> SDocDocument:
        document = SDocDocument(
            mid=None,
            title=title,
            config=None,
            view=None,
            grammar=DocumentGrammar.create_default(parent=None),
            section_contents=[],
        )
        # FIXME: Fill in the document meta correctly.
        document.meta = DocumentMeta(
            level=0,
            file_tree_mount_folder="NOT_RELEVANT",
            document_filename=target.document_file_name,
            document_filename_base="NOT_RELEVANT",
            input_doc_full_path=target.document_full_path,
            input_doc_rel_path=SDocRelativePath(target.relative_path),
            input_doc_dir_rel_path=SDocRelativePath(
                target.input_doc_dir_rel_path
            ),
            input_doc_assets_dir_rel_path=SDocRelativePath(
                target.input_doc_assets_dir_rel_path
            ),
            output_document_dir_full_path="NOT_RELEVANT",
            output_document_dir_rel_path=SDocRelativePath("FIXME"),
        )
        return document

    def _render_document_content(self, document: SDocDocument) -> str:
        assert document.meta is not None
        if document.meta.input_doc_full_path.lower().endswith(
            (".md", ".markdown")
        ):
            return SDMarkdownWriter.write(
                document,
                line_width=self.project_config.document_line_width,
            )
        return self.sdoc_writer.write(document)

    # The preflight is read-only: it must not create directories or files, so
    # that a rejected commit has nothing to clean up.
    def _run_preflight(
        self, target: _DocumentTarget, document_content: str
    ) -> Optional[DocumentCreationOutcome]:
        ancestor_failure = self._validate_existing_ancestors(target)
        if ancestor_failure is not None:
            return ancestor_failure

        if not os.path.lexists(target.document_full_path):
            self._source_already_present = False
            return None

        try:
            is_directory = os.path.isdir(target.document_full_path)
            with open(
                target.document_full_path, "rb"
            ) as existing_document_file:
                existing_content = existing_document_file.read()
        except OSError as error:
            return DocumentCreationOutcome.failed(
                DOCUMENT_PATH_FORM_FIELD,
                "Could not access the document file at "
                f"'{target.document_full_path}': {error}",
            )

        if (
            not is_directory
            and existing_content == document_content.encode("utf8")
        ):
            # Idempotent re-submission of the same form: the file already
            # contains exactly what this commit would write.
            self._source_already_present = True
            return None

        return DocumentCreationOutcome.failed(
            DOCUMENT_PATH_FORM_FIELD,
            "A document already exists at this path. Choose another path "
            "or remove the existing file first.",
        )

    def _validate_existing_ancestors(
        self, target: _DocumentTarget
    ) -> Optional[DocumentCreationOutcome]:
        current_path = target.document_directory
        while not os.path.isdir(current_path):
            if os.path.lexists(current_path):
                return DocumentCreationOutcome.failed(
                    DOCUMENT_PATH_FORM_FIELD,
                    f"'{current_path}' exists and is not a directory.",
                )
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:
                break
            current_path = parent_path
        return None

    def _stage_source_file(
        self, target: _DocumentTarget, document_content: str
    ) -> None:
        self._created_directories = self._create_missing_directories(target)

        self._inhibit_watcher(target.document_full_path)

        temporary_file_descriptor, temporary_file_path = tempfile.mkstemp(
            prefix="." + target.document_file_name + ".",
            suffix=".tmp",
            dir=target.document_directory,
        )
        try:
            with os.fdopen(
                temporary_file_descriptor, "w", encoding="utf8"
            ) as temporary_file:
                temporary_file.write(document_content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            # The atomic rename is the commit point on disk: a rejected
            # write, a full disk or a crash can only leave the temporary
            # file, never a half-written target document.
            os.replace(
                temporary_file_path, target.document_full_path
            )
        except Exception:
            self._remove_file_if_exists(temporary_file_path)
            raise
        self._source_file_created = True

    def _create_missing_directories(
        self, target: _DocumentTarget
    ) -> List[str]:
        missing_directories: List[str] = []
        current_path = target.document_directory
        while not os.path.isdir(current_path):
            missing_directories.append(current_path)
            current_path = os.path.dirname(current_path)

        created_directories: List[str] = []
        for directory in reversed(missing_directories):
            os.mkdir(directory)
            created_directories.append(directory)
        return created_directories

    def _commit_rebuilt_index(self) -> None:
        traceability_index = TraceabilityIndexBuilder.create(
            project_config=self.project_config,
            parallelizer=self.parallelizer,
        )
        # Swap the index only after the rebuild fully succeeds. A failed
        # rebuild therefore leaves the previous, consistent index in place.
        self.export_action.traceability_index = traceability_index

    def _rollback_created_paths(self, target: _DocumentTarget) -> None:
        if self._source_file_created:
            self._remove_file_if_exists(
                target.document_full_path, inhibit_watcher=True
            )
            self._source_file_created = False
        for directory in reversed(self._created_directories):
            try:
                os.rmdir(directory)
            except OSError:
                # Only the empty directories created by this commit are
                # removed. Anything else is left untouched.
                continue

    def _recover_after_failed_export(self, target: _DocumentTarget) -> None:
        derived_output_paths = self._collect_derived_output_paths(
            self.export_action.traceability_index, target
        )

        # Undo the source-file part of the commit first, then rebuild the
        # index from the restored disk so that the in-memory state matches
        # the on-disk state again.
        self._rollback_created_paths(target)

        try:
            restored_index = TraceabilityIndexBuilder.create(
                project_config=self.project_config,
                parallelizer=self.parallelizer,
            )
            self.export_action.traceability_index = restored_index
        except Exception as recovery_error:  # noqa: BLE001
            print(  # noqa: T201
                "SERVER: failed to rebuild the project index while rolling "
                f"back a document creation: {recovery_error}"
            )
            return

        for output_path in derived_output_paths:
            self._remove_file_if_exists(output_path)

        try:
            # Re-generate the shared pages (project tree, index) without the
            # rolled-back document so the derived outputs match the disk.
            self.export_action.export()
        except Exception as recovery_error:  # noqa: BLE001
            print(  # noqa: T201
                "SERVER: failed to re-generate the HTML/PDF outputs while "
                f"rolling back a document creation: {recovery_error}"
            )

    def _collect_derived_output_paths(
        self, traceability_index: TraceabilityIndex, target: _DocumentTarget
    ) -> List[str]:
        for document in traceability_index.document_tree.document_list:
            if document.meta is None:
                continue
            if (
                document.meta.input_doc_full_path
                != target.document_full_path
            ):
                continue
            return [
                document.meta.get_html_doc_path(),
                document.meta.get_html_table_path(),
                document.meta.get_html_traceability_path(),
                document.meta.get_html_deep_traceability_path(),
                document.meta.get_html_pdf_path(),
            ]
        return []

    def _inhibit_watcher(self, path: str) -> None:
        if self.document_watcher is not None:
            self.document_watcher.inhibit_next_change(path)

    def _remove_file_if_exists(
        self, path: str, *, inhibit_watcher: bool = False
    ) -> None:
        if inhibit_watcher:
            self._inhibit_watcher(path)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as error:
            print(  # noqa: T201
                f"SERVER: failed to remove '{path}' during rollback: {error}"
            )
