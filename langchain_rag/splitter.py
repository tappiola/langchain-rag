import glob
import os
from typing import Any, List

import hcl2
from langchain_core.documents import Document
from lark import Tree, Token


class RecursiveHCLSplitter:
    """
    Splits Terraform HCL files into one Document per top-level block (e.g. resource, data, variable),
    with relevant metadata. No size-based chunking is performed.
    """

    def __init__(self):
        pass

    def find_token_by_type(self, tree_or_tokens: list[Tree | Any], token_type: str, index: int = 0,
                           max_depth: int = 3) -> Token | None:
        found_index = 0

        if max_depth < 0:
            return None

        for child in tree_or_tokens:
            if isinstance(child, Token) and child.type == token_type:
                if found_index == index:
                    return child.value
                else:
                    found_index += 1
            elif isinstance(child, Tree):
                result = self.find_token_by_type(child.children, token_type, index, max_depth - 1)
                if result:
                    return result

        return None

    def extract_documents_from_file(self, file_path: str) -> List[Document]:
        documents = []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if file_path.endswith('tfvars') or file_path.endswith('variables.tf'):
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": file_path
                }
            ))

            return documents

        try:
            parsed = hcl2.parses(content)
        except Exception as e:
            print(f"⚠️ Skipping {file_path}: {e}")
            return []

        block_count = 0

        for block in parsed.children[0].children:
            if all(isinstance(child, Token) and child.type == 'NL_OR_COMMENT' for child in block.children):
                continue

            block_text = hcl2.writes(block)

            block_type = self.find_token_by_type(block.children, "NAME")
            name_token = self.find_token_by_type(block.children, "STRING_LIT")
            label_token = self.find_token_by_type(block.children, "STRING_LIT", 1)

            new_doc = Document(
                page_content=block_text,
                metadata={
                    "source": file_path,
                    "block_type": block_type,
                    "label": label_token,
                    "name": name_token
                }
            )

            documents.append(new_doc)
            block_count += 1

        if block_count > 1:
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": file_path
                }
            ))

        return documents

    def extract_documents_from_folder(self, folder_path: str) -> List[Document]:
        all_docs = []
        for file_path in glob.glob(os.path.join(folder_path, "*.tf*")):
            all_docs.extend(self.extract_documents_from_file(file_path))
        return all_docs
