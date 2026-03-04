import os
import json
import requests
import json_repair
from custom_logger import logger_config

_NOTION_API_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"
_CHUNK_SIZE = 1900
_BLOCK_BATCH = 100
_DEFAULT_RESPONSE_PLACEHOLDER = "[\n  // paste JSON here\n]"


class NotionClient:
    def __init__(self):
        self.api_key = os.getenv("NOTION_API_KEY")
        self.parent_page_id = os.getenv("NOTION_PARENT_PAGE_ID", "3159a6c4-a97e-8097-8ef5-fd65ba5bfa07")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def _chunk_text(self, text):
        return [text[i:i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)]

    def _rich_text(self, content):
        return [{"type": "text", "text": {"content": chunk}} for chunk in self._chunk_text(content)]

    def _heading2_block(self, text):
        return {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": text}}]},
        }

    def _paragraph_blocks(self, content):
        chunks = self._chunk_text(content)
        blocks = []
        for i in range(0, len(chunks), _BLOCK_BATCH):
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": c}} for c in chunks[i:i + _BLOCK_BATCH]]
                },
            })
        return blocks

    def _code_block(self, content):
        return {
            "object": "block",
            "type": "code",
            "code": {"rich_text": self._rich_text(content), "language": "json"},
        }

    def _find_page(self, title):
        response = requests.post(
            f"{_NOTION_API_BASE}/search",
            headers=self._headers(),
            json={"query": title, "filter": {"value": "page", "property": "object"}},
        )
        if response.status_code != 200:
            logger_config.error(f"Notion search failed: {response.status_code} - {response.text}")
            return None

        for page in response.json().get("results", []):
            title_arr = page.get("properties", {}).get("title", {}).get("title", [])
            page_title = title_arr[0].get("plain_text", "") if title_arr else ""
            if page_title == title:
                return page
        return None

    def _get_blocks(self, page_id):
        response = requests.get(f"{_NOTION_API_BASE}/blocks/{page_id}/children", headers=self._headers())
        if response.status_code != 200:
            logger_config.warning(f"Failed to fetch blocks: {response.status_code} - {response.text}")
            return []
        return response.json().get("results", [])

    def _delete_blocks(self, blocks):
        for block in blocks:
            requests.delete(f"{_NOTION_API_BASE}/blocks/{block['id']}", headers=self._headers())

    def _append_blocks(self, page_id, blocks):
        url = f"{_NOTION_API_BASE}/blocks/{page_id}/children"
        for i in range(0, len(blocks), _BLOCK_BATCH):
            res = requests.patch(url, headers=self._headers(), json={"children": blocks[i:i + _BLOCK_BATCH]})
            if res.status_code != 200:
                logger_config.error(f"Failed to append blocks: {res.text}")

    def _create_page(self, title, blocks):
        payload = {
            "parent": {"type": "page_id", "page_id": self.parent_page_id},
            "properties": {"title": {"title": [{"type": "text", "text": {"content": title}}]}},
            "children": blocks[:_BLOCK_BATCH],
        }
        res = requests.post(f"{_NOTION_API_BASE}/pages", headers=self._headers(), json=payload)
        if res.status_code != 200:
            logger_config.error(f"Failed to create page: {res.text}")
            return None
        page_id = res.json()["id"]
        if len(blocks) > _BLOCK_BATCH:
            self._append_blocks(page_id, blocks[_BLOCK_BATCH:])
        return page_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_json_result(self, title):
        """Return parsed JSON from the first code block of the Notion page, or None."""
        if not self.api_key:
            logger_config.warning("NOTION_API_KEY not set.")
            return None

        logger_config.info(f"Searching Notion for page: '{title}'")
        page = self._find_page(title)
        if not page:
            logger_config.info(f"No Notion page found: '{title}'")
            return None

        blocks = self._get_blocks(page["id"])
        logger_config.info(f"Fetched {len(blocks)} blocks from page '{title}'")
        for i, block in enumerate(blocks):
            if block["type"] != "code":
                continue
            code_text = "".join(t["plain_text"] for t in block["code"]["rich_text"])
            logger_config.info(f"Parsing code block {i + 1}/{len(blocks)} ({len(code_text)} chars)")
            try:
                result = json_repair.loads(code_text)
                if result:
                    logger_config.info(f"Parsed JSON result: {len(result)} items")
                    return result
            except Exception as e:
                logger_config.warning(f"JSON parse failed: {e}")

        logger_config.info("No valid JSON result found in Notion page")
        return None

    def upsert_page(self, title, user_prompt, system_prompt):
        """Create or overwrite a Notion page with system/user prompts and a response code block."""
        if not self.api_key:
            logger_config.warning("NOTION_API_KEY not set.")
            return None

        page = self._find_page(title)
        saved_response = _DEFAULT_RESPONSE_PLACEHOLDER

        if page:
            existing_blocks = self._get_blocks(page["id"])
            after_response_heading = False
            for block in existing_blocks:
                if block["type"] == "heading_2":
                    rich_text = block.get("heading_2", {}).get("rich_text", [])
                    after_response_heading = bool(rich_text and "Response:" in rich_text[0].get("plain_text", ""))
                elif block["type"] == "code" and after_response_heading:
                    saved_response = "".join(t["plain_text"] for t in block["code"]["rich_text"])
                    break
            self._delete_blocks(existing_blocks)

        blocks = [
            self._heading2_block("System Prompt"),
            *self._paragraph_blocks(system_prompt),
            self._heading2_block("User Prompt"),
            *self._paragraph_blocks(user_prompt),
            self._heading2_block("Response:"),
            self._code_block(saved_response),
        ]

        if page:
            self._append_blocks(page["id"], blocks)
        else:
            self._create_page(title, blocks)


if __name__ == "__main__":
    client = NotionClient()
    test_title = "test_cache_dir_1234"

    client.upsert_page(test_title, "[Scene 1, Scene 2]", "Match scenes to recap.")
    logger_config.info("Page created/updated. Check Notion!")

    result = client.read_json_result(test_title)
    if result:
        logger_config.info("Result from Notion:")
        print(json.dumps(result, indent=4))
    else:
        logger_config.info("No valid JSON result found in Notion yet.")
