import json
import threading
import requests
from core.config import get_logger, Config


class FollowinClient:
    """
    Followin MCP (Model Context Protocol) 輕量客戶端。
    只呼叫免費的 news 工具 (新聞檢索)，不會碰 signal / metrics / twitter 等按額度計費的工具。
    """
    def __init__(self):
        self.logger = get_logger(__name__)
        self.url = getattr(Config, 'FOLLOWIN_MCP_URL', 'https://mcp.followin.io/v2/mcp')
        self.api_key = getattr(Config, 'FOLLOWIN_API_KEY', '')
        self.session_id = None
        self._lock = threading.Lock()

    def _headers(self):
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            'x-api-key': self.api_key,
        }
        if self.session_id:
            headers['Mcp-Session-Id'] = self.session_id
        return headers

    def _parse_sse(self, response):
        """
        解析 MCP Streamable HTTP 回傳的 SSE 格式。
        用 response.text (非串流讀取) 讓 requests 處理好 chunked transfer 解碼，
        再依 SSE 規範把同一個 event 內所有 data: 行用 \n 接回去還原完整 JSON。
        """
        data_lines = []
        result = None
        # 注意：故意用 split('\n') 而不是 str.splitlines()，
        # 因為 splitlines() 連 Unicode 行分隔符 (如 U+2028) 都會切開，
        # 這些字元偶爾會出現在新聞內文裡，用 splitlines() 會把一行 JSON 切碎導致解析失敗。
        for line in response.text.split('\n'):
            if line == '':
                if data_lines:
                    payload = '\n'.join(data_lines)
                    try:
                        result = json.loads(payload)
                    except json.JSONDecodeError:
                        pass
                    data_lines = []
                continue
            if line.startswith('data:'):
                chunk = line[len('data:'):]
                if chunk.startswith(' '):
                    chunk = chunk[1:]
                data_lines.append(chunk)
        if data_lines:
            payload = '\n'.join(data_lines)
            try:
                result = json.loads(payload)
            except json.JSONDecodeError:
                pass
        return result

    def _initialize(self):
        body = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "crypto-bot", "version": "1.0"},
            },
        }
        resp = requests.post(self.url, headers=self._headers(), json=body, timeout=15)
        resp.raise_for_status()
        self.session_id = resp.headers.get('Mcp-Session-Id')
        self._parse_sse(resp)

        # MCP 協議規定 initialize 成功後要送出 initialized 通知
        notify_body = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        requests.post(self.url, headers=self._headers(), json=notify_body, timeout=15)

    def search_news(self, query, asset_type='crypto', time_range='4h', limit=8, sources=None, verbosity='concise'):
        """
        呼叫免費的 news 工具。成功回傳 dict (results.articles / results.social)，失敗回傳 None。
        """
        with self._lock:
            try:
                if not self.session_id:
                    self._initialize()

                args = {
                    'query': query,
                    'asset_type': asset_type,
                    'time_range': time_range,
                    'limit': limit,
                    'verbosity': verbosity,
                }
                if sources:
                    args['sources'] = sources

                body = {
                    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "news", "arguments": args},
                }
                resp = requests.post(self.url, headers=self._headers(), json=body, timeout=20)

                # session 過期時通常回 400/404，重新握手一次再試
                if resp.status_code in (400, 404):
                    self.session_id = None
                    self._initialize()
                    resp = requests.post(self.url, headers=self._headers(), json=body, timeout=20)

                resp.raise_for_status()
                envelope = self._parse_sse(resp)
                if not envelope:
                    return None

                result = envelope.get('result', {})
                if result.get('isError'):
                    self.logger.warning(f"Followin news 工具回傳錯誤: {result}")
                    return None

                content = result.get('content', [])
                if not content:
                    return None
                text = content[0].get('text', '')
                return json.loads(text) if text else None
            except Exception as e:
                self.logger.warning(f"⚠️ Followin 新聞查詢失敗: {e}")
                return None


# 匯出單例供全域使用 (共用同一個 MCP session)
followin_client = FollowinClient()
