from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Optional
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChatMessage:
    sender: str
    content: str
    timestamp: Optional[str] = None
    session_id: Optional[str] = None


class WeChatHTMLParser:
    """微信 HTML 聊天记录解析器

    解析微信导出的 HTML 格式聊天记录。
    HTML 结构如下:
    - 消息容器: <div id="chat-container">
    - 单条消息: <div class="item item-left"> (收到的) 或 <div class="item item-right"> (发出的)
    - 消息内容: <div class="bubble bubble-left"> 或 <div class="bubble bubble-right">
    - 时间戳: <div class="item item-center"><span>时间</span></div>
    """

    def parse(self, html_content: str) -> List[ChatMessage]:
        soup = BeautifulSoup(html_content, "lxml")
        messages = []

        chat_container = soup.find("div", id="chat-container")
        if not chat_container:
            logger.warning("未找到聊天容器 #chat-container")
            return messages

        items = chat_container.find_all("div", class_="item")

        for item in items:
            item_classes = item.get("class", [])
            if "item-center" in item_classes:
                continue

            try:
                bubble = item.find("div", class_=lambda c: c and "bubble" in c)
                if not bubble:
                    continue

                content = bubble.get_text(strip=True)
                if not content:
                    continue

                is_sent = "item-right" in item_classes

                avatar_img = item.find("div", class_="avatar")
                if avatar_img:
                    img_tag = avatar_img.find("img")
                    sender = self._extract_sender_from_avatar(img_tag, is_sent)
                else:
                    sender = "自己" if is_sent else "对方"

                timestamp = self._find_timestamp(item, items)

                messages.append(ChatMessage(
                    sender=sender,
                    content=content,
                    timestamp=timestamp
                ))
            except Exception as e:
                logger.warning(f"解析消息失败: {e}")
                continue

        return messages

    def _extract_sender_from_avatar(self, img_tag, is_sent: bool) -> str:
        if img_tag and img_tag.get("src"):
            src = img_tag["src"]
            if "wxid_aiskqnqawfgm22" in src or "162mkjs2c0dl22" in src:
                return "小郑猫猫"
            elif "rbnmv4bu" in src:
                return "自己"
        return "对方" if not is_sent else "自己"

    def _find_timestamp(self, item, all_items) -> Optional[str]:
        item_index = -1
        for i, it in enumerate(all_items):
            if it == item:
                item_index = i
                break

        if item_index > 0:
            prev_item = all_items[item_index - 1]
            if "item-center" in prev_item.get("class", []):
                span = prev_item.find("span")
                if span:
                    return span.get_text(strip=True)
        return None

    def parse_file(self, file_path: str) -> List[ChatMessage]:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return self.parse(html_content)