import pytest
from lover_agent.adapters.html_parser import WeChatHTMLParser, ChatMessage


class TestWeChatHTMLParser:
    """微信 HTML 解析器测试"""

    @pytest.fixture
    def parser(self):
        return WeChatHTMLParser()

    def test_parse_single_message(self, parser):
        html = """
        <div id="chat-container">
            <div class="item item-center"><span>8:16</span></div>
            <div class="item item-left">
                <div class="avatar">
                    <img src="https://example.com/avatar.png">
                </div>
                <div class="bubble bubble-left">你好</div>
            </div>
        </div>
        """
        messages = parser.parse(html)
        assert len(messages) == 1
        assert messages[0].content == "你好"
        assert messages[0].timestamp == "8:16"

    def test_parse_multiple_messages(self, parser):
        html = """
        <div id="chat-container">
            <div class="item item-center"><span>8:16</span></div>
            <div class="item item-left">
                <div class="bubble bubble-left">你好</div>
            </div>
            <div class="item item-center"><span>8:17</span></div>
            <div class="item item-right">
                <div class="bubble bubble-right">你好呀</div>
            </div>
        </div>
        """
        messages = parser.parse(html)
        assert len(messages) == 2
        assert messages[0].content == "你好"
        assert messages[1].content == "你好呀"

    def test_parse_empty_content(self, parser):
        html = """
        <div id="chat-container">
            <div class="item item-center"><span>8:16</span></div>
            <div class="item item-left">
                <div class="bubble bubble-left"></div>
            </div>
        </div>
        """
        messages = parser.parse(html)
        assert len(messages) == 0

    def test_parse_no_chat_container(self, parser):
        html = "<div>No chat here</div>"
        messages = parser.parse(html)
        assert len(messages) == 0

    def test_parse_with_br_tags(self, parser):
        html = """
        <div id="chat-container">
            <div class="item item-left">
                <div class="bubble bubble-left">第一行<br />第二行</div>
            </div>
        </div>
        """
        messages = parser.parse(html)
        assert len(messages) == 1
        assert "第一行" in messages[0].content
        assert "第二行" in messages[0].content

    def test_parse_message_with_html_elements(self, parser):
        html = """
        <div id="chat-container">
            <div class="item item-left">
                <div class="bubble bubble-left">
                    收到一条消息<b>加粗</b>和<i>斜体</i>
                </div>
            </div>
        </div>
        """
        messages = parser.parse(html)
        assert len(messages) == 1
        assert "加粗" in messages[0].content
        assert "斜体" in messages[0].content

    def test_parse_sent_message(self, parser):
        html = """
        <div id="chat-container">
            <div class="item item-right">
                <div class="bubble bubble-right">发送的消息</div>
                <div class="avatar">
                    <img src="https://cdn.luogu.com.cn/upload/image_hosting/rbnmv4bu.png">
                </div>
            </div>
        </div>
        """
        messages = parser.parse(html)
        assert len(messages) == 1
        assert messages[0].content == "发送的消息"
        assert messages[0].sender == "自己"

    def test_parse_received_message(self, parser):
        html = """
        <div id="chat-container">
            <div class="item item-left">
                <div class="avatar">
                    <img src="https://blog.lc044.love/static/img/a774ab7a32635db7b4254c8ff7caaa89.png">
                </div>
                <div class="bubble bubble-left">收到的消息</div>
            </div>
        </div>
        """
        messages = parser.parse(html)
        assert len(messages) == 1
        assert messages[0].content == "收到的消息"

    def test_chat_message_dataclass(self):
        msg = ChatMessage(
            sender="张三",
            content="你好",
            timestamp="2024-01-01 12:00:00",
            session_id="session123"
        )
        assert msg.sender == "张三"
        assert msg.content == "你好"
        assert msg.timestamp == "2024-01-01 12:00:00"
        assert msg.session_id == "session123"

    def test_chat_message_optional_fields(self):
        msg = ChatMessage(sender="张三", content="你好")
        assert msg.sender == "张三"
        assert msg.content == "你好"
        assert msg.timestamp is None
        assert msg.session_id is None

    def test_real_html_file(self, parser):
        import os
        file_path = "E:/workspace/loverAgent/dataset/小郑猫猫ฅ^•ﻌ•^ฅ(wxid_162mkjs2c0dl22)/小郑猫猫ฅ^•ﻌ•^ฅ.html"
        if os.path.exists(file_path):
            messages = parser.parse_file(file_path)
            assert len(messages) > 0
            assert all(isinstance(m, ChatMessage) for m in messages)
            assert all(m.content for m in messages)