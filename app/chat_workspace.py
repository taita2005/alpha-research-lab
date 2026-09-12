import json
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_ROOT = Path(
    os.environ.get(
        "ALPHA_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
).resolve()

if str(BASE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(BASE_ROOT / "src"))

from alpha_lab import chat_service, research_runs
from alpha_lab.session_workspace import create_workspace

st.set_page_config(
    page_title="Alpha Research Chat",
    page_icon="🔬",
    layout="wide",
)

live_setting = os.environ.get("ALPHA_ENABLE_LIVE_RESEARCH")
if live_setting is None:
    try:
        live_setting = st.secrets.get(
            "ALPHA_ENABLE_LIVE_RESEARCH", False
        )
    except FileNotFoundError:
        live_setting = False

live = str(live_setting).lower() in ("1", "true")

if live:
    if "live_workspace_path" not in st.session_state:
        with st.spinner("Preparing research data..."):
            st.session_state.live_workspace_path = str(
                create_workspace(BASE_ROOT)
            )
    ROOT = Path(st.session_state.live_workspace_path)
else:
    ROOT = BASE_ROOT

defaults = {
    "alpha_messages": [],
    "alpha_parent": None,
    "alpha_busy": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "alpha_next_parent" in st.session_state:
    st.session_state.alpha_parent = st.session_state.pop("alpha_next_parent")

saved_runs = research_runs.list_runs(ROOT)
usable_ids = [
    row["request_id"]
    for row in saved_runs
    if row.get("review_available")
]

with st.sidebar:
    st.title("Alpha Research")
    st.caption("Bounded-context research assistant")

    if st.button("Clear API key"):
        st.session_state["alpha_key"] = ""

    api_key = st.text_input(
        "Gemini API key",
        type="password",
        key="alpha_key",
    )

    mode_label = st.radio(
        "Chế độ",
        ["Thảo luận", "Cải tiến alpha"],
        key="alpha_mode",
    )
    mode = "discuss" if mode_label == "Thảo luận" else "improve"

    pinned = st.text_area(
        "Chỉ dẫn giữ qua các lượt",
        max_chars=250,
        key="alpha_pinned",
        help="Ví dụ: Ưu tiên công thức đơn giản, giải thích bằng tiếng Việt.",
    )

    # The parent selector is instantiated after previous-turn updates.
    if st.session_state.alpha_parent not in [None] + usable_ids:
        st.session_state.alpha_parent = None

    st.selectbox(
        "Alpha làm ngữ cảnh",
        [None] + usable_ids,
        format_func=lambda value: (
            "Chưa chọn alpha" if value is None else value
        ),
        key="alpha_parent",
    )

    st.caption(
        "Mỗi lượt nhận alpha đang chọn, số liệu tóm tắt và chỉ dẫn "
        "được ghim. Không gửi toàn bộ lịch sử chat."
    )

    if st.button("Bắt đầu hội thoại mới", disabled=st.session_state.alpha_busy):
        st.session_state.alpha_messages = []
        st.rerun()

    st.download_button(
        "Tải hội thoại và kết quả",
        data=json.dumps(
            st.session_state.alpha_messages,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        file_name="alpha_chat.json",
        mime="application/json",
    )

st.title("Alpha Research Chat")
st.caption(
    "Thảo luận kết quả hoặc tạo phiên bản alpha mới. "
    "Kết quả dev/validation mang tính thăm dò."
)

if not live:
    st.info("Recorded mode. Enable live research to send messages.")

parent_id = st.session_state.alpha_parent
if parent_id:
    selected = research_runs.load_run(ROOT, parent_id)
    with st.expander("Alpha đang được dùng làm ngữ cảnh", expanded=True):
        st.code(selected["result"]["formula"], language="text")

def render_message(item):
    with st.chat_message(item["role"]):
        st.markdown(item["text"])

        if item.get("turn_id"):
            st.caption("Turn: " + item["turn_id"])

        output = item.get("research_output")
        if output:
            st.code(output["formula"], language="text")
            st.write("Decision:", output["result"]["decision"]["status"])

            with st.expander("Kết quả thí nghiệm"):
                metrics = pd.DataFrame(output["result"]["metrics"])
                columns = [
                    "split", "strategy", "cost_bps",
                    "cagr", "net_sharpe",
                    "max_drawdown", "annual_turnover",
                ]
                st.dataframe(
                    metrics[
                        [c for c in columns if c in metrics.columns]
                    ],
                    hide_index=True,
                )
                st.caption(
                    "CAGR và drawdown ở dạng thập phân: 0.10 = 10%."
                )

for item in st.session_state.alpha_messages:
    render_message(item)

message = st.chat_input(
    "Hỏi về alpha hoặc yêu cầu sửa công thức…",
    max_chars=700,
    disabled=(not live or st.session_state.alpha_busy),
    key="alpha_chat_input",
)

if message:
    if not api_key.strip():
        st.error("Nhập Gemini API key ở sidebar trước.")
    elif api_key.strip() in message or api_key.strip() in pinned:
        st.error("Không đưa API key vào tin nhắn hoặc chỉ dẫn.")
    else:
        turn_id = uuid.uuid4().hex
        user_item = {
            "role": "user",
            "text": message,
            "mode": mode,
            "parent_request_id": parent_id,
            "turn_id": turn_id,
        }
        st.session_state.alpha_messages.append(user_item)
        st.session_state.alpha_busy = True

        # Persist pending turn before the API call; reruns do not replay it.
        st.session_state.alpha_pending = turn_id

        try:
            with st.spinner(
                "Đang phân tích…"
                if mode == "discuss"
                else "Đang tạo alpha, chạy thí nghiệm và phản biện…"
            ):
                answer = chat_service.run_turn(
                    root=ROOT,
                    turn_id=turn_id,
                    mode=mode,
                    message=message,
                    api_key=api_key,
                    parent_id=parent_id,
                    pinned=pinned,
                )

            assistant_item = {
                "role": "assistant",
                "text": answer["text"],
                "turn_id": turn_id,
            }
            if "research_output" in answer:
                assistant_item["research_output"] = answer["research_output"]

            st.session_state.alpha_messages.append(assistant_item)

            # Update the parent on the next rerun, before the widget exists.
            if answer["new_request_id"]:
                st.session_state.alpha_next_parent = answer["new_request_id"]

        except Exception as error:
            st.session_state.alpha_messages.append({
                "role": "assistant",
                "text": (
                    "Lượt này dừng do " + type(error).__name__ + ". "
                    "Không tự động chạy lại. Nếu lỗi ở Critic, "
                    "kết quả thí nghiệm có thể đã được lưu. "
                    "Kiểm tra request trước khi tạo lượt cải tiến khác."
                ),
                "turn_id": turn_id,
            })
        finally:
            st.session_state.alpha_busy = False

        st.rerun()
