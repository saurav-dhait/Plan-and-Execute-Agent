import streamlit as st
from normal_graph_maker import get_graph


def main():
    st.set_page_config(page_title="Plan-and-Execute Agent",
                       page_icon="🤖",
                       layout="centered",
                       initial_sidebar_state="expanded",
                       menu_items=None)
    # sidebar
    with st.sidebar:
        st.subheader("Chat options ")
        clear_chat = st.button("Clear chat", type="primary")
        st.subheader("")
        st.subheader("")
        st.image("assets/graph.jpeg",caption='Plan and execute graph')

    # main body
    st.title("📃 Plan-and-Execute Agent")
    if clear_chat:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hey, how can i help you ? "}]
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hey, how can i help you ? "}]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"],unsafe_allow_html=True)
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = "No response"
        with st.spinner(f"Generating response"):
            config = {"recursion_limit": 50}
            inputs = {"input": f"{prompt}"}
            graph = get_graph()
            for event in graph.stream(inputs, config=config):
                for k, v in event.items():
                    if k != "__end__":
                        response = v.popitem()
                        match response[0]:
                            case "plan":
                                styled_list = """
                                    <h4 style='margin:0;padding :0;'>📃Plan : </h4><br>
                                    <div style='font-family: Arial; font-size: 18px; color: #4CAF50;'>
                                        <ul>
                                """ + "".join(f"<li>{item}</li>" for item in response[1]) + """
                                        </ul>
                                    </div>
                                """
                                a = {"role": "assistant", "content": styled_list}
                                st.session_state.messages.append(a)
                                st.chat_message(a["role"]).markdown(a["content"],unsafe_allow_html=True)
                            case "past_steps":
                                styled_list = f"""
                                    <h4 style='margin:0;padding :0;'>🔍Step : </h4><br>
                                    <div style='font-family: Arial; font-size: 18px; color: #4CAF50;'>
                                        <ul>
                                            <h6>Instruction : </h6>
                                            <li>{response[1][0][0]}</li>
                                            <br>
                                            <h6>Output : </h6>
                                            <li>{response[1][0][1]}</li>
                                        </ul>
                                    </div>
                                """
                                a = {"role": "assistant", "content": styled_list}
                                st.session_state.messages.append(a)
                                st.chat_message(a["role"]).markdown(a["content"], unsafe_allow_html=True)
                            case "response":

                                styled_list = f"""<h4 style='margin:0;padding :0;'>✨Final Response : </h4><br>
                                    <div style='font-family: Arial; font-size: 18px; color: #4CAF50;'>
                                        <ul>
                                            <li>{response[1]}</li>                             
                                        </ul>
                                    </div>
                                """
                                a = {"role": "assistant", "content": styled_list}
                                st.session_state.messages.append(a)
                                st.chat_message(a["role"]).markdown(a["content"], unsafe_allow_html=True)


if __name__ == '__main__':
    main()
