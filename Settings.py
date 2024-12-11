import streamlit as st
import ollama
from time import sleep
from utilities.icon import page_icon

st.set_page_config(
    page_title="Model management",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    page_icon("⚙️")
    st.subheader("Model Management", divider="red", anchor=False)

    st.subheader("Download Models", anchor=False)
    model_name = st.text_input(
        "Enter the name of the model to download ↓", placeholder="mistral"
    )
    if st.button(f"📥 :green[**Download**] :red[{model_name}]"):
        if model_name:
            MAX_RETRIES = 3  # 最大リトライ回数

            for attempt in range(1, MAX_RETRIES + 1):  # 1回目から3回目まで試行
                with st.spinner(f"Attempting to download model: {model_name} (Attempt {attempt}/{MAX_RETRIES})..."):
                    try:
                        ollama.pull(model_name)  # モデルをプル
                        st.success(f"Downloaded model: {model_name}", icon="🎉")
                        st.balloons()
                        sleep(1)
                        st.rerun()  # 成功した場合、処理を終了
                    except Exception as e:
                        st.error(f"Attempt {attempt} failed to download model: {model_name}. Error: {str(e)}", icon="😳")
                        if attempt == MAX_RETRIES:  # 最後の試行で失敗した場合
                            st.error(f"Exceeded maximum retry attempts ({MAX_RETRIES}). Download failed.", icon="❌")
                        else:
                            sleep(2)  # 次の試行までの待機時間を設定
                    else:
                        print(f"Successfully downloaded model: {model_name}")
                        break  # 成功したらループを終了
        else:
            st.warning("Please enter a model name.", icon="⚠️")

    st.divider()

    st.subheader("Create model", anchor=False)
    modelfile = st.text_area(
        "Enter the modelfile",
        height=100,
        placeholder="""FROM mistral
SYSTEM You are mario from super mario bros.""",
    )
    model_name = st.text_input(
        "Enter the name of the model to create", placeholder="mario"
    )
    if st.button(f"🆕 Create Model {model_name}"):
        if model_name and modelfile:
            try:
                ollama.create(model=model_name, modelfile=modelfile)
                st.success(f"Created model: {model_name}", icon="✅")
                st.balloons()
                sleep(1)
                st.rerun()
            except Exception as e:
                st.error(
                    f"""Failed to create model: {
                         model_name}. Error: {str(e)}""",
                    icon="😳",
                )
        else:
            st.warning("Please enter a **model name** and **modelfile**", icon="⚠️")

    st.divider()

    st.subheader("Delete Models", anchor=False)
    models_info = ollama.list()
    available_models = [m["name"] for m in models_info["models"]]

    if available_models:
        selected_models = st.multiselect("Select models to delete", available_models)
        if st.button("🗑️ :red[**Delete Selected Model(s)**]"):
            for model in selected_models:
                try:
                    ollama.delete(model)
                    st.success(f"Deleted model: {model}", icon="🎉")
                    st.balloons()
                    sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(
                        f"""Failed to delete model: {
                        model}. Error: {str(e)}""",
                        icon="😳",
                    )
    else:
        st.info("No models available for deletion.", icon="🦗")


if __name__ == "__main__":
    main()
