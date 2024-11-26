import time
from collections import deque
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoTransformerBase
from pydub import AudioSegment
import queue, pydub, tempfile
import whisper
import torch
import torchaudio
import torchvision
from typing import List
import wave
import asyncio
import time
import cv2
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere.chat_models import ChatCohere
import base64
from gtts import gTTS
import os
import re
from pydub.effects import low_pass_filter, high_pass_filter
from io import BytesIO

def init_page():
    st.set_page_config(
        page_title="Mr.Yas Chatbot(Webカメの画像、音声を表示)",
        page_icon="🤖"
    )
    st.header("Mr.Yas Chatbot 🤖")
    st.write("""Webカメラに移した画像についての問合せ、音声での入出力ができます。\n
             Webブラウザのカメラ,マイクのアクセスを許可する設定にしてください。""") 

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    # clear_button が押された場合や message_history がまだ存在しない場合に初期化
    if clear_button or "message_history" not in st.session_state:
        st.session_state.message_history = [
            ("system", "You are a helpful assistant.")
        ] 
 
def select_model():
    #temperature = st.sidebar.slider(
        #"Temperature(回答バラツキ度合):", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    temperature = 0.0   
    models = ( "GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("大規模言語モデル選択）:", models)
       
    if model == "GPT-4o":  #"gpt-4o 'gpt-4o-2024-08-06'" 有料？、Best
        st.session_state.model_name = "gpt-4o"
        return ChatOpenAI(
            temperature=temperature,
            model=st.session_state.model_name,
            api_key= st.secrets.key.OPENAI_API_KEY,
            max_tokens=512,  #指定しないと短い回答になったり、途切れたりする。
            streaming=True,
        )
    elif model == "Claude 3.5 Sonnet": #コードがGood！！
        st.session_state.model_name = "claude-3-5-sonnet-20240620"
        return ChatAnthropic(
            temperature=temperature,
            model_name=st.session_state.model_name, 
            api_key= st.secrets.key.ANTHROPIC_API_KEY,
            max_tokens_to_sample=2048,  
            timeout=None,  
            max_retries=2,
            stop=None,  
        )
    elif model == "Gemini 1.5 Pro":
        st.session_state.model_name = "gemini-1.5-pro-latest"
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model=st.session_state.model_name,
            api_key= st.secrets.key.GOOGLE_API_KEY,
        )
#音声出力関数
def streaming_text_speak(llm_response):
    # 末尾の空白の数を確認
    trailing_spaces = len(llm_response) - len(llm_response.rstrip())
    print(f"末尾の空白の数: {trailing_spaces}")
    # 末尾の空白を削除
    cleaned_response = llm_response.rstrip()
    print(f"空白を除去した文字列: '{cleaned_response}'")
    # 句読点やスペースを基準に分割
    #復帰文字（\r）は、**キャリッジリターン（Carriage Return）**と呼ばれる特殊文字で、
    # ASCIIコード13（10進数）に対応します。主に改行の一部として使用される制御文字です。
    split_response = re.split(r'([\r\n!-;=:、。 \?]+)', llm_response) 
    #split_response = re.split(r'([;:、。 ]+😊🌟🚀🎉)', llm_response)  #?はなくてもOK
    split_response = [segment for segment in split_response if segment.strip()]  # 空要素を削除
    print(split_response)
    # AIメッセージ表示
    with st.chat_message("ai"):
        response_placeholder = st.empty()
        # ストリーミング応答と音声出力処理
        partial_text = ""
        
        for segment in split_response:
            if segment.strip():  # 空文字列でない場合のみ処理
                partial_text += segment
                response_placeholder.markdown(f"**{partial_text}**")  # 応答のストリーミング表示
                # gTTSで音声生成（部分テキスト）
                try:
                    # アスタリスクやその他の発音に不要な文字を削除
                    cleaned_segment = re.sub(r'[\*!-]', '', segment)
                    tts = gTTS(cleaned_segment, lang="ja")  # 音声化
                    audio_buffer = BytesIO()
                    tts.write_to_fp(audio_buffer)  # バッファに書き込み
                    audio_buffer.seek(0)

                    # pydubで再生速度を変更
                    audio = AudioSegment.from_file(audio_buffer, format="mp3")
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate * 1.3)  # 1.5倍速
                    }).set_frame_rate(audio.frame_rate)
                    audio_buffer.close()

                    # 音質調整
                    audio = audio.set_frame_rate(44100)  # サンプリングレート
                    audio = audio + 5  # 音量を5dB増加
                    audio = audio.fade_in(500).fade_out(500)  # フェードイン・アウト
                    #audio = audio.low_pass_filter(3000)  # 高音域をカット
                    audio = low_pass_filter(audio, cutoff=900)  # 高音域をカット
                    # ベースブースト（低音域を強調）
                    low_boost = low_pass_filter(audio,1000).apply_gain(10)
                    audio = audio.overlay(low_boost)

                    # バッファに再エクスポート
                    output_buffer = BytesIO()
                    audio.export(output_buffer, format="mp3")
                    output_buffer.seek(0)

                    # 音声の再生
                    # チェックする文字列
                    #patterns = ["\n\n1.", "\n\n2.","\n\n3.", "\n\n4.""\n\n5.", "\n\n6.", "\n\n7.","\n\n8.", "\n\n9.""\n\n10.", "\n\n11."]
                    if re.search(r"\n\n", segment):
                        print("文字列に '\\n\\n' が含まれています。")
                        time.sleep(2) 
                    else:
                        print("文字列に '\\n\\n' は含まれていません。")
                    #st.audio(audio_buffer, format="audio/mp3",autoplay = True)
                    # 音声データをBase64にエンコード
                    audio_base64 = base64.b64encode(output_buffer.read()).decode()
                    audio_buffer.close()  # バッファをクローズ
                    a=len(audio_base64)
                    #print(a)
                     # HTMLタグで音声を自動再生（プレイヤー非表示、再生速度調整）
                    audio_html = f"""
                   <audio id="audio-player" autoplay style="display:none;">
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)

                except Exception as e:
                    #print(f"音声生成エラー: {e}")
                    pass
                try:
                    time.sleep(a*0.00004)  # テキストストリーミング速度に同期
                except Exception as e:
                  time.sleep(2) 


#  LLM問答関数   
async def query_llm(user_input,frame):
    try:
        if st.session_state.input_img == "有":    
            # 画像を適切な形式に変換（例：base64エンコードなど）
            # 画像をエンコード
            encoded_image = cv2.imencode('.jpg', frame)[1]
            # 画像をBase64に変換
            base64_image = base64.b64encode(encoded_image).decode('utf-8')  
            #image = f"data:image/jpeg;base64,{base64_image}"
        
        if st.session_state.model_name ==  "keep_gpt-4o":
            llm = st.session_state.llm  
            stream = llm.stream([
                    *st.session_state.message_history,
                    (
                        "user",
                        [
                            {
                                "type": "text",
                                "text": user_input
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "auto"
                                },
                            }
                        ]
                    )
                ])
            #response = chain.invoke(user_input)
         
        if st.session_state.model_name ==  "command-r-plus":
            print("st.session_state.model_name=",st.session_state.model_name)
            print(user_input)
            prompt = ChatPromptTemplate.from_messages(
                [
                    *st.session_state.message_history,
                     #("user", f"{user_input}:{base64_image}"),  #やっぱりだめ
                     ("user", f"{user_input}")
                ]
            )
            
            output_parser = StrOutputParser()
            chain = prompt | st.session_state.llm | output_parser
            #stream = chain.stream(user_input,base64_image)
            
            stream = chain.stream({"user_input":user_input,"base64_image": base64_image})
            print("stream=",stream)
            #response = chain.invoke(user_input)
            
        else:
            print("st.session_state.model_name=",st.session_state.model_name)
            if st.session_state.input_img == "有":
                prompt = ChatPromptTemplate.from_messages(
                    [
                        *st.session_state.message_history,
                        (
                            "user",
                            [
                                {
                                    "type": "text",
                                    "text": user_input
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                }
                            ],
                        ),
                    ]
                )
              
                output_parser = StrOutputParser()
                chain = prompt | st.session_state.llm | output_parser
                #stream = chain.stream(user_input,base64_image)
                stream = chain.stream({"user_input":user_input,"base64_image": base64_image})
                #print("stream=",stream)
                #response = chain.invoke(user_input)
            else:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        *st.session_state.message_history,
                        (
                            "user",
                            [
                                {
                                    "type": "text",
                                    "text": user_input
                                }
                            ],
                        ),
                    ]
                )
                
            output_parser = StrOutputParser()
            chain = prompt | st.session_state.llm | output_parser
            if st.session_state.output_method == "音声":
                response = chain.invoke({"user_input":user_input})
                #speak(response)   #st.audio ok
                streaming_text_speak(response)
            else:    
                stream = chain.stream({"user_input":user_input})
            # LLMの返答を表示する  Streaming
                with st.chat_message('ai'):  
                    response =st.write_stream(stream) 
                           
            print(f"{st.session_state.model_name}=",response)
 
            # チャット履歴に追加
            st.session_state.message_history.append(("user", user_input))
            st.session_state.message_history.append(("ai", response))
        
            return response
    except StopIteration:
        # StopIterationの処理
        print("StopIterationが発生")
        pass

    user_input = ""
    base64_image = ""
    frame = ""   

class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame = None

        def recv(self, frame):   
            self.frame = frame.to_ndarray(format="bgr24")
            return frame
async def process_audio(audio_data_bytes, sample_rate):
    #with wave.open(audio_data_io, 'wb') as wf:
    #with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:    
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(temp_audio_file, 'wb') as wf:
        wf.setnchannels(2)
        #wf.setnchannels(1)  # モノラル音声として記録　NG
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data_bytes)
    # 一時ファイルのパスを transcribe に渡す
    temp_audio_file_path = temp_audio_file.name 
    temp_audio_file.close()  
    # Whisperのモデルをロード
    model = whisper.load_model("small")  # モデルのサイズは適宜選択
    #base:74M,small:244M,medium,large
    # 音声をデコード
    try:
        # Whisperで音声をテキストに変換
        result = model.transcribe(temp_audio_file_path, language="ja")  # 日本語指定
        answer = result['text']
    finally:
        # 一時ファイルを削除
        os.remove(temp_audio_file_path)
    
        
    # テキスト出力が空、または空白である場合もチェック
    if answer == "" :
        print("テキスト出力が空")
        #return None 
    elif "ご視聴" in answer or "お疲れ様" in answer:
        print("テキスト出力が「ご視聴」、または「お疲れ様」を含む")
        #return None 
    else:
        print(answer)
        return answer
########################################################################### 
def save_audio(audio_segment: AudioSegment, base_filename: str) -> None:
    filename = f"{base_filename}_{int(time.time())}.wav"
    audio_segment.export(filename, format="wav")

def transcribe(audio_segment: AudioSegment, debug: bool = False) -> str:
    """
    OpenAIのWhisper ASRシステムを使用して音声セグメントを文字起こしします。
    引数:
        audio_segment (AudioSegment): 文字起こしする音声セグメント。
        debug (bool): Trueの場合、デバッグ目的で音声セグメントを保存します。
    戻り値:
        str: 文字起こしされたテキスト。
    """
    if debug:
        save_audio(audio_segment, "debug_audio")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        audio_segment.export(tmpfile.name, format="wav")
        # 一時ファイルのパスを指定
        audio = whisper.load_audio(tmpfile.name)
        audio = whisper.pad_or_trim(audio)
        # Whisperのモデルをロード
        model = whisper.load_model("small")  # モデルのサイズは適宜選択
        #base:74M,small:244M,medium,large
        # 音声をデコード
        result = model.transcribe(audio, language="ja")  # 日本語を指定
        answer = result['text']
        
        # テキスト出力が空、または空白である場合もチェック
        if len(answer) < 5 or "ご視聴" in answer or "お疲れ様" in answer:
            #print("テキスト出力が空")
            #print("transcribeルーチンのtext(answer)=",answer)
            return None
        #elif "ご視聴" in answer or "お疲れ様" in answer:
            #print("テキスト出力が「ご視聴」、または「お疲れ様」を含む")
            #return None 
        
    tmpfile.close()  
    os.remove(tmpfile.name)
    print("transcribeルーチンのtext(answer)=",answer)
    st.session_state.text_output = answer
    return answer
###############################################################    
def frame_energy(frame):
    # フレームのデータをnumpy配列として読み込み
    samples = np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16)
    # NaN、正の無限大、負の無限大を0に置換
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    # 配列の長さが0の場合はエネルギーを0として返す
    if len(samples) == 0: 
        return 0.0
    # 負の値を絶対値に変換して処理 
    samples = np.abs(samples)
    try:
        #print(np.mean(samples**2))
        energy = np.sqrt(np.mean(samples**2))
        #print("energy=",energy)  #50-90
        return energy  #if not np.isnan(energy) else 0.0 これ付加するとだめ音声が途切れる
    except Exception as e:
        #print(f"Error exporting audio: {e}")
        return 0.0
        
def frame_amplitude(audio_frame):
    samples = np.frombuffer(audio_frame.to_ndarray().tobytes(), dtype=np.int16)
    max_amplitude = np.max(np.abs(samples))
    #print("max_amplitude=",max_amplitude)
    return max_amplitude 

def process_audio_frames(audio_frames, sound_chunk, silence_frames, energy_threshold, amp_threshold):
    """
    音声フレームを順次処理し、無音フレームの数をカウントすることです。
    無音フレームが一定数以上続いた場合、無音区間として処理し、後続の処理（例えば、音声認識のトリガー）に役立てます。
    この処理により、無音や音声の有無を正確に検出することができます。

    音声フレームのリストを処理します。 
    引数：
        audio_frames (list[VideoTransformerBase.Frame]): 処理する音声フレームのリスト。
        sound_chunk (AudioSegment): 現在のサウンドチャンク。
        silence_frames (int): 現在の無音フレームの数。
        energy_threshold (int): 無音検出に使用するエネルギーしきい値。
        amp_threshold:無音検出に使用する最大振幅しきい値。
    戻り値：
        tuple[AudioSegment, int]: 更新されたサウンドチャンクと無音フレームの数。
        
    """
        
    for audio_frame in audio_frames:
        sound_chunk = add_frame_to_chunk(audio_frame, sound_chunk)

        energy = frame_energy(audio_frame)
        amplitude = frame_amplitude(audio_frame)

        if energy < energy_threshold or amplitude < amp_threshold:
            silence_frames += 1 
            #無音のエネルギー又は最大振幅がしきい値以下である場合、無音フレームの数を1つ増やします。
        else:
            silence_frames = 0 
            #エネルギー又は最大振幅がしきい値を超える場合、無音フレームをリセットして0にします。

    return sound_chunk, silence_frames,energy,amplitude

def add_frame_to_chunk(audio_frame, sound_chunk):
    """
    オーディオフレームをサウンドチャンクに追加します。 
    引数：
        audio_frame (VideoTransformerBase.Frame): 追加するオーディオフレーム。
        sound_chunk (AudioSegment): 現在のサウンドチャンク。
    戻り値：
        AudioSegment: 更新されたサウンドチャンク。
   
    """
    sound = pydub.AudioSegment(
        data=audio_frame.to_ndarray().tobytes(),
        sample_width=audio_frame.format.bytes,
        frame_rate=audio_frame.sample_rate,
        channels=len(audio_frame.layout.channels),
    )
    sound_chunk += sound
    return sound_chunk

def handle_silence(sound_chunk, silence_frames, silence_frames_threshold, text_output):
    """
    オーディオストリーム内の無音を処理します。 
    引数：
        sound_chunk (AudioSegment): 現在のサウンドチャンク。
        silence_frames (int): 現在の無音フレームの数。
        silence_frames_threshold (int): 無音フレームのしきい値。
        text_output (st.empty): Streamlitのテキスト出力オブジェクト。
    戻り値：
        tuple[AudioSegment, int]: 更新されたサウンドチャンクと無音フレームの数。
   
    """
    if silence_frames >= silence_frames_threshold: 
        #無音フレーム数が100以上の時、音声の途切れ（間隔）として扱う
        if len(sound_chunk) > 0:
            #無音フレーム数が連続したら、音声の途切れとして、そこまでの音声データをテキストに変換している
            text = transcribe(sound_chunk)
            #text_output.write(text)
            #print("handle_silenceルーチンのtext=",text)
            #print("オーディオストリーム内の無音時の応答=",text)

            sound_chunk = pydub.AudioSegment.empty()
            silence_frames = 0

    return sound_chunk, silence_frames

def handle_queue_empty(sound_chunk, text_output):
    """
    オーディオフレームキューが空の場合の処理を行います。
    引数:
        sound_chunk (AudioSegment): 現在のサウンドチャンク。
        text_output (st.empty): Streamlitのテキスト出力オブジェクト。
    戻り値:
        AudioSegment: 更新されたサウンドチャンク。
    """
    if len(sound_chunk) > 0:
        text = transcribe(sound_chunk)
        text_output.write(text)
        #print("handle_queue_emptyルーチンのtext=",text)
        #st.session_state.text_output = text
        sound_chunk = pydub.AudioSegment.empty()

    return sound_chunk

def app_sst_with_video():
    """
    リアルタイム音声認識のための主なアプリケーション機能。
        この機能は、WebRTCストリーマーを作成し、音声データの受信を開始し、音声フレームを処理し、
        一定の閾値を超える静寂が続いたときに音声をテキストに文字起こしします。
    引数:
        audio_receiver_size:処理音声フレーム数。デフォルト512
            小さいとQueue overflow. Consider to set receiver size bigger. Current size is 1024.
        status_indicator: ステータス（実行中または停止中）を表示するためのStreamlitオブジェクト。
        text_output: 文字起こしされたテキストを表示するためのStreamlitオブジェクト。
        timeout (int, オプション): オーディオ受信機からフレームを取得するためのタイムアウト。デフォルトは3秒。
        energy_threshold (int, オプション): フレームが静寂と見なされるエネルギーの閾値。デフォルトは2000。
        silence_frames_threshold (int, オプション): 文字起こしをトリガーするための連続する静寂フレームの数。デフォルトは100フレーム。
    """
    text_input = ""
   
    st.session_state.audio_receiver_size =4096 #2048
    # サイドバーにWebRTCストリームを表示
    #with st.sidebar:
    st.header("Webcam Stream")
    webrtc_ctx1 = webrtc_streamer(
        key="example",
        desired_playing_state=True, 
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoTransformer,
        )
    #st.sidebar.header("Capture Image") 
    cap_title = st.sidebar.empty()    
    cap_image = st.sidebar.empty() # プレースホルダーを作成 
    status_indicator = st.sidebar.empty() # プレースホルダーを作成
    st.sidebar.title("Options")
    init_messages()
    text_output = st.empty()
    #stで使う変数初期設定
    st.session_state.llm = select_model()
    st.session_state.input_method = ""
    st.session_state.user_input = ""
    st.session_state.result = ""
    st.session_state.frame = "" 
    st.session_state.text_output = ""

    col1, col2 ,col3= st.sidebar.columns(3)
     # 各列にボタンを配置
    with col1:
        # 入力方法の選択
        input_method = st.sidebar.radio("入力方法", ("テキスト", "音声"))
        st.session_state.input_method = input_method
    with col2:
        # 画像についての問合せ有無の選択
        input_img = st.sidebar.radio("  カメラ画像問合せ", ("有", "無"))
        st.session_state.input_img = input_img
    with col3:
        # 出力方法の選択
        output_method = st.sidebar.radio("出力方法", ("テキスト", "音声"))
        st.session_state.output_method = output_method
    
    # チャット履歴の表示 
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)
      
    ###################################################################
    #音声入力（テキストに変換した入力）の対話ループ
    #print("Before_st.session_state.input_method=",st.session_state.input_method)
    if st.session_state.input_method == "音声": 
        st.write("🤖何か話して!")
        #status_indicator.write("音声認識動作中...")
        
        text_output = st.empty() # プレースホルダーを作成

        audio_receiver_size = st.sidebar.slider(
        "audio_receiver_size(音声受信容量。デフォルト4096):", 
        min_value=512, max_value=4096, value=4096, step=512
        )
        # 無音を検出するための閾値    
        energy_threshold = st.sidebar.slider(
        "energy_threshold(無音最大エネルギー。デフォルト300):", 
        min_value=10, max_value=600, value=300, step=50
        )
        amp_threshold = st.sidebar.slider(
            "amp_threshold(無音最大振幅。デフォルト600):", 
            min_value=0, max_value=1200, value=600, step=50
            )
        silence_frames_threshold = st.sidebar.slider(
            "silence_frames_threshold(連続無音区間（音声途切れフレーム数）。デフォルト100):", 
            min_value=0, max_value=200, value=100, step=10
            )
        #60がBest,デフォルト100
        timeout = st.sidebar.slider(
            "timeout(フレームを取得するためのタイムアウト。デフォルト3秒):", 
            min_value=1, max_value=3, value=1, step=1
            )
        
        with st.sidebar:
            webrtc_ctx = webrtc_streamer(
                key="speech-to-text and video",
                desired_playing_state=True, 
                mode=WebRtcMode.SENDONLY,  #SENDRECV, #.
                audio_receiver_size=st.session_state.audio_receiver_size, #audio_receiver_size,  #1024　#512 #デフォルトは4
                #queued_audio_frames_callback=queued_audio_frames_callback,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": False, "audio": True},
                video_processor_factory=VideoTransformer,
                )
            
        if not webrtc_ctx.state.playing:
            return
        #stで使う変数初期設定
        st.session_state.energy_threshold = energy_threshold
        st.session_state.amp_threshold = amp_threshold
        st.session_state.silence_frames_threshold = silence_frames_threshold
        st.session_state.timeout = timeout

        sound_chunk = pydub.AudioSegment.empty()
        silence_frames = 0

        
        while True:
            if webrtc_ctx.audio_receiver:
                timeout=st.session_state.timeout
                energy_threshold=st.session_state.energy_threshold
                amp_threshold=st.session_state.amp_threshold
                silence_frames_threshold= st.session_state.silence_frames_threshold    

                try:
                    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=timeout)
                  
                except queue.Empty:
                    status_indicator.write("No frame arrived.")
                    sound_chunk = handle_queue_empty(sound_chunk, text_output)
                    continue
                sound_chunk, silence_frames ,energy,amplitude= process_audio_frames(audio_frames, sound_chunk, silence_frames, energy_threshold,amp_threshold)
                sound_chunk, silence_frames = handle_silence(sound_chunk, silence_frames, silence_frames_threshold, text_output)
                
                try:
                    energy = 0.0 if np.isnan(energy) else energy
                    energy = round(energy)

                except Exception as e:
                    #print(f"Error exporting round(energy): {e}")
                    energy = 0
                status_indicator.write(f"音声レベル:\n エネルギー={energy}/threshold={energy_threshold},\n 最大振幅={amplitude}/threshold={amp_threshold}")
            else:    
                status_indicator.write("音声認識停止")

            if len(st.session_state.text_output) > 4 :
                print("st.session_state.text_output=",st.session_state.text_output)    
                text_input =  st.session_state.text_output 
                st.session_state.text_output = ""
            #これ以降は、音声入力、テキスト入力共通の処理へ
            if text_input: 
                qa(text_input,webrtc_ctx1,cap_title,cap_image)
                st.write(f"🤖何か話して!")  
                text_input = ""
                
    ################################################################### 
    # テキスト入力の場合
    # テキスト入力フォーム
    if st.session_state.input_method == "テキスト":
        button_input = ""
        # 4つの列を作成
        col1, col2, col3, col4 = st.columns(4)
        # 各列にボタンを配置
        with col1:
            if st.button("画像の内容を説明して"):
                button_input = "画像の内容を説明して"
        with col2:
            if st.button("前の画像と何が変わりましたか？"):
                button_input = "前の画像と何が変わりましたか？"
        with col3:
            if st.button("この画像の文を翻訳して"):
                button_input = "この画像の文を翻訳して"
        with col4:
            if st.button("人生の意義は？"):
                button_input = "人生の意義？"
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            if st.button("日本語に翻訳してください。"):
                button_input = "日本語に翻訳してください。"
        with col6:
            if st.button("善悪は何で決まりますか？"):
                button_input = "善悪は何で決まりますか？"
        with col7:
            if st.button("小松市の観光地を教えてください。"):
                button_input = "小松市の観光地を教えてください。"
        with col8:
            if st.button("今日の料理はなにがいいかな"):
                button_input = "今日の料理はなにがいいかな"
        #if button_input !="":
            #st.session_state.user_input=button_input

        text_input =st.chat_input("🤗テキストで問い合わせる場合、ここに入力してね！") #,key=st.session_state.text_input)
        #text_input = st.text_input("テキストで問い合わせる場合、以下のフィールドに入力してください:", key=st.session_state.text_input) 
        if button_input:
            text_input = button_input
        if text_input:
            qa(text_input,webrtc_ctx1,cap_title,cap_image)
    ###################################################################################
def qa(text_input,webrtc_ctx1,cap_title,cap_image):
     # 末尾の空白の数を確認
    trailing_spaces = len(text_input) - len(text_input.rstrip())
    print(f"入力テキスト末尾の空白の数: {trailing_spaces}")
    # 末尾の空白を削除
    cleaned_text = text_input.rstrip()
    print(f"入力テキスト末尾の空白を除去した文字列: '{cleaned_text}'")
    with st.chat_message('user'):   
        st.write(cleaned_text) 
    # 画像と問い合わせ入力があったときの処理
    cap = None 
    if st.session_state.input_img == "有":
        # 画像と問い合わせ入力があったときの処理
        #現在の画像をキャプチャする
        #キャプチャー画像入力
        if webrtc_ctx1.video_transformer: 
            cap = webrtc_ctx1.video_transformer.frame
        if cap is not None :
            #st.sidebar.header("Capture Image") 
            cap_title.header("Capture Image")     
            cap_image.image(cap, channels="BGR")
        else:
            st.warning("WebRTCストリームがまだ初期化されていません。")
            
    # if st.button("Query LLM : 画像の内容を説明して"):
    with st.spinner("Querying LLM..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        st.session_state.result= ""
        result = loop.run_until_complete(query_llm(cleaned_text,cap))
        st.session_state.result = result
    result = ""
    text_input=""

################################################################### 
def main():
    #画面表示
    init_page()
    app_sst_with_video()  
###################################################################      
if __name__ == "__main__":
    main()
