import threading
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

def init_page():
    st.set_page_config(
        page_title="Mr.Yas Chatbot",
        page_icon="🤖"
    )
    st.header("Mr.Yas Chatbot 🤖")
    st.write("""Safari,Chrome,FirefoxなどWebブラウザのカメラ,マイクのアクセスを許可する設定にしてください。
         support.apple.com,support.google.com,support.mozilla.org参照。""") 
    

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    # clear_button が押された場合や message_history がまだ存在しない場合に初期化
    if clear_button or "message_history" not in st.session_state:
        st.session_state.message_history = [
            ("system", "You are a helpful assistant.")
        ] 
    

def select_model():
    # スライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.01とする
    #temperature = st.sidebar.slider(
        #"Temperature(回答バラツキ度合):", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    temperature = 0.0   
    models = ( "GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("Choose a model（大規模言語モデルを選択）:", models)
       
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
            #model=st.session_state.model_name,
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
def speak(text):
    # アスタリスクやその他の不要な文字を削除
    cleaned_text = re.sub(r'\*!?', '', text)
    
    # テキストを音声に変換
    tts = gTTS(text=cleaned_text, lang='ja')

    # 一時ファイルを作成
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        tts.save(temp_file.name)
        temp_file_path = temp_file.name

    # 音声ファイルの再生速度を変更
    audio = AudioSegment.from_file(temp_file_path)
    # 速度を1.5倍にする（2.0にすると2倍速）
    faster_audio = audio.speedup(playback_speed=1.5)

    # もう一つの一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as faster_temp_file:
        faster_audio.export(faster_temp_file.name, format="mp3")
        faster_temp_file_path = faster_temp_file.name

    # 音声ファイルを提供
    with open(faster_temp_file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3", start_time=0, autoplay=True)

    # 一時ファイルを削除
    os.remove(temp_file_path)
    os.remove(faster_temp_file_path)

#  LLM問答関数   
async def query_llm(user_input,frame):
    print("user_input=",user_input)
    
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
            
        elif st.session_state.model_name ==  "keep_command-r-plus":
            print("st.session_state.model_name=",st.session_state.model_name)
            prompt = ChatPromptTemplate.from_messages([
                    *st.session_state.message_history,
                    ("user", "{user_input}")  # ここにあとでユーザーの入力が入る
                ])
            output_parser = StrOutputParser()
            chain = prompt | st.session_state.llm | output_parser
            stream = chain.stream(user_input)
            
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
                stream = chain.stream({"user_input":user_input})
        # LLMの返答を表示する  Streaming
        #llm_output = st.empty()
        with st.chat_message('ai'):   #llm_output.chat_message('ai'):
            #st.write(response)  
            response =st.write_stream(stream) 
        #print("response=",response)            
        print(f"{st.session_state.model_name}=",response)
 
        # 音声出力処理                
        if st.session_state.output_method == "音声":
            #st.write("音声出力を開始します。")
            speak(response)   #st.audio ok
            #speak1(response) pygame NG
            #speak_thread = speak_async(response)
            # 必要に応じて音声合成の完了を待つ
            #speak_thread.join() 
            #print("音声再生が完了しました。次の処理を実行します。")
            #st.write("音声再生が完了しました。次の処理を実行できます。")
            
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

        ##########################################################
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
        if answer == "" :
            #print("テキスト出力が空")
            return None 
        elif "ご視聴" in answer or "お疲れ様" in answer:
            #print("テキスト出力が「ご視聴」、または「お疲れ様」を含む")
            return None 
        #else:
            #print("transcribeルーチンのtext(answer)=",answer)
            #st.session_state.text_output = answer
            #return answer
        
        ############################################################
    tmpfile.close()  
    os.remove(tmpfile.name)
    print("transcribeルーチンのtext(answer)=",answer)
    st.session_state.text_output = answer

    return answer
###############################################################    
def frame_energy(frame):
    samples = np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16)
    #################################################################
    # デバッグ用にサンプルの一部を出力 
    #print("Samples:", samples[:10])
    # NaNや無限大の値を除去 
    #if not np.isfinite(samples).all(): 
        #samples = samples[np.isfinite(samples)]
    #np.isfinite() で無効な値をフィルタリングするだけでは、
    # 空配列のエラーが再び発生する可能性があるため、
    # np.nan_to_num を使用したほうが安全に処理できます。
    # 無効な値を安全な値に置換
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    ##################################################################
    if len(samples) == 0: 
        return 0.0
    energy = np.sqrt(np.mean(samples**2)) 
    #print("Energy:", energy) 
    # エネルギーを出力 
    return energy
###########################################################################
def is_silent_frame(audio_frame, amp_threshold):
    """
    フレームが無音かどうかを最大振幅で判定する関数。
    """
    samples = np.frombuffer(audio_frame.to_ndarray().tobytes(), dtype=np.int16)
    max_amplitude = np.max(np.abs(samples))
    return max_amplitude < amp_threshold

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
        
        if energy < energy_threshold or is_silent_frame(audio_frame, amp_threshold):
            silence_frames += 1 
            #無音のエネルギー又は最大振幅がしきい値以下である場合、無音フレームの数を1つ増やします。
        else:
            silence_frames = 0 
            #エネルギー又は最大振幅がしきい値を超える場合、無音フレームをリセットして0にします。

    return sound_chunk, silence_frames

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
            text = transcribe(sound_chunk)
            text_output.write(text)
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
    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])
    #複数のスレッドが同時に共有リソースを操作することで、競合状態が発生します。
    #スレッドセーフにするには、リソースにアクセスする際に排他制御（mutexやlock）を使用します。
    #Pythonのqueue.Queueなどはスレッドセーフに設計されています。

    async def queued_audio_frames_callback(
        frames: List[av.AudioFrame],
    ) -> av.AudioFrame:
        with frames_deque_lock:
            frames_deque.extend(frames)
        # 受信したフレームをスレッドセーフに処理するため、
        # frames_deque_lockでロックをかけながら
        # frames_deque（キューのような構造）にフレームを蓄積します。
        # この処理により、別のスレッドでフレームを逐次取り出して音声認識処理を行うことができます。
        # オリジナルの音声フレームはスレッドセーフなframes_dequeに保存されます。

        # Return empty frames to be silent.
        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            #新しい無音フレームをav.AudioFrame.from_ndarrayを用いて生成し、
            # 同じレイアウトとサンプリングレートを設定します。
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)

        return new_frames
        #消音フレームの返却
        #処理後の無音化されたオーディオフレームをリストnew_framesに追加し、それを返却します。
        #消音フレームを返す理由は、オリジナルの音声をその場で出力しない（送信しない）ようにするためです。
        # 代わりに処理が進むバックエンドで音声認識を実行します。

    st.session_state.audio_receiver_size =2048
    #audio_receiver_size = 2048

    # サイドバーにWebRTCストリームを表示
    with st.sidebar:
        st.header("Webcam Stream")
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text and video",
            desired_playing_state=True, 
            mode=WebRtcMode.SENDRECV,  #SENDRECV, #.SENDONLY
            audio_receiver_size=st.session_state.audio_receiver_size, #2048, #audio_receiver_size,  #1024　#512 #デフォルトは4
            queued_audio_frames_callback=queued_audio_frames_callback,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": True},
            video_processor_factory=VideoTransformer,  #機能している？
            )
        
    st.sidebar.header("Capture Image")    
    cap_image = st.sidebar.empty() # プレースホルダーを作成 
    
    st.sidebar.title("Options")
    init_messages()
    status_indicator = st.empty()
    #usr_input = st.empty()
    text_output = st.empty()
    #llm_output = st.empty()
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
    #データ初期値
    user_input = ""
    base64_image = ""
    frame = ""  

    if not webrtc_ctx.state.playing:
        return
    #status_indicator.write("Loading...")

    ###################################################################
    #音声入力（テキストに変換した入力）の対話ループ
    #print("Before_st.session_state.input_method=",st.session_state.input_method)
    if st.session_state.input_method == "音声": 
        audio_receiver_size = st.sidebar.slider(
            "audio_receiver_size(処理音声フレーム数。デフォルト1024):", 
            min_value=64, max_value=2048, value=1024, step=64
            )
        st.session_state.audio_receiver_size = audio_receiver_size
        energy_threshold = st.sidebar.slider(
        "energy_threshold(無音エネルギーしきい値。デフォルト2000):", 
        min_value=100, max_value=5000, value=2000, step=100
        )
        amp_threshold = st.sidebar.slider(
            "amp_threshold(無音最大振幅しきい値。デフォルト0.3):", 
            min_value=0.00, max_value=1.00, value=0.30, step=0.05
            )
        # 無音を検出するための閾値 0.01 0.05 1.00以下
            #amp_threshold = 0.30  #0.05
        silence_frames_threshold = st.sidebar.slider(
            "silence_frames_threshold(トリガー用連続無音フレーム数。デフォルト100):", 
            min_value=20, max_value=300, value=60, step=20
            )
        #60がBest,デフォルト100
        #timeout = st.sidebar.slider(
            #"timeout(フレームを取得するためのタイムアウト。デフォルト3秒):", 
            #min_value=1, max_value=3, value=1, step=1
            #)
        #stで使う変数初期設定
        #st.session_state.energy_threshold = energy_threshold
        #st.session_state.amp_threshold = amp_threshold
        #st.session_state.silence_frames_threshold = silence_frames_threshold
        #st.session_state.timeout = timeout
      
        sound_chunk = pydub.AudioSegment.empty()
        #音声データを蓄積するためのpydub.AudioSegmentの空のオブジェクトとして初期化
        silence_frames = 0

        st.write("🤖何か話して!")
        status_indicator = st.empty() # プレースホルダーを作成
        status_indicator.write("Loading...")
        text_output = st.empty() # プレースホルダーを作成
        
        while True:
            #if webrtc_ctx.audio_receiver:
            if webrtc_ctx.state.playing:
                #timeout=st.session_state.timeout
                #energy_threshold=st.session_state.energy_threshold
                #amp_threshold=st.session_state.amp_threshold
                #silence_frames_threshold= st.session_state.silence_frames_threshold 
                #print("ここを通過A") #ここまでOK 

                #オリジナルの音声フレームはスレッドセーフなframes_dequeに保存されている。
                #frames_deque_lockを使用してスレッドセーフに操作しています。
                audio_frames = []
                with frames_deque_lock:
                    while len(frames_deque) > 0:
                        frame = frames_deque.popleft()
                        audio_frames.append(frame)
                #audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=timeout)
                    # キューを定期的にクリアtimeout
                #audio_frames = audio_frames 
                #except queue.Empty:
                # フレーム未到達時の処理
            # フレーム未到達時の処理
                if not audio_frames:
                    status_indicator.write("No frame arrived.")
                    sound_chunk = handle_queue_empty(sound_chunk, text_output)
                    continue
                #except Exception as e: 
                #    print(f"Error while clearing audio queue: {e}") 
                #    time.sleep(1) # 1秒ごとにクリア,必要に応じて調整

                #静寂のカウントやエネルギーしきい値判定    
                sound_chunk, silence_frames = process_audio_frames(audio_frames, sound_chunk, silence_frames, energy_threshold,amp_threshold)
                #一定の静寂フレームが検出された場合の処理（例: 音声認識のトリガー）を実行
                sound_chunk, silence_frames = handle_silence(sound_chunk, silence_frames, silence_frames_threshold, text_output)
                
            else:
                #ストリームが停止した（webrtc_ctx.state.playing == False）場合、
                # 蓄積された音声（sound_chunk）をテキストに変換して表示します。
                # transcribe関数で音声認識を行います。
                status_indicator.write("Stopping.")
                if len(sound_chunk) > 0:
                    #print("len(sound_chunk)=",len(sound_chunk))
                    try:
                        text = transcribe(sound_chunk.raw_data)
                        text_output.write(text)
                        print("else_Stoppingルーチンのtext=",text)
                    except Exception as e:
                        text_output.write("Error during transcription.")
                        print(f"Error during transcription: {e}")
                                        
                    print("ここを通過E2") #ここまでOK
                break
                       
            st.session_state.user_input=st.session_state.text_output
            if st.session_state.user_input != "":    
                print("user_input=",st.session_state.user_input)
            
                with st.chat_message('user'):   
                    st.write(st.session_state.user_input)

                cap = None 
                if st.session_state.input_img == "有":
                    # 画像と問い合わせ入力があったときの処理
                    #現在の画像をキャプチャする
                    #キャプチャー画像入力
                    if webrtc_ctx.video_transformer:  
                        cap = webrtc_ctx.video_transformer.frame
                    if cap is not None :
                        #st.sidebar.header("Capture Image")
                        cap_image.image(cap, channels="BGR")
                        # if st.button("Query LLM : 画像の内容を説明して"):
                
                with st.spinner("Querying LLM..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    st.session_state.result= ""
                    result = loop.run_until_complete(query_llm(st.session_state.user_input,cap))
                    st.session_state.result = result
                    st.write("🤖何か話して!")
                result = ""
                st.session_state.text_output=""
                st.session_state.user_input=""
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
            if st.button("日本の観光地を教えてください。"):
                button_input = "日本の観光地を教えてください。"
        with col8:
            if st.button("今日の料理はなにがいいかな"):
                button_input = "今日の料理はなにがいいかな"
        if button_input !="":
            st.session_state.user_input=button_input

        text_input =st.chat_input("🤗テキストで問い合わせる場合、ここに入力してね！") #,key=st.session_state.text_input)
        #text_input = st.text_input("テキストで問い合わせる場合、以下のフィールドに入力してください:", key=st.session_state.text_input) 
        if text_input:
            st.session_state.user_input=text_input
            text_input=""
            #llm_in()
        if st.session_state.user_input != "":    
            print("user_input=",st.session_state.user_input)
            with st.chat_message('user'):   
                st.write(st.session_state.user_input) 
            # 画像と問い合わせ入力があったときの処理
            cap = None 
            if st.session_state.input_img == "有":
                # 画像と問い合わせ入力があったときの処理
                #現在の画像をキャプチャする
                #キャプチャー画像入力
                if webrtc_ctx.video_transformer:  
                    cap = webrtc_ctx.video_transformer.frame
                if cap is not None :
                    #st.sidebar.header("Capture Image")
                    cap_image.image(cap, channels="BGR")
                    # if st.button("Query LLM : 画像の内容を説明して"):
                    
            # if st.button("Query LLM : 画像の内容を説明して"):
            with st.spinner("Querying LLM..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                st.session_state.result= ""
                result = loop.run_until_complete(query_llm(st.session_state.user_input,cap))
                st.session_state.result = result
                result = ""
                st.session_state.user_input=""

################################################################### 
def main():
    #st.header("Real Time Speech-to-Text with_video")
    #画面表示
    init_page()
   
     
    app_sst_with_video()
        #webcam_placeholder,
        #webrtc_ctx,
        #status_indicator,
        #text_output,
        #cap_image,
        #audio_receiver_size,
        #timeout,
        #energy_threshold,
        #amp_threshold,
        #silence_frames_threshold,
        #)  
###################################################################      
if __name__ == "__main__":
    main()
