import threading
import time
from collections import deque
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoTransformerBase
from typing import List,Tuple
import wave
import asyncio
import time
from time import sleep
import cv2
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere.chat_models import ChatCohere
from langchain_ollama import ChatOllama
import base64
from gtts import gTTS
import os
import whisper
import torch
import torchaudio
import torchvision
import re
import queue, pydub, tempfile
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
from io import BytesIO
import psutil
import gc
from scipy.signal import resample
import librosa
import subprocess

# 関数でメモリ使用量を取得
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    # メモリ使用量をMB単位で返す
    return mem_info.rss / (1024 * 1024)

def current_memory_use(i,memory_use,memory_alt,memory_ok):
    # 現在のメモリ使用量を取得
    current_memory_usage = get_memory_usage()
    # メモリ使用量を表示
    
    #memory_use.metric("現在のメモリ使用量 (MB)", f"{current_memory_usage:.2f}")
    memory_use.write(f"現在のメモリ使用量:\n\n    ループ{i}回目:{current_memory_usage:.0f}MB")
    #print("現在のメモリ使用量 (MB)", f"{current_memory_usage:.2f}")
    # メモリ制約を定義
    MEMORY_LIMIT_MB = 2700  # 1GB
    # メモリ使用量が制約を超えた場合の警告
    if current_memory_usage > MEMORY_LIMIT_MB:
      
        memory_alt.error(f"メモリ使用量が制約 ({MEMORY_LIMIT_MB} MB) を超えました。処理を中断してください。")
        memory_ok.empty()
        #st.stop()
        print(f"メモリ使用量が制約 ({MEMORY_LIMIT_MB} MB) を超えました。処理を中断してください。")
    else:
        
        memory_ok.success("メモリ使用量は正常範囲内です。")
        memory_alt.empty()
        #print("メモリ使用量は正常範囲内です。")

def select_model():
    # スライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.01とする
    temperature = 0.0
    #models = ( "GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    #model = st.sidebar.radio("大規模言語モデルを選択:", models)
    model = st.sidebar.selectbox(
        "LLM大規模言語モデルを選択",
        ["gemma2","phi4","llava-llama3","GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro"]
    )
    if model == "phi4":  
        st.session_state.model_name = "phi4"
        command = f"ollama pull {model} "
        subprocess.run(command, shell=True, capture_output=True, text=True, encoding='utf-8')
        return ChatOllama(
            temperature=temperature,
            model=st.session_state.model_name,
            #api_key= st.secrets.key.OPENAI_API_KEY,
            #streaming=True,
        )
    elif model == "gemma2":
        st.session_state.model_name = "gemma2"
        command = f"ollama pull {model} "
        subprocess.run(command, shell=True, capture_output=True, text=True, encoding='utf-8')
        return ChatOllama(
            temperature=temperature,
            model=st.session_state.model_name,
            #api_key= st.secrets.key.OPENAI_API_KEY,
            #streaming=True,
        )
    elif model == "llava-llama3":  
        st.session_state.model_name = "llava-llama3"
        command = f"ollama pull {model} "
        subprocess.run(command, shell=True, capture_output=True, text=True, encoding='utf-8')
        return ChatOllama(
            temperature=temperature,
            model=st.session_state.model_name,
            #api_key= st.secrets.key.OPENAI_API_KEY,
            #streaming=True,
        )            
    elif model == "GPT-4o":  #"gpt-4o 'gpt-4o-2024-08-06'" 有料？、Best
        st.session_state.model_name = "gpt-4o"
        return ChatOpenAI(
            temperature=temperature,
            model=st.session_state.model_name,
            api_key= st.secrets.key.OPENAI_API_KEY,
            max_tokens=12800,  #指定しないと短い回答になったり、途切れたりする。
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
async def streaming_text_speak(llm_response):
    # 末尾の空白の数を確認
    #trailing_spaces = len(llm_response) - len(llm_response.rstrip())
    #print(f"末尾の空白の数: {trailing_spaces}")
    # 末尾の空白を削除
    #cleaned_response = llm_response.rstrip()
    #print(f"空白を除去した文字列: '{cleaned_response}'")
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
                    cleaned_segment = re.sub(r'[\*#*!-]', '', segment)
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
                    if re.search(r"\n\n", segment):
                        print("文字列に '\\n\\n' が含まれています。")
                        #time.sleep(1) 
                    #else:
                        #print("文字列に '\\n\\n' は含まれていません。")
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

def trim_message_history(message_history, max_tokens=8192):  
    """  
    メッセージ履歴をトークン数で制限  
    GPT-4
    GPT-4: 8,192トークン
    GPT-4 Turbo: 128,000トークン
    GPT-4o: 128,000トークン
    Claude
    Claude 3 Haiku: 約200,000トークン
    Claude 3 Sonnet: 約200,000トークン
    Claude 3 Opus: 約200,000トークン
    Claude 2: 100,000トークン
    Gemini
    Gemini Pro: 32,000トークン
    Gemini Ultra: 最大1,000,000トークン
    Llama 2/3
    Llama 2 (7B-70B): 4,096トークン
    Llama 3 (8B): 8,192トークン
    Llama 3 (70B): 8,192トークン
    日本語モデル
    Rinna: 2,048トークン
    ELYZA: 4,096トークン
    Nekomata: 4,096トークン
    その他
    Command R+: 128,000トークン
    Mistral 7B: 8,192トークン
    Cohere: 4,096トークン
    推奨される一般的な戦略:

    安全サイズ: 4,000-8,000トークン
    トリミング関数の実装
    モデル固有の制限を確認

    """  
    total_tokens = 0  
    trimmed_history = []  
    
    # 最新のメッセージから逆順に追加  
    for message in reversed(message_history):  
        message_tokens = len(message[1])  # メッセージ長さを計算  
        if total_tokens + message_tokens <= max_tokens:  
            trimmed_history.insert(0, message)  
            total_tokens += message_tokens  
        else:  
            break  
    
    return trimmed_history  

#  LLM問答関数
async def query_llm(user_input,frame):
    if st.session_state.model_name ==  "llava-llama3":
        user_input = " 次の質問に日本語で答えてください。" + user_input 
    try:
        if st.session_state.input_img == "有":    
            # 画像を適切な形式に変換（例：base64エンコードなど）
            # 画像をエンコード
            encoded_image = cv2.imencode('.jpg', frame)[1]
            # 画像をBase64に変換
            base64_image = base64.b64encode(encoded_image).decode('utf-8')  
            #image = f"data:image/jpeg;base64,{base64_image}"
        
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
                        ("user", user_input), 
                        #(
                            #"user",
                            #[
                                #{
                                    #"type": "text",
                                    #"text": user_input
                                #}
                           # ],
                        #),
                    ]
                )
                
            output_parser = StrOutputParser()
            chain = prompt | st.session_state.llm | output_parser
            if st.session_state.output_method == "音声":
                response = chain.invoke({"user_input":user_input})
                #speak(response)   #st.audio ok
                await streaming_text_speak(response)
            else:    
                stream = chain.stream({"user_input":user_input})
            # LLMの返答を表示する  Streaming
                with st.chat_message('ai'):  
                    response =st.write_stream(stream) 
                           
            print(f"{st.session_state.model_name}=",response)
 
            # チャット履歴に追加
            st.session_state.message_history.append(("user", user_input))
            st.session_state.message_history.append(("ai", response))
            #多くのLLMには入力トークン数の制限がある
            #履歴が長すぎると、モデルが全コンテキストを処理できなくなる
            st.session_state.message_history = trim_message_history(st.session_state.message_history)
            return response
    except StopIteration:
        # StopIterationの処理
        print("StopIterationが発生")
        pass

    user_input = ""
    base64_image = ""
    frame = ""   

def qa(text_input,webrtc_ctx,cap_title,cap_image):
     # 末尾の空白の数を確認
    trailing_spaces = len(text_input) - len(text_input.rstrip())
    print(f"入力テキスト末尾の空白の数: {trailing_spaces}")
    # 末尾の空白を削除
    cleaned_text = text_input.rstrip()
    #print(f"入力テキスト末尾の空白を除去した文字列: '{cleaned_text}'")
    with st.chat_message('user'):   
        st.write(cleaned_text) 
    
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
            cap_title.header("Capture Image")     
            cap_image.image(cap, channels="BGR")
            # if st.button("Query LLM : 画像の内容を説明して"):
    # if st.button("Query LLM : 画像の内容を説明して"):
    with st.spinner("Querying LLM..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        st.session_state.result= ""
        result = loop.run_until_complete(query_llm(cleaned_text,cap))
        st.session_state.result = result
    result = ""
    text_input="" 

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):   
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

def whis_seg2(audio_segment):
    # AudioSegmentから直接NumPy配列を取得
    #audio_data = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    #audio_data /= np.iinfo(audio_segment.array_type).max  # 音声データを正規化
    audio_data = np.frombuffer(audio_segment.raw_data, dtype=np.int16).astype(np.float32)
    audio_data /= np.iinfo(np.int16).max  # 正規化
    # サンプリングレートを取得
    sample_rate = audio_segment.frame_rate
    #audio_segment = ""
    # Whisperが16kHzを期待するため、サンプリングレートを変換
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    # 音声データを適切な長さに調整
    audio_data = whisper.pad_or_trim(audio_data)
    if not "whisper_model" in st.session_state:
        st.session_state.whisper_model = whisper.load_model("small")
    whisper_model = st.session_state.whisper_model
    #whisper_model = whisper.load_model("small")
    result = whisper_model.transcribe(audio_data, language="ja")
    #audio_data = ""
    answer2 = result['text']
    #result = ""
    return answer2

def save_audio(audio_segment: AudioSegment, base_filename: str) -> None:
    filename = f"{base_filename}_{int(time.time())}.wav"
    audio_segment.export(filename, format="wav")

def transcribe(audio_segment: AudioSegment, debug: bool = False) ->  Tuple[str, str]:
    answer2 ="（休止中）"
    # ステレオの場合、モノラルに変換
    #print("audio_segment.channels=",audio_segment.channels)  
    if audio_segment.channels > 1:  
        audio_segment = audio_segment.set_channels(1)      
    
    if debug:
        save_audio(audio_segment, "debug_audio")
    #if st.session_state.output_whi2:
    answer2 = whis_seg2(audio_segment)
    return answer2 

async def process_audio(audio_data_bytes, sample_rate, sound_chunk):
    sound = pydub.AudioSegment(
        data=audio_data_bytes,
        sample_width=2, # audio_data_bytes.format.bytes,
        frame_rate=sample_rate,
        channels=2 , #len(audio_data_bytes.layout.channels), NG 1：文字化けする
    )
    sound_chunk += sound
    if len(sound_chunk) > 0:
       answer2 = transcribe(sound_chunk)
    return answer2 

async def process_audio_loop_with_silence_detection(
    frames_deque_lock,
    frames_deque,
    sound_chunk,
    amp_indicator,
    ):
    """
    音声を無音区切りでまとめ、無音が一定時間続いたらテキスト変換を行う。
    """
    audio_buffer = []
    last_sound_time = time.time()
    silence_detected = False

    while True:
        # フレームを取得
        with frames_deque_lock:
            while len(frames_deque) > 0:
                frame = frames_deque.popleft() # 左端から要素を取り出して削除
                audio_chunk = frame.to_ndarray().astype(np.int16)
                audio_buffer.append(audio_chunk)
                st.session_state.frame_sample_rate = frame.sample_rate
                amp=np.max(np.abs(audio_chunk)) 
                #st.session_state.amp = amp
                amp_indicator.write(f"音声振幅(無音閾値{st.session_state.amp_threshold})={amp}")
                #print(f"音声振幅/無音閾値={amp}/{SILENCE_THRESHOLD}")
                #amp_indicator.write(f"音声振幅(無音閾値{st.session_state.amp_threshold})={amp}")
                if not amp < st.session_state.amp_threshold:
                    last_sound_time = time.time()
                    silence_detected = False
                else:
                    silence_detected = time.time() - last_sound_time >= st.session_state.silence_threshold
        #print(f"無音判定={silence_detected}")
        #print(f"audio_buffer={len(audio_buffer)}")
        # 無音区切りが検出された場合、音声データを処理
        if silence_detected and audio_buffer:
            audio_data = np.concatenate(audio_buffer).tobytes()
            try:
                answer2 = await process_audio(audio_data, st.session_state.frame_sample_rate, sound_chunk)
                ##########################################################
                #text_output.write(f"認識結果: {answer}")
                #おかしな回答を除去
                # テキスト出力が空、または空白である場合もチェック
                phrases = (
                    "ありがとう", 
                    "お疲れ様", "んんんんんん", 
                    "by H.","スタッフさんのお話を",
                    "いいえ- いいえ- いいえ-",
                    "ごちそうさまでした"
                    )
                if len(answer2) < 5:
                    pass
                elif any(phrase in answer2 for phrase in phrases):
                    pass
                else:
                    #with text_output.chat_message('user'):
                        #st.write(answer2)
                    print("[Whis_seg]",answer2) 
                    return answer2 
                audio_buffer = []  # バッファをクリア
                silence_detected = False
            except Exception as e:
                st.error(f"音声認識エラー: {e}")
                continue
        #amp_indicator.write(f"音声振幅(無音閾値{st.session_state.amp_threshold})={st.session_state.amp}")
        #amp_indicator.write(f"音声振幅(無音閾値{st.session_state.amp_threshold})={amp}")
        # 処理負荷を抑えるために短い遅延を挿入
        time.sleep(0.1)


def app_sst_with_video():
    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])
    async def queued_audio_frames_callback(
        frames: List[av.AudioFrame],
    ) -> av.AudioFrame:
        with frames_deque_lock:
            frames_deque.extend(frames)
        # Return empty frames to be silent.
        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)
        return new_frames
        
     # ストリーミング状態を管理するセッション状態を初期化
    if "streaming" not in st.session_state:
        st.session_state["streaming"] = True  # 初期状態でストリーミング再生中
    # サイドバーにWebRTCストリームを表示
    with st.sidebar:
        st.header("Webcam Stream")
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text-w-video",
            desired_playing_state=st.session_state["streaming"], 
            mode=WebRtcMode.SENDRECV, #.SENDONLY,  #
            #audio_receiver_size=2048,  #1024　#512 #デフォルトは4
            #小さいとQueue overflow. Consider to set receiver size bigger. Current size is 1024.
            queued_audio_frames_callback=queued_audio_frames_callback,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": True},
            video_processor_factory=VideoTransformer,  
        )
    if not webrtc_ctx.state.playing:
        return
    #status_indicator.write("Loading...")
    cap_title = st.sidebar.empty()    
    cap_image = st.sidebar.empty() # プレースホルダーを作成 
    
    text_input = ""
    st.sidebar.title("Options")
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
    if st.session_state.input_method == "音声":
        st.session_state.message_history = [
            ("system", "You are a helpful assistant.")
        ]  # 会話履歴を初期化,画面クリア
        status_indicator = st.empty() # プレースホルダーを作成
        status_indicator.write("🤖準備中。しばらくお待ちください。")
        status_indicator = st.sidebar.empty()    
        amp_indicator = st.sidebar.empty()
        st.session_state.amp_threshold = st.sidebar.slider(
            "無音振幅閾値。デフォルト1000):",
            min_value=300, max_value=3000, value=1000, step=100
            )
        st.session_state.silence_threshold = st.sidebar.slider(
            "無音最小時間（デフォルト0.5秒）",
            min_value=0.1, max_value=3.0, value=0.5, step=0.1
            )
        if not "whisper_model" in st.session_state:
            st.session_state.whisper_model = whisper.load_model("small") #,device = "cuda")
        #base:74M,small:244M,medium:769M,large:1550M 
        #st.session_state["streaming"] = False  # Webカメラストリーミング停止
        #status_indicator.empty  #準備ができたので、"🤖準備中。しばらくお待ちください。")を消す NG

        memory_use = st.sidebar.empty()
        memory_alt = st.sidebar.empty()
        memory_ok = st.sidebar.empty()
        #for key, label, default in [
            #("output_whi2", "Whis-segm出力（無料）", True),
            #]:
            #st.session_state[key] = st.sidebar.toggle(label, value=default)

        i = 0
        #current_memory_use(i,memory_use,memory_alt,memory_ok)
        frames_deque_lock = threading.Lock()
        # frames_deque_lockを使用してスレッドセーフに音声フレームを処理していますが、
        # dequeのクリア操作などでリソース競合が起きる可能性があります。
        # dequeの最大長を設定（例: deque([], maxlen=100)) し、バッファ溢れを防止する方が安全です。
        frames_deque: deque = deque([], maxlen=100) #NG 1
        
        sound_chunk = pydub.AudioSegment.empty()  
        while True:
            # メモリ使用量を監視
            current_memory_use(i,memory_use,memory_alt,memory_ok)
            mem_use = get_memory_usage()
            i += 1
            #print(f"メモリ使用量={i}回目:{mem_use}")
            st.write("🤖何か話して!")
            status_indicator.write(f"len(frames_deque)={len(frames_deque)}")
            # 音声処理の非同期タスクを起動
            text_input = asyncio.run(process_audio_loop_with_silence_detection(
                frames_deque_lock,
                frames_deque,
                sound_chunk,
                amp_indicator,
                ))
            qa(text_input,webrtc_ctx,cap_title,cap_image)
            text_input = ""
    ################################################################### 
    # テキスト入力の場合
    # テキスト入力フォーム
    if st.session_state.input_method == "テキスト":
        st.session_state.message_history = [
            ("system", "You are a helpful assistant.")
        ]  # 会話履歴を初期化,画面クリア
        st.session_state["streaming"] = True  # Webカメラストリーミング再生
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
            if st.button("日本の悪いところは？"):
                button_input = "日本の悪いところは？"
        with col6:
            if st.button("善悪は何で決まりますか？"):
                button_input = "善悪は何で決まりますか？"
        with col7:
            if st.button("小松市のおいしい料理店は？"):
                button_input = "小松市のおいしい料理店は？"
        with col8:
            if st.button("Web画面でプレイするオセロのコードを作成して"):
                button_input = "Web画面でプレイするオセロのコードを作成して"
        if button_input !="":
            st.session_state.user_input=button_input

        text_input =st.chat_input("🤗テキストで問い合わせる場合、ここに入力してね！") #,key=st.session_state.text_input)
        #text_input = st.text_input("テキストで問い合わせる場合、以下のフィールドに入力してください:", key=st.session_state.text_input) 
        if button_input:
            text_input = button_input
        if text_input:
            qa(text_input,webrtc_ctx,cap_title,cap_image)


def init_page():
    st.set_page_config(
        page_title="Yas Chatbot",
        page_icon="🤖"
    )
    st.header("Yas Chatbot(画像、音声対応) 🤖")
    st.write("""Webカメラ画像についての問合せ、音声での入出力ができます。\n
             ブラウザのカメラ,マイクのアクセスを許可して使用。""") 
    
def init_messages():
    clear_button = st.sidebar.button("会話履歴クリア", key="clear")
    # clear_button が押された場合や message_history がまだ存在しない場合に初期化
    if clear_button or "message_history" not in st.session_state:
        st.session_state.message_history = [
            ("system", "You are a helpful assistant.")
        ]   

def main():
    #st.header("Real Time Speech-to-Text with_video")
    #画面表示
    init_page()
    init_messages()
    
    #stで使う変数初期設定
    st.session_state.llm = select_model()
    st.session_state.input_method = ""
    st.session_state.user_input = ""
    st.session_state.result = ""
    st.session_state.frame = "" 
    
    app_sst_with_video() 
     
###################################################################      
if __name__ == "__main__":
    main()
