import yt_dlp
import time
import re
import os
from pydub import AudioSegment
import speech_recognition as sr
import math
from tqdm import tqdm
from googletrans import Translator
from threading import Thread

# Example usage
youtube_url = 'https://www.youtube.com/watch?v=kW_XDUAjFVM'
credentials_path = '/Users/apoorvareddy/Downloads/Academic/DATS6501/compact-retina-447922-c6-2889a6e457b3.json'

ydl_opts={}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
video_title = info_dict['title']
video_name = re.sub('[\\\\/*?:"<>|]', '', video_title)
name = video_name
ydl_opts = {
     'format': 'm4a/bestaudio/best',
         'noplaylist': True,
         'continue_dl': True,
         'outtmpl': f'./{name}.wav',
         'postprocessors': [{
             'key': 'FFmpegExtractAudio',
             'preferredcodec': 'wav',
             'preferredquality': '192',
         }],
         'geobypass':True,
         'ffmpeg_location':'/usr/bin/ffmpeg'
 }

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
     error_code = ydl.download(youtube_url)

