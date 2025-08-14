from gtts import gTTS
import pygame
import time

# Text to speech
text = "Tes tes. Hai. Semua ini adalah sebuah tes dari Tuhan."
tts = gTTS(text=text, lang='id')
tts.save("output.mp3")

# Play the audio using pygame
pygame.mixer.init()
pygame.mixer.music.load("output.mp3")
pygame.mixer.music.play()

# Wait until it finishes playing
while pygame.mixer.music.get_busy():
    time.sleep(0.5)

