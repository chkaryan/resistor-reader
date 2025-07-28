from gtts import gTTS
import os
import time

def speak_resistor_value(value_ohm):
    text = f"Nilai resistor adalah {value_ohm} Ohm"
    tts = gTTS(text=text, lang='id')  # 'id' = Indonesian
    filename = "resistor_output.mp3"
    tts.save(filename)

    # Play the sound
    os.system(f'start {filename}')  # Works on Windows

    # Optional: wait a bit then delete the file
    time.sleep(4)
    os.remove(filename)

# Example usage
speak_resistor_value("empat ratus tujuh puluh")  # Say: 470 Ohm
