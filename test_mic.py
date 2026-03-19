import speech_recognition as sr
print('Testing microphone...')
r = sr.Recognizer()
try:
    with sr.Microphone() as source:
        print('Microphone found! Adjusting for noise...')
        r.adjust_for_ambient_noise(source, duration=1)
        print('Listening for 5 seconds...')
        audio = r.listen(source, timeout=5)
        print('Got audio! Trying to recognize...')
        text = r.recognize_google(audio)
        print(f'You said: {text}')
except Exception as e:
    print(f'Error: {e}')
