from vosk import Model, KaldiRecognizer
import pyaudio
import json
import os

# --- Configuration ---
MODEL_PATH = "tools/audio/vosk-model-small-es-0.42"
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096  # Optimized buffer size for lower latency

# --- Grammar for numbers ---
# keywords and esential numbers
grammar = [
    "subtotal", "total", "itbms", "venta", "impuesto",
    "[unk]", # unknown para cualquier otra cosa
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
    "punto", "coma", "dolares", "pesos", "balboas"
]
# concert python list to JSON (string)
grammar_json = json.dumps(grammar)

# --- map word numbers to digits ---
digit_map = {
    "cero": "0",
    "uno": "1",
    "dos": "2",
    "tres": "3",
    "cuatro": "4",
    "cinco": "5",
    "seis": "6",
    "siete": "7",
    "ocho": "8",
    "nueve": "9",
    "punto": ".",
    "coma": ","
}

# --- Setup and Initialization ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Vosk model not found at {MODEL_PATH}")
    print("Please download a Vosk model and update the MODEL_PATH variable.")
    exit()

try:
    # 1. Initialize Vosk model
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetGrammar(grammar_json)

    # 2. Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print("\n" + "="*50)
    print("LISTENING... Start speaking to get a transcription.")
    print("Press Ctrl+C to stop.")
    print("="*50)

    # 3. Main Streaming Loop
    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

        # Check for end of stream
        if not data:
            break

        # Process the audio chunk
        if recognizer.AcceptWaveform(data):
            # Final result is ready (user paused speaking)
            result = json.loads(recognizer.Result())
            final_text = result.get("text", "")

            if final_text:
                split_words = final_text.split()
                formatted_text = ""
                for word in split_words:
                    formatted_text += digit_map.get(word, word + " ") # Add space if not a digit/point

                print(f"FINAL TEXT: {formatted_text}")

        else:
            # Partial result is available (user is still speaking)
            partial_result = json.loads(recognizer.PartialResult())
            # print("Partial:", partial_result.get("partial", "")) # Uncomment for live feedback

except KeyboardInterrupt:
    print("\nStopping recognition.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    # 4. Clean up resources
    if 'stream' in locals() and stream.is_active():
        # Get any remaining text after the loop exits or is interrupted
        final_result = json.loads(recognizer.FinalResult())
        if final_result.get("text"):
             print(f"FINAL TEXT (Cleanup): {final_result['text']}")

        stream.stop_stream()
        stream.close()
    if 'p' in locals():
        p.terminate()
