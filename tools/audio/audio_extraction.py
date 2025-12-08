from vosk import Model, KaldiRecognizer
import pyaudio
import json
import os
from typing import Dict, List, Any
from tools.data_extraction import extract_key_values


# --- Configuration ---
MODEL_PATH = "tools/audio/vosk-model-small-es-0.42"
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096

# --- Grammar for numbers ---
# Keywords must align with those expected by extract_final_receipt_values
KEYWORDS_FOR_EXTRACTION = ["SUB", "IMPUESTO", "TOTAL", "VENTA"]

grammar = KEYWORDS_FOR_EXTRACTION + [
    "[unk]",
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
    "punto", "coma"
]
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
    "coma": "."
}

def format_transcription(result_text: str) -> str:
    """Converts transcribed words to formatted text with digits."""
    split_words = result_text.split()
    formatted_text = ""
    for word in split_words:
        # Use digit_map for conversion, inserting spaces around non-mapped words
        formatted_text += digit_map.get(word, f" {word} ") 
    return formatted_text.strip() # Remove any leading/trailing spaces

def transcribe_and_extract_data(model_path: str, keywords: List[str]) -> Dict[str, Any]:
    """
    Initializes Vosk, runs audio streaming transcription, and returns a dictionary 
    of extracted key-value pairs using the external data extraction module.
    """
    if not os.path.exists(model_path):
        print(f"Error: Vosk model not found at {model_path}")
        return {}

    p = None
    stream = None
    full_transcription = []

    try:
        # 1. Initialize Vosk model
        model = Model(model_path)
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
            if not data:
                break

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                result_text = result.get("text", "")

                if result_text:
                    formatted_text = format_transcription(result_text)
                    print(f"Segment: {formatted_text}")
                    # Accumulate all formatted words for final extraction
                    full_transcription.extend(formatted_text.split())

    except KeyboardInterrupt:
        print("\nStopping recognition.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # 4. Clean up resources and get final segment
        if 'recognizer' in locals():
            final_result = json.loads(recognizer.FinalResult())
            if final_result.get("text"):
                formatted_cleanup = format_transcription(final_result['text'])
                print(f"Cleanup: {formatted_cleanup}")
                full_transcription.extend(formatted_cleanup.split())

        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()
        if p:
            p.terminate()
            print("PyAudio closed")

    # 5. Extract data using the external module
    if full_transcription:
        print("\nRunning data extraction on full transcript...")
        # Call the external function
        extracted_data = extract_key_values(full_transcription, keywords)
        return extracted_data

    return {}

# --- Example of running the function ---
if __name__ == "__main__":
    final_extracted_dict = transcribe_and_extract_data(MODEL_PATH, KEYWORDS_FOR_EXTRACTION)

    print("\n==================================")
    print("FINAL EXTRACTED DATA DICTIONARY:")
    print(json.dumps(final_extracted_dict, indent=4))
    print("==================================")
