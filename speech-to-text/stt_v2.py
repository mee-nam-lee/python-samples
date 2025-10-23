

import time
import os
import wave
import argparse
import csv
import re
import string
import wordninja
from kiwipiepy import Kiwi

from google.cloud import speech_v2
from google.api_core.client_options import ClientOptions


client_us = speech_v2.SpeechClient(client_options=ClientOptions(
            api_endpoint="us-speech.googleapis.com",
        ))

client_global = speech_v2.SpeechClient(client_options=ClientOptions(
            api_endpoint="speech.googleapis.com",
        ))
client_us_central = speech_v2.SpeechClient(client_options=ClientOptions(
            api_endpoint="us-central1-speech.googleapis.com",
        ))

PROJECT_ID="YOUR-PROJECT_ID"

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def calculate_cer(reference, hypothesis):
    """Calculates Character Error Rate (CER)"""
    if not reference:
        return 1.0 if hypothesis else 0.0
    distance = levenshtein_distance(reference, hypothesis)
    return distance / len(reference)

def calculate_wer(reference, hypothesis):
    """Calculates Word Error Rate (WER)"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)

def normalize_mixed_language(text: str, kiwi: Kiwi) -> str:
    # Remove quotes and commas first
    text = text.replace('"', '').replace("'", '').replace(",", '').replace("?", "")

    text_no_space = text.replace(" ", "")
    pattern = re.compile(r'([a-zA-Z]+|[가-힣]+)')
    
    parts = []
    last_end = 0
    
    for match in pattern.finditer(text_no_space):
        start, end = match.span()
        
        if start > last_end:
            parts.append(text_no_space[last_end:start])
            
        word = match.group(0)
        if re.match(r'^[a-zA-Z]+', word):
            word = word.lower()
            parts.append(' '.join(wordninja.split(word)))
        else:
            parts.append(kiwi.space(word))
        
        last_end = end
        
    if last_end < len(text_no_space):
        parts.append(text_no_space[last_end:])
        
    result = ' '.join(parts)
    return ' '.join(result.split())

def transcribe_stt_v2(audio_file: str, model:str = "chirp_3"):
    client = client_global
    recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/_"
    
    if model == "chirp_3":
        client = client_us
        recognizer=f"projects/{PROJECT_ID}/locations/us/recognizers/_"
    elif model == "chirp_2" or model == "chirp_telephony":
         client = client_us_central
         recognizer=f"projects/{PROJECT_ID}/locations/us-central1/recognizers/_"
    
    features = speech_v2.RecognitionFeatures(
        enable_spoken_punctuation=False,
        enable_automatic_punctuation=False
    )

    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        language_codes=["ko-KR", "en-US"],
        features=features,
        model=model,
    )
    
    with open(audio_file, "rb") as f:
            audio_content = f.read()

    request = speech_v2.RecognizeRequest(
        content=audio_content,
        recognizer=recognizer,
        config=config
    )

    start_time = time.time()
    response = client.recognize(request=request)
    end_time = time.time() 
    elapsed_time = end_time - start_time
    
    transcript = " ".join(
        [result.alternatives[0].transcript for result in response.results]
    )

    transcript = transcript.strip().rstrip(string.punctuation)

    return elapsed_time, transcript

def main(model: str, start_index: int, end_index: int, output_filename: str):
    ground_truth = {}
    with open('audio_truth.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                try:
                    ground_truth[int(row[0])] = row[1]
                except (ValueError, IndexError):
                    print(f"Skipping invalid row in CSV: {row}")
    
    kiwi = Kiwi()
    total_cer = 0
    total_wer = 0
    file_count = 0

    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        header = ["File", "WER", "CER", "Original Reference", "Original Transcript", "Normalized Reference", "Normalized Transcript"]
        csv_writer.writerow(header)
        print(",".join(header))

        for i in range(start_index, end_index + 1):
            audio_file = f"./audio/{i}.wav"
            if not os.path.exists(audio_file):
                continue

            if i not in ground_truth:
                continue

            elapsed_time, hypothesis = transcribe_stt_v2(audio_file, model)
            reference = ground_truth[i]

            reference_normalized = normalize_mixed_language(reference, kiwi)
            hypothesis_normalized = normalize_mixed_language(hypothesis, kiwi)
            
            cer = calculate_cer(reference_normalized, hypothesis_normalized)
            wer = calculate_wer(reference_normalized, hypothesis_normalized)
            
            total_cer += cer
            total_wer += wer
            file_count += 1
            
            row = [os.path.basename(audio_file), f"{wer:.4f}", f"{cer:.4f}", reference, hypothesis, reference_normalized, hypothesis_normalized]
            csv_writer.writerow(row)
            print(",".join(row))

    if file_count > 0:
        avg_cer = total_cer / file_count
        avg_wer = total_wer / file_count
        print("\n" + "="*30)
        print(f"Average CER: {avg_cer:.4f}")
        print(f"Average WER: {avg_wer:.4f}")
        print("="*30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files and calculate CER/WER.")
    parser.add_argument("--model", type=str, default="chirp_3", help="The speech recognition model to use.")
    parser.add_argument("--start_index", type=int, default=1, help="Start index of files to process.")
    parser.add_argument("--end_index", type=int, default=52, help="End index of files to process.")
    args = parser.parse_args()
    
    output_filename = f"{args.model}.csv"
    
    main(
        model=args.model, 
        start_index=args.start_index, 
        end_index=args.end_index,
        output_filename=output_filename
    )
        