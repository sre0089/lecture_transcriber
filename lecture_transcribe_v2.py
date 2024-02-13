import cv2
import speech_recognition as sr
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_transcript(video_file):
    # Open the video file
    video_capture = cv2.VideoCapture(video_file)
    
    # Initialize variables
    transcript = ""
    
    # Initialize speech recognizer
    recognizer = sr.Recognizer()
    
    # Loop through video frames
    while True:
        # Read frame
        success, frame = video_capture.read()
        if not success:
            break
        
        # Convert frame to audio
        audio = recognizer.record(frame)
        
        # Transcribe audio to text
        try:
            text = recognizer.recognize_google(audio)
            transcript += " " + text
        except sr.UnknownValueError:
            continue
    
    # Release the video capture object
    video_capture.release()
    
    return transcript

def preprocess_text(text):
    # Tokenization
    sentences = sent_tokenize(text)
    
    # Remove stop words and punctuation
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    preprocessed_sentences = []
    for sentence in sentences:
        words = [ps.stem(word) for word in sentence.split() if word.lower() not in stop_words]
        preprocessed_sentences.append(" ".join(words))
    
    return preprocessed_sentences

def generate_summary(text):
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_text)
    
    # Calculate cosine similarity matrix
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Generate summary based on similarity scores
    summary = []
    for i in range(len(cosine_similarities)):
        sentence_scores = list(enumerate(cosine_similarities[i]))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentence_index = sentence_scores[1][0]  # Exclude the first sentence (itself)
        summary.append(preprocessed_text[top_sentence_index])
    
    return ". ".join(summary)

def generate_notes(text):
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Generate notes
    notes = []
    for sentence in sentences:
        # Identify key points using NLP techniques (e.g., named entity recognition, keyword extraction)
        # Append key points to notes list
        notes.append(sentence)
    
    return notes

def main():
    # Input video file
    video_file = input("Enter the path or filename of the video to process: ")
    
    # Extract transcript from video
    transcript = extract_transcript(video_file)
    
    # Generate summary
    summary = generate_summary(transcript)
    
    # Generate notes
    notes = generate_notes(transcript)
    
    # Output transcript, summary, and notes
    print("Transcript:")    
    print(transcript)
    print("\nSummary:")
    print(summary)
    print("\nNotes:")
    for i, note in enumerate(notes, 1):
        print(f"{i}. {note}")

if __name__ == "__main__":
    main()
