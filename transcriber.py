import assemblyai as aai

def transcribe_audio(audio_file: str) -> str:
    """
    Transcribes an audio file using AssemblyAI.
    
    Args:
        audio_file (str): Path to the audio file to transcribe.
        
    Returns:
        str: Transcribed text from the audio file.
        
    Raises:
        RuntimeError: If transcription fails.
    """
    aai.settings.api_key = "8d1b9a35017445d38d6d6c409f5827c2"
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal)
    transcript = aai.Transcriber(config=config).transcribe(audio_file)
    
    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")
    
    return transcript.text