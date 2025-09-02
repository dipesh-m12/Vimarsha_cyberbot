import os
import logging
from datetime import datetime
import aiofiles
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from pathlib import Path
from threat_detector import classify_threat
import json

from transcriber import transcribe_audio

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot configuration
BOT_TOKEN = "7651620398:AAHWZyLY9_wEdIaLe5reCj6S6ZDwvL5Fb68"

# Create directories for storing different media types
MEDIA_DIRS = {
    'images': 'media/images',
    'videos': 'media/videos', 
    'audio': 'media/audio',
    'voice': 'media/voice',
    'documents': 'media/documents',
    'text': 'media/text'
}

# Create media directories
for dir_path in MEDIA_DIRS.values():
    Path(dir_path).mkdir(parents=True, exist_ok=True)


# Wrapper for threat detection to match expected format
def detect_threat(text):
    """
    Wrapper for classify_threat to return expected format.
    
    Args:
        text (str): Text to classify.
    
    Returns:
        dict: Dictionary with threat_type, confidence, risk_level, is_threat.
    """
    # Call classify_threat
    result_json = classify_threat(text)
    result = json.loads(result_json)[0]  # Get first result (single text)
    
    threat_type = result['predicted']
    confidence_scores = result['confidence_scores']
    confidence = max(confidence_scores.values())  # Max probability
    
    # Determine is_threat and risk_level
    is_threat = threat_type != 'safe'
    
    # Assign risk_level based on category and confidence
    if threat_type in ['terrorism', 'violence']:
        risk_level = 'CRITICAL' if confidence > 0.8 else 'HIGH'
    elif threat_type in ['sexual_offense', 'hate_speech', 'illicit_drugs']:
        risk_level = 'HIGH' if confidence > 0.7 else 'MEDIUM'
    elif threat_type == 'scam':
        risk_level = 'MEDIUM' if confidence > 0.6 else 'LOW'
    else:  # safe
        risk_level = 'NONE'
    
    return {
        'threat_type': threat_type,
        'confidence': confidence,
        'risk_level': risk_level,
        'is_threat': is_threat
    }

class MediaCollector:
    """Handles collection and storage of different media types"""
    
    @staticmethod
    async def download_file(bot, file_id: str, save_path: str) -> bool:
        """Download file from Telegram servers"""
        try:
            file = await bot.get_file(file_id)
            await file.download_to_drive(save_path)
            return True
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return False
    
    @staticmethod
    async def save_text_message(text: str, user_id: int, message_id: int):
        """Save text messages to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{MEDIA_DIRS['text']}/text_{user_id}_{message_id}_{timestamp}.txt"
        
        async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
            await f.write(f"User ID: {user_id}\n")
            await f.write(f"Message ID: {message_id}\n")
            await f.write(f"Timestamp: {timestamp}\n")
            await f.write(f"Content: {text}\n")
        
        return filename

# Bot handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_message = """
🤖 Welcome to the Multimodal Agent Bot with Threat Detection!

I can process and analyze:
📝 Text messages (with threat detection)
🖼️ Images 
🎵 Audio files
🎤 Voice messages  
🎬 Videos
📄 Documents

🛡️ **Threat Detection Active:**
• Illicit drugs detection
• Violence/weapons detection
• Fraud/scam detection  
• Hate speech detection

Commands:
/start - Show this welcome message
/status - Check bot status
    """
    await update.message.reply_text(welcome_message)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    status_message = "🟢 Bot is running with threat detection active!"
    await update.message.reply_text(status_message)

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages with robust threat detection"""
    try:
        user_id = update.effective_user.id
        message_id = update.message.message_id
        text = update.message.text
        
        # Detect threats using classify_threat
        result = detect_threat(text)
        
        # Save text message
        filename = await MediaCollector.save_text_message(text, user_id, message_id)
        
        # ALWAYS show the detected category
        category = result['threat_type']
        confidence = result['confidence']
        risk_level = result['risk_level']
        
        if result['is_threat']:
            # Format threat type for display
            threat_display = category.replace('_', ' ').title()
            
            # Response based on risk level
            if risk_level == 'CRITICAL':
                response = f"🚨 CRITICAL THREAT DETECTED!\n📊 Category: {threat_display}\n⚡ Risk: {risk_level}\n📈 Confidence: {confidence:.2f}\n\n❌ This content violates our terms of service."
                
            elif risk_level == 'HIGH':
                response = f"⚠️ HIGH RISK CONTENT!\n📊 Category: {threat_display}\n⚡ Risk: {risk_level}\n📈 Confidence: {confidence:.2f}\n\n⚠️ Please follow community guidelines."
                
            elif risk_level == 'MEDIUM':
                response = f"⚠️ MEDIUM RISK DETECTED\n📊 Category: {threat_display}\n⚡ Risk: {risk_level}\n📈 Confidence: {confidence:.2f}\n\n💡 Please be mindful of your language."
                
            else:  # LOW
                response = f"⚠️ Low Risk Content\n📊 Category: {threat_display}\n⚡ Risk: {risk_level}\n📈 Confidence: {confidence:.2f}\n\n✅ Content flagged for review."
            
            # Log detailed threat info
            logger.warning(f"THREAT DETECTED - User: {user_id}, Category: {category}, Risk: {risk_level}, Confidence: {confidence:.2f}, Text: {text[:100]}")
            
        else:
            # Safe content - still show analysis
            response = f"✅ CONTENT SAFE\n📊 Category: {category}\n⚡ Risk: {risk_level}\n📈 Analysis Complete\n\n✅ Message processed successfully!"
            
            logger.info(f"SAFE CONTENT - User: {user_id}, Category: {category}")
        
        await update.message.reply_text(response)
        logger.info(f"Text message saved: {filename}")
        
    except Exception as e:
        logger.error(f"Error handling text message: {e}")
        await update.message.reply_text("❌ Error processing text message")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo messages"""
    try:
        user_id = update.effective_user.id
        message_id = update.message.message_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get the largest photo size
        photo = update.message.photo[-1]
        file_id = photo.file_id
        
        # Create filename
        filename = f"{MEDIA_DIRS['images']}/photo_{user_id}_{message_id}_{timestamp}.jpg"
        
        # Download and save photo
        success = await MediaCollector.download_file(context.bot, file_id, filename)
        
        if success:
            logger.info(f"Photo saved: {filename}")
            await update.message.reply_text("📸 Photo received and saved!")
        else:
            await update.message.reply_text("❌ Error saving photo")
            
    except Exception as e:
        logger.error(f"Error handling photo: {e}")
        await update.message.reply_text("❌ Error processing photo")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages with transcription and robust threat detection"""
    try:
        user_id = update.effective_user.id
        message_id = update.message.message_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        voice = update.message.voice
        file_id = voice.file_id
        duration = voice.duration
        
        # Create filename for voice file
        filename = f"{MEDIA_DIRS['voice']}/voice_{user_id}_{message_id}_{timestamp}.ogg"
        
        # Download and save voice
        success = await MediaCollector.download_file(context.bot, file_id, filename)
        
        if not success:
            logger.error(f"Failed to save voice message: {filename}")
            await update.message.reply_text("❌ Error saving voice message")
            return
        
        logger.info(f"Voice message saved: {filename} (Duration: {duration}s)")
        
        # Transcribe voice message
        try:
            transcribed_text = transcribe_audio(filename)
            logger.info(f"Voice message transcribed: {transcribed_text[:100]}...")
            
            # Save transcribed text
            text_filename = await MediaCollector.save_text_message(transcribed_text, user_id, message_id)
            logger.info(f"Transcribed text saved: {text_filename}")
            
            # Detect threats using classify_threat
            result = detect_threat(transcribed_text)
            
            # Extract detection results
            category = result['threat_type']
            confidence = result['confidence']
            risk_level = result['risk_level']
            
            if result['is_threat']:
                # Format threat type for display
                threat_display = category.replace('_', ' ').title()
                
                # Response based on risk level
                if risk_level == 'CRITICAL':
                    response = f"🚨 CRITICAL THREAT DETECTED IN VOICE MESSAGE!\n📊 Category: {threat_display}\n⚡ Risk: {risk_level}\n📈 Confidence: {confidence:.2f}\n\n❌ This content violates our terms of service."
                elif risk_level == 'HIGH':
                    response = f"⚠️ HIGH RISK CONTENT IN VOICE MESSAGE!\n📊 Category: {threat_display}\n⚡ Risk: {risk_level}\n📈 Confidence: {confidence:.2f}\n\n⚠️ Please follow community guidelines."
                elif risk_level == 'MEDIUM':
                    response = f"⚠️ MEDIUM RISK DETECTED IN VOICE MESSAGE\n📊 Category: {threat_display}\n⚡ Risk: {risk_level}\n📈 Confidence: {confidence:.2f}\n\n💡 Please be mindful of your language."
                else:  # LOW
                    response = f"⚠️ Low Risk Content in Voice Message\n📊 Category: {threat_display}\n⚡ Risk: {risk_level}\n📈 Confidence: {confidence:.2f}\n\n✅ Content flagged for review."
                
                # Log detailed threat info
                logger.warning(f"THREAT DETECTED IN VOICE - User: {user_id}, Category: {category}, Risk: {risk_level}, Confidence: {confidence:.2f}, Text: {transcribed_text[:100]}")
            
            else:
                # Safe content - still show analysis
                response = f"✅ VOICE MESSAGE SAFE\n📊 Category: {category}\n⚡ Risk: {risk_level}\n📈 Analysis Complete\n\n✅ Voice message processed successfully!"
                
                logger.info(f"SAFE VOICE CONTENT - User: {user_id}, Category: {category}")
            
            await update.message.reply_text(f"🎤 Voice message transcribed and analyzed!\n{response}")
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            await update.message.reply_text("❌ Error transcribing voice message")
        
    except Exception as e:
        logger.error(f"Error handling voice message: {e}")
        await update.message.reply_text("❌ Error processing voice message")

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle video messages"""
    try:
        user_id = update.effective_user.id
        message_id = update.message.message_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        video = update.message.video
        file_id = video.file_id
        duration = video.duration
        
        # Create filename
        filename = f"{MEDIA_DIRS['videos']}/video_{user_id}_{message_id}_{timestamp}.mp4"
        
        # Download and save video
        success = await MediaCollector.download_file(context.bot, file_id, filename)
        
        if success:
            logger.info(f"Video saved: {filename} (Duration: {duration}s)")
            await update.message.reply_text("🎬 Video received and saved!")
        else:
            await update.message.reply_text("❌ Error saving video")
            
    except Exception as e:
        logger.error(f"Error handling video: {e}")
        await update.message.reply_text("❌ Error processing video")

def main():
    """Main function to run the bot"""
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    
    # Add message handlers for different media types
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    
    # Start bot
    logger.info("Starting Multimodal Telegram Bot with Threat Detection...")
    print("🚀 Bot is starting...")
    print("🛡️ Threat detection loaded!")
    print("📁 Media directories created:")
    for media_type, path in MEDIA_DIRS.items():
        print(f"   {media_type}: {path}")
    print("✅ Bot is ready!")
    
    # Run the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()