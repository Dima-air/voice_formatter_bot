# bot/telegram_bot.py
import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
import torch

TELEGRAM_TOKEN = ""
WHISPER_MODEL_PATH = "C:/Users/User/PycharmProjects/models/whisper-small-ru-final"
NLU_MODEL_PATH = "C:/Users/User/PycharmProjects/models/nlu_model"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Глобальные модели
asr_model = None
asr_processor = None
nlu_model = None
formatter = None

def load_models():
    global asr_model, asr_processor, nlu_model, formatter
    if asr_model is None:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        logger.info("Загрузка дообученной Whisper..")
        asr_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
        asr_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH)
        asr_model.eval()
    if nlu_model is None:
        from nlu.nlu_model import NLUModel
        logger.info("Загрузка NLU-модели..")
        nlu_model = NLUModel(NLU_MODEL_PATH)
    if formatter is None:
        class TextFormatter:
            @staticmethod
            def apply_formatting(text, entity, intent):
                if intent == "format_bold":
                    return text.replace(entity, f"<b>{entity}</b>", 1)
                elif intent == "format_italic":
                    return text.replace(entity, f"<i>{entity}</i>", 1)
                elif intent == "format_header":
                    return f"<b><u>{entity}</u></b>\n" + text.replace(entity, "", 1)
                else:
                    return text
        formatter = TextFormatter()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    keyboard = [[InlineKeyboardButton("🎙️ Отправить голосовое", callback_data="record_voice")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        f"Привет, {user.first_name}! 🎙️\n\n"
        "Отправьте мне голосовое сообщение с командой, например:\n"
        "• «Сделай жирным слово привет»\n"
        "• «Курсивом напиши это важно»\n"
        "• «Заголовок для раздела контакты»",
        reply_markup=reply_markup
    )

async def record_voice_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    await update.callback_query.edit_message_text("🎙️ Отправьте голосовое сообщение.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Голосовое сообщение получено...")
    load_models()
    user = update.effective_user
    voice = update.message.voice

    try:
        #скачиваем аудио
        file = await context.bot.get_file(voice.file_id)
        await file.download_to_drive("temp_voice.ogg")

        #загружаем аудио
        import librosa
        audio_array, sr = librosa.load("temp_voice.ogg", sr=16000)

        #ASR
        inputs = asr_processor(audio_array, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = asr_model.generate(
                inputs["input_features"],
                attention_mask=inputs.get("attention_mask")
            )
        raw_text = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        #NLU
        intent, entity = nlu_model.predict(raw_text)

        #форматирование
        formatted_text = formatter.apply_formatting(raw_text, entity, intent)

        await update.message.reply_text(
            f"Распознано: <i>{raw_text}</i>\n\n"
            f" Результат:\n{formatted_text}",
            parse_mode="HTML"
        )

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await update.message.reply_text("Произошла ошибка при обработке голосового.")

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(record_voice_button, pattern="^record_voice$"))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logger.info("Бот запущен")
    application.run_polling()

if __name__ == "__main__":
    main()