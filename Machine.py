import pickle
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import telebot
from kandinsky2 import get_kandinsky2
import translators as ts

# Load Kandinsky2 model
model1 = get_kandinsky2('cuda', task_type='text2img', cache_dir='/tmp/kandinsky2', model_version='2.1', use_flash_attention=False)

# Load character mappings
with open('idx_to_char.pickle', 'rb') as f:
    idx_to_char = pickle.load(f)
with open('char_to_idx.pickle', 'rb') as f:
    char_to_idx = pickle.load(f)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load LSTM model
class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers, dropout=0.3 if n_layers > 1 else 0)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))

model = torch.load('model.pt', map_location=device)
model.load_state_dict(torch.load('modelall.pt', map_location=device))
model.eval()

# Text generation function
def evaluate(model, char_to_idx, idx_to_char, start_text='.', prediction_len=60, temp=0.5):
    model.eval()
    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    with torch.no_grad():
        _, hidden = model(train, hidden)
        inp = train[-1].view(-1, 1, 1)
        for _ in range(prediction_len):
            output, hidden = model(inp.to(device), hidden)
            output_logits = output.cpu().data.view(-1)
            p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().numpy()
            top_index = np.random.choice(len(char_to_idx), p=p_next)
            inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
            predicted_text += idx_to_char[top_index]
    return predicted_text

def generate_text(user_text):
    user_text = user_text.lower() + ','
    ai_text = evaluate(model, char_to_idx, idx_to_char, start_text=user_text, prediction_len=60, temp=0.5)
    end_text = re.sub("[^А-Яа-я0-9\s!-~ёЁ]", "", ai_text)
    end_text = re.sub("^\s+|\n|\r|\s+$", '', end_text)
    translated_text = ts.translate_text(end_text, from_language='ru', to_language='en')
    return end_text, translated_text

# Telegram bot setup
token = '***'
bot = telebot.TeleBot(token)
user_resolution = {}  # Store user-selected resolution

@bot.message_handler(commands=['start'])
def start(message):
    user_resolution[message.chat.id] = 512  # Default resolution
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add("Генерация изображения", "Размер изображения", "/help")
    bot.send_message(message.chat.id, "Привет! Отправь мне текст, и я создам изображение по этому тексту.", reply_markup=keyboard)

@bot.message_handler(commands=['help'])
def help_command(message):
    bot.send_message(message.chat.id, "/start — запустить бот\n/generate <текст> — сгенерировать изображение\nРазмер изображения — выбрать разрешение")

@bot.message_handler(content_types=['text'])
def handle_text(message):
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add("Генерация изображения", "Размер изображения", "/help")
    text = message.text.lower()

    if text == "генерация изображения":
        bot.send_message(message.chat.id, "Напиши любое слово или фразу.", reply_markup=keyboard)
        bot.register_next_step_handler(message, generate_image)
    elif text == "размер изображения":
        keyboard1 = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        keyboard1.add("512x512", "512x768", "Назад")
        bot.send_message(message.chat.id, "Выбери разрешение изображения.", reply_markup=keyboard1)
    elif text == "512x512":
        user_resolution[message.chat.id] = 512
        bot.send_message(message.chat.id, "Установлено разрешение 512x512. Напиши слово для генерации.", reply_markup=keyboard)
    elif text == "512x768":
        user_resolution[message.chat.id] = 768
        bot.send_message(message.chat.id, "Установлено разрешение 512x768. Напиши слово для генерации.", reply_markup=keyboard)
    elif text == "назад":
        bot.send_message(message.chat.id, "Отправь текст для генерации изображения.", reply_markup=keyboard)
    elif text.startswith('/generate'):
        query = text.replace('/generate', '').strip()
        if query:
            generate_image(message, query)
        else:
            bot.send_message(message.chat.id, "Укажи текст после /generate, например: /generate стакан", reply_markup=keyboard)
    elif text == '/mtuci':
        bot.send_message(message.chat.id, "Официальный сайт: https://mtuci.ru")
    else:
        bot.send_message(message.chat.id, "Извините, я вас не понял. Попробуйте команду /generate или выберите действие.", reply_markup=keyboard)

def generate_image(message, query=None):
    try:
        bot.send_message(message.chat.id, "Генерирую...")
        user_text = query if query else message.text.lower()
        if not user_text:
            bot.send_message(message.chat.id, "Пожалуйста, укажите текст для генерации.")
            return

        end_text, translated_text = generate_text(user_text)
        bot.send_message(message.chat.id, f"Текст запроса: {ts.translate_text(translated_text, from_language='en', to_language='ru')}\nСгенерированный текст: {end_text}")

        images = model1.generate_text2img(
            translated_text, num_steps=40, batch_size=1, guidance_scale=4,
            h=user_resolution[message.chat.id], w=512, sampler='p_sampler',
            prior_cf_scale=4, prior_steps="5"
        )
        bot.send_photo(message.chat.id, images[0])
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка генерации: {str(e)}. Попробуйте снова.")

bot.infinity_polling()