import telebot
import traceback
import config
from PIL import Image, ImageOps
import tensorflow as tf

bot = telebot.TeleBot(config.TOKEN)

classes = ['съедобный', 'несъедобный']

model = tf.keras.models.load_model('mushrooms_final.h5')


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id,
                     'Привет! Пришли фото сюда, а нейронная сеть определит что это за вид цветка)))')


@bot.message_handler(content_types=['photo'])
def repeat_all_messages(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open("image.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)

        image = Image.open("image.jpg")
        size = (180, 180)
        image = image.convert("RGB")
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array).flatten()
        predictions = tf.where(predictions < 0.5, 0, 1)
        predicted_class_index = tf.argmax(predictions)

        bot.send_message(message.chat.id, text=f'Этот гриб скорее всего {classes[predicted_class_index]}')


    except Exception as e:
        traceback.print_exc()
        bot.send_message(message.chat.id, 'Что-то пошло не по плану!')

bot.polling(none_stop=True)