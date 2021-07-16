import telebot
import traceback
from tg import config
from .handler import *
from src.connections import connections

bot = telebot.TeleBot(config.HTTP_TOKEN)

def get_photo(message):
    photo = message.photo[1].file_id
    file_info = bot.get_file(photo)
    file_content = bot.download_file(file_info.file_path)
    return file_content

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, config.START_MESSAGE)

@bot.message_handler(content_types=['photo'])
def repeat_all_messages(message):
    try:
        file_content = get_photo(message)
        print('file_content')
        image = byte2numpy(file_content)
        print('image')
        backup(message, image)
        print('backup')
        ok, rectangles, landmarks, facemarks = get_features(image)
        if not ok:
            bot.send_message(message.chat.id, 'Ничего не нашел!')
        else:   
            biometrics = get_face_biometrics(facemarks)
            faces = get_face_images(image, rectangles, face_images_transform)
            # expressions = get_face_expression(biometrics, predictor_fer_fc)
            # expressions = get_face_expression(faces, predictor_fer_cnn)
            expressions = get_face_expression((faces, biometrics), predictor_fer_cascade)
            print(expressions)
            print_faces(image, rectangles, landmarks, expressions, mesh=True)

            bio = numpy2byte(image)
            bot.send_photo(message.chat.id, photo=bio)
    
    except Exception:
        traceback.print_exc()
        bot.send_message(message.chat.id, config.ERROR_MESSAGE)

if __name__ == '__main__':
    import time
    while True:
        try:
            bot.polling(none_stop=True)
        except Exception as e:
            time.sleep(15)
            print('Restart!')
