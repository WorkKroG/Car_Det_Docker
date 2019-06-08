import logging 
from telegram.ext import MessageHandler, Filters 
from telegram.ext import Updater 
import os 
from os import system
import os.path 
import datetime 
import sys 
updater = Updater(token= '694223697:AAGwBLzwFaSs3fciTX0l4WZgLXJ5XQB30y4') 
from telegram.ext import Updater, CommandHandler 
import telebot
 


def start(bot, update):
    print('Bot has started polling')
    update.message.reply_text('Hello {}:'.format(update.message.from_user.first_name))

    

def hello(bot, update):
    print('Starting conversation')
    #update.message.reply_text('Your photo {}:'.format(update.message.from_user.first_name))
    chat_id = bot.get_updates()[-1].message.chat_id
    #bot.send_photo(chat_id, open("C:/Program1/Projects/Python/Car-Segmentation/test_1_yolo_out_py.jpg", 'rb')) 
    bot.send_photo(chat_id=chat_id, photo=open('C:/Program1/Projects/Python/Car-Segmentation/test_1_yolo_out_py.png', 'rb'))
    #photo = open('test_1_yolo_out_py.png', 'rb')
    #bot.send_photo(chat_id, photo)
    #bot.send_photo(chat_id, "FILEID")
    update.message.reply_text('Your photo {}:'.format(update.message.from_user.first_name))

def photo(bot, update):
    print('Took photo')
    test = 'test_'
    numb = 1
    save_path = 'C:/Program1/Projects/Python/Car-Segmentation' 
    file_id = update.message.photo[-1].file_id 
    newFile = bot.getFile(file_id) 
    newFile.download(os.path.join(save_path, test + str(numb) +'.jpg')) 
    bot.sendMessage(chat_id=update.message.chat_id, text="download succesful") 
    filename = ( test + str(numb+1) +'.jpg')

    print('starting serving photo')
    import car_detection_yolo
    print('photo is ready')

    with open(filename,"rb") as f: 
     Jpegcontents = (f.read()) 
     if Jpegcontents.startswith(b"\xff\xd8") and Jpegcontents.endswith(b"\xff\xd9"): 
      bot.sendMessage(chat_id=update.message.chat_id, text="Valid Image") 
     if not Jpegcontents.startswith(b"\xff\xd8") and Jpegcontents.endswith(b"\xff\xd9"): 
      os.system("rm ",filename)

print('Initializing')

photo_handler = MessageHandler(Filters.photo, photo) 
updater.dispatcher.add_handler(photo_handler) 
updater.dispatcher.add_handler(CommandHandler('start', start)) 
updater.dispatcher.add_handler(CommandHandler('get', hello)) 

print('Car detector is ready')

updater.start_polling()
updater.idle()
updater.stop()
