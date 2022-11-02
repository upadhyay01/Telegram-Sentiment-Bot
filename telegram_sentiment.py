
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from prediction import predict

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)
logger = logging.getLogger(__name__)

def start(update,context):
    update.message.reply_text("Lets begin the word war shall we?")

def write_file(update,context):
    with open("textfile.txt","a+") as tf:
        tf.write(update.message.text+",")   
    
def pred(update,context):
    op=open("textfile.txt","r")
    op_f=op.read()
    print(predict(op_f))
    update.message.reply_text(predict(op_f))

    
def error(update,context):
    logger.warning('Update "%s" caused "%s" error',update,context.error)
    
def main():
    Token="5462812124:AAHzZGLTjHXXXXXXXXXXXX"
    updater = Updater(Token, use_context = True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start",start))
    dp.add_handler(CommandHandler("pred",pred))
    dp.add_handler(MessageHandler(Filters.text,write_file))
    
    dp.add_error_handler(error)
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
    
#{'Toxic': 0.03447449, 'Severely Toxic': 0.00231676, 'Obscene': 0.009034775, 'Threat': 0.0030774085, 'Insult': 0.011153859, 'Identity Hate': 0.003922691}    
#"{'Toxic': 0.07052451, 'Severely Toxic': 0.0018840205, 'Obscene': 0.016963858, 'Threat': 0.0015996258, 'Insult': 0.018414717, 'Identity Hate': 0.0034141382}"
