from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_name = update.message.from_user.first_name
    await update.message.reply_text(
        f"안녕하세요 {user_name}님! 당신의 채팅 ID는 {chat_id} 입니다.\n"
        f"이 ID를 config.yaml 파일에 입력하세요."
    )

def main():
    token = input("BotFather에게서 받은 API 토큰을 입력하세요: ")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    
    print("✅ 봇이 시작되었습니다. 텔레그램에서 봇에게 /start 메시지를 보내보세요.")
    print("❌ 종료하려면 Ctrl + C 를 누르세요.")
    app.run_polling()

if __name__ == "__main__":
    main()
