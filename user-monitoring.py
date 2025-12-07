from telethon import TelegramClient, events
from telethon.tl.functions.channels import CreateChannelRequest, UpdateUsernameRequest
import asyncio
import re

API_ID = 28314907
API_HASH = '01bffe95d8af217cf794e66d989c227e'
PHONE_NUMBER = '+998774490625'

TARGET_USERNAME = 'lockpy'


async def main():
    client = TelegramClient('session_name', API_ID, API_HASH)
    await client.start(PHONE_NUMBER)

    async def check_and_take_username():
        while True:
            try:
                entity = await client.get_entity(TARGET_USERNAME)
                print(f"{TARGET_USERNAME} hali band. Qayta tekshirilmoqda...")
            except Exception as e:
                if "Could not find" in str(e) or "No user has" in str(e):
                    print(f"{TARGET_USERNAME} bo'sh! Kanol ochilmoqda...")

                    channel = await client(CreateChannelRequest(
                        title=f"{TARGET_USERNAME} kanali",
                        about="Avtomatik ochilgan kanal",
                        megagroup=False
                    ))

                    await client(UpdateUsernameRequest(
                        channel.chats[0].id,
                        TARGET_USERNAME
                    ))

                    print(f"🎉 {TARGET_USERNAME} usernameni muvaffaqiyatli egallab olindi!")
                    break
                else:
                    print(f"Xatolik: {e}")

            await asyncio.sleep(1)

    await check_and_take_username()
    await client.disconnect()


if __name__ == '__main__':
    asyncio.run(main())