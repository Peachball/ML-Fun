import Skype4Py as skpy
from chatterbot import ChatBot

chatTestBot = ChatBot("Joe Odle")

#Learns from responses
msglist = []
def responder(Message, Status):
    if Status == 'RECEIVED':
        msg = Message.Body.lower()
        msglist.append(msg)
        response = chatTestBot.get_response(msg)
        Message.Chat.SendMessage(response)
        msglist.append(response)
        if len(msglist) > 3:
            chatTestBot.train(msglist[-3:-1])
    if Status == 'SENT':
        msg = Message.Body.lower()
        if msg == 'viewmsgs':
            print msglist

skype = skpy.Skype()

skype.OnMessageStatus = responder

if skype.Client.IsRunning == False:
    skype.Client.Start()
skype.Attach()

for chat in skype.Chats:
    print chat.Name
    for m in chat.Messages:
        print m.Body

while True:
    pass
