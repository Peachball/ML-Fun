import Skype4Py as skpy

print

def responder(Message, Status):
    if Status == 'RECEIVED':
        msg = Message.Body.lower()
        Message.Chat.SendMessage('sure')

skype = skpy.Skype()

skype.OnMessageStatus = responder

if skype.Client.IsRunning == False:
    skype.Client.Start()
skype.Attach()

for chat in skype.Chats:
    print chat.Name

while True:
	input('running')
