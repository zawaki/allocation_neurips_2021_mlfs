class GlobalMessageBuffer:
    '''
    General: Super class to allow network-component and network-request
    communication via a global r/w message buffer.
    '''
    global_message_buffer = []

    def send_message(self,message):
            self.global_message_buffer.append(message)